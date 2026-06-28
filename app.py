import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import warnings
import base64
import io
warnings.filterwarnings('ignore')

# ============================================================
# Load model
# ============================================================
print("Loading model...")
model = tf.keras.models.load_model('pneumonia_cnn_model.keras')
IMG_SIZE = 150

# Warm up: forces full graph trace so output_shape is available
_dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
model.predict(_dummy, verbose=0)
print("Model warmed up.")

# Print every layer so you can hardcode LAST_CONV correctly
print("=" * 60)
print("LAYER LIST - find your final Conv2D name here:")
conv_layers_found = []
for i, layer in enumerate(model.layers):
    is_conv = isinstance(layer, tf.keras.layers.Conv2D)
    tag = "  <-- Conv2D" if is_conv else ""
    print(f"  [{i:02d}] {layer.name:<35} {type(layer).__name__}{tag}")
    if is_conv:
        conv_layers_found.append(layer.name)

print(f"\nAll Conv2D layers: {conv_layers_found}")

# ============================================================
# SET LAST_CONV HERE
# After first deployment, check HuggingFace logs for the layer
# list printed above, then hardcode the final Conv2D name, e.g.:
#   LAST_CONV = "conv2d_5"
# Until you do that, auto-detection is used as a default.
# ============================================================
# Hardcoded from HuggingFace logs - final Conv2D before Flatten/Dense
LAST_CONV = "conv2d_5"
print(f"Grad-CAM target layer: {LAST_CONV}")
print("=" * 60)


# ============================================================
# X-Ray image validator
# ============================================================
def is_xray_image(pil_image):
    """
    Heuristic checks: is this image plausibly a chest X-ray?
    Returns (is_valid: bool, reason: str)
    """
    img_rgb = np.array(pil_image.convert('RGB'), dtype=np.float32)

    # 1. X-rays are grayscale - R/G/B channels should be nearly identical
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    avg_color_diff = (np.mean(np.abs(r - g)) +
                      np.mean(np.abs(r - b)) +
                      np.mean(np.abs(g - b))) / 3.0
    if avg_color_diff > 20:
        return False, (f"Image appears to be a colour photo "
                       f"(colour variance: {avg_color_diff:.1f}). "
                       f"Please upload a grayscale chest X-ray.")

    # 2. X-rays have high contrast
    gray = np.mean(img_rgb, axis=2)
    std_dev = np.std(gray)
    if std_dev < 20:
        return False, (f"Image has very low contrast (std: {std_dev:.1f}). "
                       f"Chest X-rays typically have high contrast.")

    # 3. X-rays are mostly dark
    mean_brightness = np.mean(gray)
    if mean_brightness > 200:
        return False, (f"Image is too bright (mean: {mean_brightness:.1f}). "
                       f"Chest X-rays are typically dark with bright lung regions.")

    # 4. Minimum size
    w, h = pil_image.size
    if w < 64 or h < 64:
        return False, "Image is too small. Please upload a proper chest X-ray."

    return True, "OK"


# ============================================================
# Grad-CAM  (standard TF implementation per architecture spec)
# ============================================================
def generate_gradcam(img_array):
    """
    Grad-CAM for Sequential models.
    Builds a functional sub-model by running a forward pass through
    the Sequential model up to LAST_CONV, then to the output.
    img_array: float32 numpy (1, H, W, 3), values in [0, 1]
    Returns heatmap float32 numpy [0, 1], or None on failure.
    """
    if LAST_CONV is None:
        print("Grad-CAM: LAST_CONV is None - no Conv2D layer found.")
        return None

    # For Sequential models, model.inputs/model.output are only available
    # after the model has been called. We use a tf.keras.Input to rebuild
    # a functional sub-model manually.
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inp
    conv_out_tensor = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == LAST_CONV:
            conv_out_tensor = x

    if conv_out_tensor is None:
        print(f"Grad-CAM: layer {LAST_CONV} not found when rebuilding graph.")
        return None

    grad_model = tf.keras.models.Model(
        inputs  = inp,
        outputs = [conv_out_tensor, x]   # [last_conv_output, final_output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        tape.watch(conv_outputs)
        loss = predictions[:, 0]   # binary sigmoid

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        print("Grad-CAM: tape.gradient returned None.")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]   # (h, w, C)

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap_np = heatmap.numpy()

    # Apply ReLU first to keep only positive activations (regions that
    # push the prediction TOWARD pneumonia, not away from it).
    heatmap_np = np.maximum(heatmap_np, 0)

    hmax = heatmap_np.max()
    if hmax < 1e-8:
        # ReLU killed everything -- gradients were all negative.
        # Fall back: use raw heatmap, take absolute value so at least
        # the most-activated regions show as high attention.
        print("Grad-CAM: ReLU zeroed heatmap, falling back to abs().")
        heatmap_np = np.abs(heatmap.numpy())
        hmax = heatmap_np.max()
        if hmax < 1e-8:
            print("Grad-CAM: heatmap is completely flat.")
            return None

    heatmap_np = heatmap_np / hmax

    print(f"Grad-CAM OK - shape:{heatmap_np.shape} "
          f"min:{heatmap_np.min():.3f} max:{heatmap_np.max():.3f} "
          f"mean:{heatmap_np.mean():.3f}")
    return heatmap_np


def make_gradcam(img_batch, orig_pil):
    """
    Runs Grad-CAM and returns (orig_rgb_uint8, overlay_uint8).
    img_batch: float32 numpy (1, IMG_SIZE, IMG_SIZE, 3), values in [0,1]
    """
    try:
        heatmap = generate_gradcam(img_batch)
        if heatmap is None:
            return None, None

        SIZE = 380
        orig_arr = np.array(orig_pil.convert('RGB').resize((SIZE, SIZE)))

        if orig_arr.ndim == 2:
            orig_rgb = np.stack([orig_arr] * 3, axis=-1)
        elif orig_arr.shape[2] == 4:
            orig_rgb = orig_arr[:, :, :3]
        else:
            orig_rgb = orig_arr.copy()

        heat_rsz = cv2.resize(heatmap, (SIZE, SIZE))
        heat_u8  = np.uint8(255 * heat_rsz)
        heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(
            orig_rgb.astype(np.uint8), 0.5,
            heat_rgb.astype(np.uint8), 0.5, 0
        )

        return orig_rgb.astype(np.uint8), overlay.astype(np.uint8)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"make_gradcam failed: {e}")
        return None, None


# ============================================================
# Helpers
# ============================================================
def arr_to_b64(arr):
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def build_invalid_html(reason):
    return f"""
<div style="font-family:'Inter',system-ui,sans-serif;
            background:#FEF2F2;border-radius:12px;
            border:2px solid #FECACA;padding:36px 28px;
            text-align:center;max-width:100%;">
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
       stroke="#DC2626" stroke-width="1.5"
       style="margin:0 auto 16px;display:block;">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="8" x2="12" y2="12"/>
    <line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
  <p style="font-size:1rem;font-weight:700;color:#DC2626;margin:0 0 8px;">
    Not a Valid Chest X-Ray
  </p>
  <p style="font-size:.875rem;color:#7F1D1D;margin:0 0 16px;line-height:1.6;">
    {reason}
  </p>
  <div style="background:#FEE2E2;border-radius:8px;padding:12px 16px;
              display:inline-block;text-align:left;">
    <p style="font-size:.8rem;color:#991B1B;margin:0;line-height:1.6;">
      <b>Please upload:</b><br>
      - A standard PA or AP chest X-ray<br>
      - Grayscale / black-and-white medical image<br>
      - DICOM-exported JPEG or PNG from a hospital system
    </p>
  </div>
</div>"""


# ============================================================
# Result HTML builder
# ============================================================
def build_result_html(prob, orig_rgb, overlay, show_gc):
    is_pneumonia = prob >= 0.5
    confidence   = prob * 100 if is_pneumonia else (1 - prob) * 100
    conf_str     = f"{confidence:.1f}"

    if is_pneumonia:
        label         = "PNEUMONIA DETECTED"
        badge_bg      = "#DC2626"
        badge_border  = "#B91C1C"
        bar_col       = "#DC2626"
        finding_title = "Pneumonia indicators found"
        bullets = [
            "Increased opacity observed in lung regions",
            "Possible pulmonary consolidation or infiltrates detected",
            "Areas of concern highlighted in the attention map",
        ]
        rec = ("Clinical correlation and evaluation by a qualified "
               "radiologist is strongly recommended.")
    else:
        label         = "NORMAL"
        badge_bg      = "#16A34A"
        badge_border  = "#15803D"
        bar_col       = "#16A34A"
        finding_title = "No significant abnormalities detected"
        bullets = [
            "Lung fields appear clear with no obvious infiltrates",
            "No significant regions of pulmonary opacity identified",
            "Grad-CAM shows diffuse low-intensity attention - normal pattern",
        ]
        rec = ("No immediate follow-up indicated. Continue routine "
               "clinical assessment as advised by your physician.")

    orig_b64 = arr_to_b64(orig_rgb) if orig_rgb is not None else ""
    gc_b64   = arr_to_b64(overlay)  if overlay  is not None else ""

    orig_card = ""
    if orig_b64:
        orig_card = f"""
        <div style="background:white;border-radius:12px;padding:16px;
                    border:1px solid #E5E7EB;box-shadow:0 1px 4px rgba(0,0,0,.05);">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="#2563EB" stroke-width="2">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
            <span style="font-size:12px;font-weight:600;color:#374151;">Original X-Ray</span>
          </div>
          <img src="data:image/png;base64,{orig_b64}"
               style="width:100%;border-radius:8px;display:block;"/>
        </div>"""

    gc_card = ""
    if gc_b64 and show_gc:
        gc_card = f"""
        <div style="background:white;border-radius:12px;padding:16px;
                    border:1px solid #E5E7EB;box-shadow:0 1px 4px rgba(0,0,0,.05);">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="#2563EB" stroke-width="2">
              <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"/>
              <path d="M12 8v4l3 3"/>
            </svg>
            <span style="font-size:12px;font-weight:600;color:#374151;">
              AI Attention Map (Grad-CAM)
            </span>
          </div>
          <img src="data:image/png;base64,{gc_b64}"
               style="width:100%;border-radius:8px;display:block;"/>
          <!-- Colour legend with plain-language labels -->
          <div style="margin-top:10px;">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">
              <div style="width:12px;height:12px;border-radius:3px;
                          background:#00008B;flex-shrink:0;border:1px solid #ccc;"></div>
              <span style="font-size:10.5px;color:#1F2937;font-weight:500;">
                <b style="color:#111827;">Blue</b> = Normal &mdash; AI did not focus here
              </span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
              <div style="width:12px;height:12px;border-radius:3px;
                          background:#FF0000;flex-shrink:0;border:1px solid #ccc;"></div>
              <span style="font-size:10.5px;color:#1F2937;font-weight:500;">
                <b style="color:#111827;">Red</b> = High suspicion &mdash; AI focused here
              </span>
            </div>
            <div style="height:7px;border-radius:999px;
                        background:linear-gradient(to right,
                          #00008B,#0000FF,#00FFFF,#00FF00,#FFFF00,#FF0000);">
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:2px;">
              <span style="font-size:9px;color:#9CA3AF;">Low</span>
              <span style="font-size:9px;color:#9CA3AF;font-weight:600;">
                AI Attention Level
              </span>
              <span style="font-size:9px;color:#9CA3AF;">High</span>
            </div>
          </div>
        </div>"""
    elif show_gc and not gc_b64:
        gc_card = """
        <div style="background:#FFFBEB;border-radius:12px;padding:16px;
                    border:1px solid #FCD34D;">
          <p style="font-size:12px;color:#92400E;margin:0;">
            Grad-CAM could not be generated. Check HuggingFace logs
            and hardcode LAST_CONV to the correct layer name.
          </p>
        </div>"""

    images_section = ""
    if orig_card or gc_card:
        images_section = f"""
      <div class="rg-images">
        {orig_card}
        {gc_card}
      </div>"""

    bullets_html = "".join([f"""
        <div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:6px;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="#2563EB"
               style="margin-top:2px;flex-shrink:0;">
            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <span style="font-size:12.5px;color:#4B5563;line-height:1.5;">{b}</span>
        </div>""" for b in bullets])

    html = f"""
<div style="font-family:'Inter','Segoe UI',system-ui,sans-serif;
            max-width:100%;padding:4px 0;">
  {images_section}
  <div style="background:white;border-radius:12px;padding:18px;
              border:1px solid #E5E7EB;box-shadow:0 1px 4px rgba(0,0,0,.05);
              margin-bottom:14px;">
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:14px;
                padding-bottom:12px;border-bottom:1px solid #F3F4F6;">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
           stroke="#2563EB" stroke-width="2">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
      <span style="font-size:13px;font-weight:700;color:#111827;">
        AI Analysis Results
      </span>
    </div>
    <div class="rg-analysis">
      <div>
        <p style="font-size:11px;font-weight:600;text-transform:uppercase;
                  letter-spacing:.07em;color:#2563EB;margin:0 0 8px;">Diagnosis</p>
        <div style="background:{badge_bg};border:1px solid {badge_border};
                    border-radius:8px;padding:12px 16px;
                    display:flex;align-items:center;gap:10px;margin-bottom:12px;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10
                     10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5
                     1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
          <span style="color:white;font-size:14px;font-weight:700;
                       letter-spacing:.03em;">{label}</span>
        </div>
        <p style="font-size:10px;color:#9CA3AF;margin:0 0 10px;">Model Prediction</p>
        <p style="font-size:11px;font-weight:600;text-transform:uppercase;
                  letter-spacing:.07em;color:#2563EB;margin:0 0 6px;">
          Confidence Score
        </p>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span style="font-size:10px;color:#9CA3AF;">0%</span>
          <span style="font-size:15px;font-weight:700;color:{bar_col};">
            {conf_str}%
          </span>
          <span style="font-size:10px;color:#9CA3AF;">100%</span>
        </div>
        <div style="height:8px;background:#F3F4F6;border-radius:999px;
                    overflow:hidden;margin-bottom:4px;">
          <div style="height:100%;width:{conf_str}%;background:{bar_col};
                      border-radius:999px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;">
          <span style="font-size:10px;color:#9CA3AF;"></span>
          <span style="font-size:10px;color:#9CA3AF;">50%</span>
          <span style="font-size:10px;color:#9CA3AF;"></span>
        </div>
      </div>
      <div>
        <p style="font-size:11px;font-weight:600;text-transform:uppercase;
                  letter-spacing:.07em;color:#2563EB;margin:0 0 8px;">AI Findings</p>
        <p style="font-size:13px;color:#374151;margin:0 0 10px;
                  line-height:1.55;font-weight:500;">{finding_title}</p>
        <div style="margin-bottom:12px;">{bullets_html}</div>
        <div style="background:#FFFBEB;border:1px solid #FCD34D;
                    border-radius:8px;padding:10px 12px;
                    display:flex;gap:8px;align-items:flex-start;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="#F59E0B"
               style="margin-top:1px;flex-shrink:0;">
            <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm1
                     15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
          </svg>
          <p style="font-size:11px;color:#92400E;margin:0;line-height:1.55;">
            <b>Important:</b> {rec}
          </p>
        </div>
      </div>
    </div>
  </div>
  <div style="background:#F0F9FF;border:1px solid #BAE6FD;
              border-radius:8px;padding:10px 14px;">
    <p style="font-size:11px;color:#0C4A6E;margin:0;line-height:1.6;">
      <b>Medical Disclaimer:</b> This tool is for educational and research
      purposes only. It does not constitute medical advice and must not
      replace consultation with a qualified physician or radiologist.
    </p>
  </div>
</div>"""
    return html


# ============================================================
# Main predict function
# ============================================================
def predict(image, show_gradcam):
    if image is None:
        return PLACEHOLDER_HTML

    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))

        # Step 1: Validate it's an X-ray
        valid, reason = is_xray_image(image)
        if not valid:
            return build_invalid_html(reason)

        # Step 2: Preprocess - MUST match training: img / 255.0
        img_rgb   = image.convert('RGB')
        img_rsz   = img_rgb.resize((IMG_SIZE, IMG_SIZE))
        img_arr   = np.array(img_rsz, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_arr, axis=0)   # (1, 150, 150, 3)

        # Step 3: Predict
        prob = float(model.predict(img_batch, verbose=0)[0][0])
        print(f"Sigmoid output: {prob:.6f} -> "
              f"{'PNEUMONIA' if prob >= 0.5 else 'NORMAL'} "
              f"({prob*100:.1f}% pneumonia probability)")

        # Step 4: Grad-CAM
        orig_rgb, overlay = None, None
        if show_gradcam:
            orig_rgb, overlay = make_gradcam(img_batch, img_rgb)
            if orig_rgb is None:
                orig_rgb = np.array(img_rgb.resize((380, 380)))

        return build_result_html(prob, orig_rgb, overlay, show_gradcam)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"""<div style="padding:24px;color:#DC2626;
                               font-family:system-ui;font-size:13px;">
                    <b>Error:</b> {str(e)}
                  </div>"""


# ============================================================
# Placeholder
# ============================================================
PLACEHOLDER_HTML = """
<div style="font-family:'Inter',system-ui,sans-serif;
            background:#F9FAFB;border-radius:12px;
            border:2px dashed #D1D5DB;padding:56px 32px;text-align:center;">
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
       stroke="#9CA3AF" stroke-width="1.5"
       style="margin:0 auto 16px;display:block;">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
  <p style="font-size:1rem;font-weight:600;color:#374151;margin:0 0 6px;">
    Ready for Analysis
  </p>
  <p style="font-size:.85rem;color:#9CA3AF;margin:0;">
    Upload a chest X-ray on the left, then click <b>Analyse X-Ray</b>
  </p>
</div>"""


# ============================================================
# CSS
# ============================================================
CSS = """
footer { display:none !important; }

/* ── Base container ── */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    background: #F8FAFC !important;
    font-family: 'Inter','Segoe UI',system-ui,sans-serif !important;
    padding: 0 !important;
}

/* ── Kill every Gradio-generated gap/padding above our content ── */
.gradio-container > .main > .wrap,
.gradio-container > .main,
.contain,
.gap,
div.gap {
    gap: 0 !important;
    padding: 0 !important;
}

/* ── Header ── */
.header-wrap {
    background: white;
    border-bottom: 1px solid #E5E7EB;
    padding: 14px 20px;
    margin-bottom: 0;
}

/* ── Left panel card ── */
.left-panel {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    margin: 16px 0 16px 16px;
}

/* ── Right panel spacing ── */
.right-panel-wrap {
    padding: 16px 16px 16px 8px;
}

/* ── Buttons ── */
.analyse-btn button {
    background: #2563EB !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
    height: 44px !important;
    color: white !important;
    width: 100% !important;
    letter-spacing: .02em !important;
    margin-top: 8px !important;
    transition: background .15s !important;
}
.analyse-btn button:hover { background: #1D4ED8 !important; }
.reset-btn button {
    background: white !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    height: 38px !important;
    color: #374151 !important;
    width: 100% !important;
    margin-top: 6px !important;
}
.reset-btn button:hover { background: #F9FAFB !important; }

/* ── Kill Gradio's ghost label above image upload ── */
#upload_box .label-wrap,
#upload_box > .wrap > .label-wrap,
.left-panel label.svelte-1b6s6s,
.left-panel .label-wrap { display: none !important; }
#upload_box > .wrap  { padding-top: 0 !important; }

/* ── Kill Gradio checkbox extra padding ── */
.left-panel .block { padding: 0 !important; margin: 0 !important; }
.left-panel .form  { gap: 0 !important; }

/* ── Responsive grids (used inside result HTML) ── */
.rg-images {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 16px;
}
.rg-analysis {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

/* ── Mobile ── */
@media (max-width: 640px) {
    .left-panel   { margin: 10px 10px 10px 10px; padding: 12px; }
    .rg-images    { grid-template-columns: 1fr; gap: 10px; }
    .rg-analysis  { grid-template-columns: 1fr; gap: 10px; }
    .header-icon  { display: none !important; }
    .header-badge { display: none !important; }
    .header-title { font-size: 0.95rem !important; }
    .header-sub   { display: none !important; }
}
"""

# ============================================================
# UI
# ============================================================
with gr.Blocks(
    title="Chest X-Ray AI Assistant",
    theme=gr.themes.Base(primary_hue="blue", font=gr.themes.GoogleFont("Inter")),
    css=CSS
) as demo:

    # ── Header ──────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-wrap">
      <div style="display:flex;align-items:center;
                  justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div style="display:flex;align-items:center;gap:10px;min-width:0;">
          <div class="header-icon"
               style="background:#EFF6FF;border:1px solid #BFDBFE;
                      border-radius:10px;padding:8px;flex-shrink:0;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                 stroke="#2563EB" stroke-width="2">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <div style="min-width:0;">
            <h1 class="header-title"
                style="font-size:1.05rem;font-weight:700;
                       color:#111827;margin:0 0 2px;white-space:nowrap;">
              Chest X-Ray AI Assistant
            </h1>
            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
              <span style="background:#DBEAFE;color:#1D4ED8;font-size:10px;
                           font-weight:600;padding:1px 7px;border-radius:999px;">
                Pneumonia Detection
              </span>
              <span class="header-sub" style="font-size:10.5px;color:#6B7280;">
                Paediatric chest X-ray analysis · Grad-CAM explainability
              </span>
            </div>
          </div>
        </div>
        <div class="header-badge"
             style="display:flex;align-items:center;gap:6px;
                    background:#F0FDF4;border:1px solid #BBF7D0;
                    border-radius:8px;padding:6px 12px;">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="#16A34A">
            <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"/>
          </svg>
          <span style="font-size:11px;font-weight:600;color:#15803D;">
            Live · CNN + Grad-CAM
          </span>
        </div>
      </div>
    </div>
    """)

    # ── Main layout ─────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left: upload + controls
        with gr.Column(scale=4, min_width=260):
            gr.HTML('<div class="left-panel">')

            gr.HTML("""
            <p style="font-size:10px;font-weight:700;text-transform:uppercase;
                      letter-spacing:.08em;color:#6B7280;margin:0 0 8px;">
              Upload Chest X-Ray
            </p>""")

            image_input = gr.Image(
                type="pil", label="", height=230,
                show_label=False,
                sources=["upload", "clipboard"],
                elem_id="upload_box"
            )

            gr.HTML("""
            <p style="font-size:10px;color:#9CA3AF;text-align:center;
                      margin:4px 0 10px;">
              JPG · PNG · JPEG &nbsp;|&nbsp; Must be a chest X-ray
            </p>""")

            gradcam_toggle = gr.Checkbox(
                label="Show Grad-CAM heatmap", value=True,
                info="Highlights lung regions the model focused on"
            )

            predict_btn = gr.Button(
                "Analyse X-Ray", variant="primary", size="lg",
                elem_classes=["analyse-btn"]
            )
            reset_btn = gr.Button(
                "Analyse Another Image", variant="secondary",
                elem_classes=["reset-btn"]
            )

            gr.HTML("""
            <div style="border-top:1px solid #F3F4F6;margin:14px 0 10px;"></div>
            <p style="font-size:10px;font-weight:700;text-transform:uppercase;
                      letter-spacing:.08em;color:#6B7280;margin:0 0 6px;">
              Try a Sample
            </p>""")

            gr.Examples(
                examples=[
                    ["examples/normal.jpeg"],
                    ["examples/pneumonia.jpeg"]
                ],
                inputs=image_input,
                label=""
            )

            gr.HTML("""</div>""")  # close left-panel

        # Right: results
        with gr.Column(scale=6, min_width=380):
            result_html = gr.HTML(value=PLACEHOLDER_HTML)

    # ── Footer with all disclaimers ──────────────────────────────────────
    gr.HTML("""
    <div style="margin:20px 16px 16px;display:grid;
                grid-template-columns:1fr 1fr 1fr;gap:10px;">

      <div style="background:white;border:1px solid #E5E7EB;
                  border-radius:10px;padding:12px 14px;">
        <p style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:.07em;color:#6B7280;margin:0 0 5px;">
          About
        </p>
        <p style="font-size:11px;color:#4B5563;margin:0;line-height:1.6;">
          CNN trained on the
          <a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
             target="_blank"
             style="color:#2563EB;font-weight:600;text-decoration:none;">
            Kaggle Chest X-Ray dataset
          </a>
          — 5,863 images, binary classification (Normal / Pneumonia).
        </p>
      </div>

      <div style="background:#FFF7ED;border:1px solid #FED7AA;
                  border-radius:10px;padding:12px 14px;">
        <p style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:.07em;color:#92400E;margin:0 0 5px;">
          &#x1F476; Paediatric Only
        </p>
        <p style="font-size:11px;color:#9A3412;margin:0;line-height:1.6;">
          Trained exclusively on chest X-rays of <b>children aged 1-5</b>
          from Guangzhou Women and Children's Medical Center.
          Adult X-rays will give unreliable results.
        </p>
      </div>

      <div style="background:#F0F9FF;border:1px solid #BAE6FD;
                  border-radius:10px;padding:12px 14px;">
        <p style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:.07em;color:#0C4A6E;margin:0 0 5px;">
          Medical Disclaimer
        </p>
        <p style="font-size:11px;color:#0C4A6E;margin:0;line-height:1.6;">
          For educational and research purposes only.
          Not a substitute for clinical diagnosis by a qualified physician
          or radiologist.
        </p>
      </div>

    </div>

    <div style="text-align:center;padding:0 0 20px;">
      <p style="font-size:10px;color:#9CA3AF;margin:0;">
        Built with TensorFlow · Gradio · Grad-CAM
        &nbsp;·&nbsp; Deployed on HuggingFace Spaces
      </p>
    </div>
    """)

    predict_btn.click(
        fn=predict,
        inputs=[image_input, gradcam_toggle],
        outputs=[result_html]
    )
    reset_btn.click(
        fn=lambda: PLACEHOLDER_HTML,
        inputs=[],
        outputs=[result_html]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    