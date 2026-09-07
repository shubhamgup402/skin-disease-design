import os
import numpy as np
import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.cm as cm

from model_arch import (
    SKIN_LABELS, SKIN_LABELS_FULL, IMG_SIZE, MALIGNANT,
    adaptive_preprocess, check_image_quality, segment_lesion, extract_dual_views,
    MultiViewSkinModel, predict_with_uncertainty, GradCAM,
    load_trained_single_view_model, predict_single_view_with_uncertainty, GradCAMSingleView,
)

st.set_page_config(page_title="Dermoscopy AI Assistant", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Mobile-friendly tweaks */
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; padding-left: 1rem; padding-right: 1rem; max-width: 720px;}
div.stButton > button, div.stDownloadButton > button {
    width: 100%; padding: 0.75rem 1rem; font-size: 1.05rem; border-radius: 10px;
}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img {border-radius: 10px;}
.stRadio > div {gap: 0.5rem;}
h2, h3 {margin-top: 1.2rem;}
[data-testid="stMetricValue"] {font-size: 1.1rem;}
img {border-radius: 8px;}
@media (max-width: 480px) {
    .block-container {padding-left: 0.6rem; padding-right: 0.6rem;}
    h1 {font-size: 1.4rem;}
    h2, h3 {font-size: 1.1rem;}
}
</style>
""", unsafe_allow_html=True)

CHECKPOINT_PATH = "skin-model.pth"  # backbone checkpoint from training.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFER_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

SITE_LIST = [
    "unknown", "face", "scalp", "ear", "neck", "trunk", "chest", "back",
    "abdomen", "genital", "upper extremity", "lower extremity", "hand",
    "foot", "acral",
]


@st.cache_resource(show_spinner="Loading model...")
def load_single_view_model(ckpt_path: str, ckpt_mtime: float):
    """The model that actually classifies — trained backbone + trained
    classifier head from training.py, loaded strictly (no silent
    partial loads). `ckpt_mtime` is part of the cache key so replacing
    the checkpoint file (or fixing a bad one) invalidates the cache
    instead of Streamlit silently reusing a stale/failed load."""
    return load_trained_single_view_model(ckpt_path, DEVICE)


@st.cache_resource(show_spinner="Loading experimental multi-view architecture...")
def load_multiview_model():
    """Architecture-preview only: backbone weights are shared from the
    checkpoint, but ReliabilityFusion / MetadataEncoder / EnsembleClassifier
    are randomly initialized and have NOT been trained. Do not treat its
    output as a real prediction until fine-tuned on dual-view labeled data."""
    model = MultiViewSkinModel(num_sites=len(SITE_LIST)).to(DEVICE)
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_backbone_from_checkpoint(state_dict)
    except FileNotFoundError:
        pass
    model.eval()
    return model


def to_tensor(rgb_img):
    t = INFER_TRANSFORM(image=rgb_img)["image"].unsqueeze(0).to(DEVICE)
    return t


def build_metadata_vector(age, sex, site):
    age_norm = torch.tensor([[age / 100.0]], dtype=torch.float32)
    sex_vec = torch.tensor([[1.0, 0.0] if sex == "Male" else [0.0, 1.0]], dtype=torch.float32)
    site_vec = torch.zeros((1, len(SITE_LIST)), dtype=torch.float32)
    site_vec[0, SITE_LIST.index(site)] = 1.0
    return torch.cat([age_norm, sex_vec, site_vec], dim=1).to(DEVICE)


def overlay_cam(rgb_img, cam):
    rgb_resized = np.array(Image.fromarray(rgb_img).resize((IMG_SIZE, IMG_SIZE)))
    heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    return (0.55 * rgb_resized + 0.45 * heat).astype(np.uint8)


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("🔬 Dermoscopy AI Assistant")
st.caption("Camera/upload → quality check → lesion segmentation → dual-view fusion → classification → explanation")

single_view_model = None
if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"Checkpoint not found at '{CHECKPOINT_PATH}'. Place your trained skin-model.pth next to app.py, then rerun.")
    st.stop()

try:
    ckpt_mtime = os.path.getmtime(CHECKPOINT_PATH)
    single_view_model = load_single_view_model(CHECKPOINT_PATH, ckpt_mtime)
    st.caption("✅ Trained checkpoint loaded — predictions come from your trained model.")
except RuntimeError as e:
    st.error("Checkpoint architecture doesn't match EnhancedModel — this checkpoint wasn't produced by training.py, "
              "or training.py's architecture has since changed.")
    with st.expander("Details"):
        st.code(str(e))
    st.stop()

show_experimental = st.toggle(
    "Show experimental multi-view fusion architecture (untrained heads — for architecture/patent preview only)",
    value=False,
)

st.subheader("1 · Capture or upload")
capture_mode = st.radio("Source", ["📱 Camera", "🖼️ Upload"], horizontal=True, label_visibility="collapsed")
img_file = st.camera_input("Take a photo") if capture_mode == "📱 Camera" else st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])

if img_file is None:
    st.info("Provide an image to begin.")
    st.stop()

raw_rgb = np.array(Image.open(img_file).convert("RGB"))

with st.expander("⚙️ Preprocessing stage controls (debug)"):
    d1, d2, d3 = st.columns(3)
    do_hair = d1.checkbox("Hair removal", value=True)
    do_crop = d2.checkbox("FOV crop", value=True)
    do_cc = d3.checkbox("Color constancy", value=True)

pre_rgb = adaptive_preprocess(raw_rgb, do_hair_removal=do_hair, do_fov_crop=do_crop, do_color_constancy=do_cc)

with st.expander("Compare raw vs. preprocessed"):
    r1, r2 = st.columns(2)
    r1.image(raw_rgb, caption="Raw capture")
    r2.image(pre_rgb, caption="After preprocessing")

st.subheader("2 · Image quality check")
quality = check_image_quality(pre_rgb)
qc1, qc2 = st.columns(2)
qc1.metric("Sharpness", quality["blur_score"])
qc2.metric("Brightness", quality["brightness"])
st.caption(f"Resolution: {quality['resolution'][0]}×{quality['resolution'][1]}")

if not quality["passed"]:
    for issue in quality["issues"]:
        st.warning(issue)
    if not st.checkbox("Proceed anyway"):
        st.stop()
else:
    st.success("Image quality OK.")

st.subheader("3 · Select the lesion")
h, w = pre_rgb.shape[:2]
st.image(pre_rgb, caption="Preprocessed image", use_container_width=True)
st.caption("Auto-detects the lesion in the center of frame by default. If it's off-center, "
           "recrop your photo so the lesion is roughly centered, or fine-tune below.")
with st.expander("Fine-tune lesion box (optional)"):
    zoom = st.slider("Box size (% of frame)", 20, 100, 60, help="Smaller = tighter crop around center")
    x_shift = st.slider("Shift left/right", -40, 40, 0)
    y_shift = st.slider("Shift up/down", -40, 40, 0)
    bw_pct = zoom
    bh_pct = zoom
    x0_pct = max(0, min(100 - bw_pct, (100 - bw_pct) // 2 + x_shift))
    y0_pct = max(0, min(100 - bh_pct, (100 - bh_pct) // 2 + y_shift))
    bbox = (int(x0_pct / 100 * w), int(y0_pct / 100 * h), int(bw_pct / 100 * w), int(bh_pct / 100 * h))

st.subheader("4 · Patient metadata (optional, improves fusion)")
age = st.number_input("Age", 0, 120, 45)
mc1, mc2 = st.columns(2)
sex = mc1.selectbox("Sex", ["Male", "Female"])
site = mc2.selectbox("Anatomic site", SITE_LIST)

run = st.button("🔬 Run diagnosis", type="primary", use_container_width=True)
if not run:
    st.stop()

with st.spinner("Segmenting lesion..."):
    lesion_mask = segment_lesion(pre_rgb, bbox=bbox)
    lesion_view, context_view = extract_dual_views(pre_rgb, lesion_mask)

vc1, vc2 = st.columns(2)
vc1.image(lesion_view, caption="Lesion view")
vc2.image(context_view, caption="Skin context view")

# ---- REAL prediction: trained EnhancedModel on the full preprocessed
# image, matching exactly what training.py trained on. ----
with st.spinner("Classifying..."):
    full_t = to_tensor(pre_rgb)
    result = predict_single_view_with_uncertainty(single_view_model, full_t, n_passes=20)

probs = result["mean_probs"][0].cpu().numpy()
uncertainty = result["std_probs"][0].cpu().numpy()
entropy = float(result["predictive_entropy"][0])

order = np.argsort(probs)[::-1]
top_label = SKIN_LABELS[order[0]]

st.subheader("5 · Prediction")
risk = "🔴 Malignant category" if top_label in MALIGNANT else "🟢 Benign category"
st.markdown(f"### {SKIN_LABELS_FULL[top_label]}  \n{risk} · confidence {probs[order[0]]*100:.1f}%")

for idx in order[:3]:
    lbl = SKIN_LABELS[idx]
    st.progress(float(probs[idx]), text=f"{SKIN_LABELS_FULL[lbl]} — {probs[idx]*100:.1f}% (±{uncertainty[idx]*100:.1f})")

st.subheader("6 · Uncertainty")
max_entropy = float(np.log(len(SKIN_LABELS)))
conf_level = "Low" if entropy > 0.6 * max_entropy else ("Moderate" if entropy > 0.3 * max_entropy else "High")
st.write(f"Predictive entropy: **{entropy:.3f}** / max {max_entropy:.3f} → model confidence: **{conf_level}**")
if conf_level == "Low":
    st.warning("High predictive uncertainty — recommend clinical review rather than relying on this output.")

st.subheader("7 · Explanation (Grad-CAM)")
with st.spinner("Generating explanation..."):
    cam_engine = GradCAMSingleView(single_view_model)
    cam = cam_engine.generate(full_t, class_idx=int(order[0]))
    overlay = overlay_cam(pre_rgb, cam)
st.image(overlay, caption=f"Regions driving the '{SKIN_LABELS_FULL[top_label]}' prediction")

# ---- Experimental multi-view fusion architecture (patent preview only) ----
if show_experimental:
    st.divider()
    st.subheader("🧪 Experimental: multi-view reliability fusion")
    st.warning(
        "This path shows the dual-view + reliability-weighted-fusion architecture "
        "for design/patent review. Its fusion and classifier heads are randomly "
        "initialized and NOT trained — treat the numbers below as illustrative "
        "of the mechanism only, not a real diagnosis."
    )
    mv_model = load_multiview_model()
    lesion_t = to_tensor(lesion_view)
    context_t = to_tensor(context_view)
    meta_t = build_metadata_vector(age, sex, site)
    mv_result = predict_with_uncertainty(mv_model, lesion_t, context_t, meta_t, n_passes=10)
    mv_probs = mv_result["mean_probs"][0].cpu().numpy()
    mv_rel_w = mv_result["reliability_weights"][0].cpu().numpy()
    mv_order = np.argsort(mv_probs)[::-1]
    st.caption(f"View reliability weights — lesion: {mv_rel_w[0]*100:.0f}%, context: {mv_rel_w[1]*100:.0f}%")
    for idx in mv_order[:3]:
        lbl = SKIN_LABELS[idx]
        st.progress(float(mv_probs[idx]), text=f"(untrained) {SKIN_LABELS_FULL[lbl]} — {mv_probs[idx]*100:.1f}%")
    st.caption("To make this path real: fine-tune MultiViewSkinModel's fusion + classifier heads "
               "on a labeled dataset of (lesion_view, context_view, metadata) → dx.")

st.caption("Research/decision-support prototype only — not a substitute for professional diagnosis.")
