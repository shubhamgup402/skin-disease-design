"""
Multi-view dermoscopy diagnostic architecture.

Pipeline stages (mirrors the system diagram):
  Image -> Quality Check -> Lesion Segmentation -> {Lesion view, Context view}
  -> twin CNN/ViT encoders -> Reliability-Weighted Fusion (+ metadata)
  -> Ensemble Classifier -> {Prediction, Uncertainty} -> Grad-CAM XAI

Kept intentionally in one module so the full inference graph (and its
claim-relevant components: adaptive preprocessing, reliability-weighted
fusion, dual-view acquisition) is auditable in one place.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

SKIN_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
SKIN_LABELS_FULL = {
    'akiec': 'Actinic keratoses / intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesion',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevus',
    'vasc': 'Vascular lesion',
}
IMG_SIZE = 256
BACKBONE_NAME = 'tf_efficientnet_b3.ns_jft_in1k'
MALIGNANT = {'akiec', 'bcc', 'mel'}


# =====================================================================
# STAGE 1 — Adaptive pre-processing (identical to training pipeline,
# required so inference-time statistics match what the model was
# trained on)
# =====================================================================

def estimate_hair_density(gray_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return np.count_nonzero(mask) / mask.size, mask


def remove_hair_if_needed(bgr_img, density_threshold=0.015, max_density=0.12):
    """
    Stage A: DullRazor-style hair removal, applied ONLY if hair density
    falls in a plausible range. Real hair is a SPARSE mask of thin
    curvilinear structures. A mask that's too large (> max_density)
    usually means the blackhat filter latched onto something else —
    specular highlights, skin pores, uneven lighting — and inpainting
    that would smear/pixelate the image instead of cleaning it, so we
    skip in that case rather than risk it.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    density, mask = estimate_hair_density(gray)
    if density_threshold < density <= max_density:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        bgr_img = cv2.inpaint(bgr_img, mask, 3, cv2.INPAINT_TELEA)
    return bgr_img, density


def crop_to_field_of_view(bgr_img, dark_thresh=20, min_area_frac=0.5):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr_img
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    area_frac = (w * h) / (bgr_img.shape[0] * bgr_img.shape[1])
    if area_frac < min_area_frac or area_frac > 0.98:
        return bgr_img
    return bgr_img[y:y + h, x:x + w]


def shades_of_gray_white_balance(bgr_img, p=6, strength=None):
    img = bgr_img.astype(np.float32)
    channel_means = np.mean(np.power(img, p), axis=(0, 1)) ** (1.0 / p)
    channel_means = np.clip(channel_means, 1e-6, None)
    scale = channel_means.mean() / channel_means
    if strength is None:
        deviation = np.std(channel_means) / (channel_means.mean() + 1e-6)
        strength = float(np.clip(deviation * 4.0, 0.0, 1.0))
    scale = 1.0 + strength * (scale - 1.0)
    corrected = np.clip(img * scale, 0, 255).astype(np.uint8)
    return corrected, strength


def adaptive_preprocess(rgb_img, do_hair_removal=True, do_fov_crop=True, do_color_constancy=True):
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    if do_hair_removal:
        bgr, _ = remove_hair_if_needed(bgr)
    if do_fov_crop:
        bgr = crop_to_field_of_view(bgr)
    if do_color_constancy:
        bgr, _ = shades_of_gray_white_balance(bgr)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =====================================================================
# STAGE 2 — Image quality gate
# =====================================================================

def check_image_quality(rgb_img):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    h, w = gray.shape
    issues = []
    if blur_score < 60:
        issues.append("Image looks blurry — hold the camera steady and refocus.")
    if brightness < 40:
        issues.append("Image is too dark — improve lighting.")
    if brightness > 220:
        issues.append("Image is overexposed — reduce glare/flash.")
    if min(h, w) < 224:
        issues.append("Resolution too low — move closer or use a higher-res capture.")
    return {
        "blur_score": round(blur_score, 1),
        "brightness": round(brightness, 1),
        "resolution": (w, h),
        "passed": len(issues) == 0,
        "issues": issues,
    }


# =====================================================================
# STAGE 3 — Lesion segmentation + dual-view extraction
# =====================================================================

def segment_lesion(rgb_img, bbox=None):
    """
    GrabCut segmentation seeded by a user-provided bounding box
    (x, y, w, h) in image pixel coordinates. Falls back to a centered
    prior box covering 60% of the frame if none is supplied.
    Returns a binary mask (uint8, {0,1}) the same size as rgb_img.
    """
    h, w = rgb_img.shape[:2]
    if bbox is None:
        bw, bh = int(w * 0.6), int(h * 0.6)
        bbox = ((w - bw) // 2, (h - bh) // 2, bw, bh)

    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, bbox, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        lesion_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except cv2.error:
        lesion_mask = np.zeros((h, w), np.uint8)
        x, y, bw, bh = bbox
        lesion_mask[y:y + bh, x:x + bw] = 1

    if lesion_mask.sum() < 0.01 * h * w:
        x, y, bw, bh = bbox
        lesion_mask[:] = 0
        lesion_mask[y:y + bh, x:x + bw] = 1
    return lesion_mask


def extract_dual_views(rgb_img, lesion_mask, margin=0.25):
    """
    Splits the image into the two views the architecture fuses:
      lesion_view   – tight crop around the segmented lesion (masked
                       background suppressed so the encoder attends to
                       morphology, not surrounding skin)
      context_view  – the full frame with the lesion masked OUT, i.e.
                       the surrounding skin field used for contextual
                       cues (skin tone, nearby nevi, anatomic site)
    """
    ys, xs = np.where(lesion_mask > 0)
    h, w = rgb_img.shape[:2]
    if len(xs) == 0:
        return rgb_img.copy(), rgb_img.copy()

    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - margin * bw))
    x1 = min(w, int(x1 + margin * bw))
    y0 = max(0, int(y0 - margin * bh))
    y1 = min(h, int(y1 + margin * bh))

    lesion_crop = rgb_img[y0:y1, x0:x1].copy()
    lesion_view = lesion_crop if lesion_crop.size else rgb_img.copy()

    context_view = rgb_img.copy()
    context_view[lesion_mask > 0] = context_view[lesion_mask > 0] // 3  # dim lesion, keep field
    return lesion_view, context_view


# =====================================================================
# STAGE 4 — Twin encoders
# =====================================================================

class ViewEncoder(nn.Module):
    """Single-view feature extractor: shared backbone family, dual
    avg/max pooling head, matching the checkpoint trained in training.py."""

    def __init__(self, model_name=BACKBONE_NAME, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            feat = self.backbone(dummy)
            self.feature_dim = feat.shape[1]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x, return_map=False):
        fmap = self.backbone(x)
        pooled = torch.cat([self.avg_pool(fmap).flatten(1), self.max_pool(fmap).flatten(1)], dim=1)
        if return_map:
            return pooled, fmap
        return pooled

    @property
    def out_dim(self):
        return self.feature_dim * 2


# =====================================================================
# STAGE 5 — Reliability-weighted fusion
# =====================================================================

class ReliabilityFusion(nn.Module):
    """
    Learns a per-view scalar reliability score r_i in [0,1] from each
    view's own pooled features (a lightweight self-assessment head),
    then fuses views by their normalized reliability weights instead of
    naive concatenation/averaging. Views the model finds less
    informative for a given sample are automatically down-weighted.
    """

    def __init__(self, view_dim, num_views=2):
        super().__init__()
        self.num_views = num_views
        self.reliability_head = nn.Sequential(
            nn.Linear(view_dim, view_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(view_dim // 4, 1),
        )
        self.project = nn.Linear(view_dim, view_dim)

    def forward(self, view_feats):
        # view_feats: list of [B, D] tensors, one per view
        scores = torch.cat([self.reliability_head(v) for v in view_feats], dim=1)  # [B, num_views]
        weights = F.softmax(scores, dim=1)  # normalized reliability
        stacked = torch.stack([self.project(v) for v in view_feats], dim=1)  # [B, num_views, D]
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return fused, weights


# =====================================================================
# STAGE 6 — Ensemble classifier (fused visual features + metadata)
# =====================================================================

class MetadataEncoder(nn.Module):
    """Age (scalar), sex (one-hot), anatomic site (one-hot) -> embedding."""

    def __init__(self, num_sites=15, out_dim=32):
        super().__init__()
        in_dim = 1 + 2 + num_sites
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, meta_vec):
        return self.net(meta_vec)


class EnsembleClassifier(nn.Module):
    def __init__(self, fused_dim, meta_dim, num_classes=len(SKIN_LABELS), dropout=0.4):
        super().__init__()
        in_dim = fused_dim + meta_dim
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, fused, meta_embed):
        return self.net(torch.cat([fused, meta_embed], dim=1))


# =====================================================================
# Full model
# =====================================================================

class MultiViewSkinModel(nn.Module):
    """
    Twin-view encoder (shared weights) -> reliability-weighted fusion
    -> metadata fusion -> ensemble classifier.

    Sharing backbone weights between the lesion and context encoders
    keeps parameter count in line with the single-view checkpoint
    produced by training.py, so that checkpoint's backbone weights can
    be loaded into both branches at inference time; the fusion and
    classifier heads are the newly-introduced components.
    """

    def __init__(self, model_name=BACKBONE_NAME, num_sites=15, num_classes=len(SKIN_LABELS)):
        super().__init__()
        self.encoder = ViewEncoder(model_name)  # shared across views
        self.fusion = ReliabilityFusion(self.encoder.out_dim, num_views=2)
        self.meta_encoder = MetadataEncoder(num_sites=num_sites)
        self.classifier = EnsembleClassifier(self.encoder.out_dim, out_dim_meta(self.meta_encoder), num_classes)

    def forward(self, lesion_x, context_x, meta_vec):
        lesion_feat = self.encoder(lesion_x)
        context_feat = self.encoder(context_x)
        fused, weights = self.fusion([lesion_feat, context_feat])
        meta_embed = self.meta_encoder(meta_vec)
        logits = self.classifier(fused, meta_embed)
        return logits, weights

    def load_backbone_from_checkpoint(self, state_dict):
        """Loads a single-view `EnhancedModel` checkpoint's backbone weights
        (from training.py) into the shared twin encoder; classifier/fusion
        heads are left at their initialized values."""
        backbone_sd = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        missing, unexpected = self.encoder.backbone.load_state_dict(backbone_sd, strict=False)
        return missing, unexpected


def out_dim_meta(meta_encoder):
    return meta_encoder.net[-1].out_features


# =====================================================================
# Trained single-view model — EXACT mirror of EnhancedModel in
# training.py, used so checkpoint weights (backbone + classifier) load
# with zero mismatch and produce real, trained predictions. This is
# the model that actually classifies; the multi-view fusion pipeline
# above runs alongside it for segmentation/XAI/patent-architecture
# purposes but its classifier head is untrained until fine-tuned on a
# dual-view labeled dataset.
# =====================================================================

class EnhancedModel(nn.Module):
    def __init__(self, model_name=BACKBONE_NAME, num_classes=len(SKIN_LABELS), dropout_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            feature_dim = self.backbone(dummy).shape[1]

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, x, return_features=False):
        fmap = self.backbone(x)
        avg_pool = self.global_avg_pool(fmap).flatten(1)
        max_pool = self.global_max_pool(fmap).flatten(1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        logits = self.classifier(combined)
        if return_features:
            return logits, fmap
        return logits


def load_trained_single_view_model(ckpt_path, device):
    """Strictly loads a training.py checkpoint (backbone + classifier)
    into EnhancedModel. Raises if any key mismatches, since silent
    partial loads are exactly what produced the untrained-head bug."""
    model = EnhancedModel().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_single_view_with_uncertainty(model, x, n_passes=20):
    """MC-Dropout uncertainty for the trained single-view model."""
    model.eval()
    _enable_mc_dropout(model)
    probs_stack = []
    for _ in range(n_passes):
        logits = model(x)
        probs_stack.append(F.softmax(logits, dim=1))
    probs_stack = torch.stack(probs_stack)
    mean_probs = probs_stack.mean(dim=0)
    std_probs = probs_stack.std(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-9))).sum(dim=1)
    return {"mean_probs": mean_probs, "std_probs": std_probs, "predictive_entropy": entropy}


class GradCAMSingleView:
    """Grad-CAM for EnhancedModel (trained single-view classifier)."""

    def __init__(self, model: 'EnhancedModel'):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = GradCAM._find_last_conv(model.backbone)
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# =====================================================================
# STAGE 7 — Predictive uncertainty (MC-Dropout)
# =====================================================================

def _enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_with_uncertainty(model, lesion_x, context_x, meta_vec, n_passes=20):
    model.eval()
    _enable_mc_dropout(model)  # keep dropout active for MC sampling
    probs_stack = []
    weights_stack = []
    for _ in range(n_passes):
        logits, weights = model(lesion_x, context_x, meta_vec)
        probs_stack.append(F.softmax(logits, dim=1))
        weights_stack.append(weights)
    probs_stack = torch.stack(probs_stack)  # [N, B, C]
    mean_probs = probs_stack.mean(dim=0)
    std_probs = probs_stack.std(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-9))).sum(dim=1)
    mean_reliability = torch.stack(weights_stack).mean(dim=0)
    return {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "predictive_entropy": entropy,
        "reliability_weights": mean_reliability,
    }


# =====================================================================
# STAGE 8 — Grad-CAM explanation
# =====================================================================

class GradCAM:
    """Grad-CAM over the shared encoder's backbone, run separately on the
    lesion view (primary explanation surface)."""

    def __init__(self, model: MultiViewSkinModel):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = self._find_last_conv(model.encoder.backbone)
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    @staticmethod
    def _find_last_conv(module):
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, lesion_x, context_x, meta_vec, class_idx):
        self.model.zero_grad()
        logits, _ = self.model(lesion_x, context_x, meta_vec)
        score = logits[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
