"""Dominant-color classification, per the dual-camera recipe v3 §6's exact
algorithm (superseding the HSV-based version this module originally shipped
with for the v1/v2-era text recipe):

1. Crop to the central 60% of the Logitech object ROI (reduces shadow and
   label-text bias at the crop's edges).
2. Convert BGR -> LAB (`cv2.cvtColor`) -- perceptually-uniform, unlike RGB.
3. k-means (k=4) on LAB pixels; very dark (L<20) or very light (L>95) "pure
   near-black/white" pixels are excluded from centroid computation only
   (they still count toward cluster size/assignment), so a few
   near-black/white outlier pixels can't drag a genuinely-colored cluster's
   centroid toward achromatic.
4. Dominant cluster = the one with the most assigned pixels (including the
   excluded-from-centroid ones). Classified achromatic-first from L and
   chroma (sqrt(a^2+b^2)): L<35 -> Black, L>75 -> White, else Grey. Only once
   a centroid clears the achromatic band does its a*/b* hue axis matter:
   strongly positive a* -> Red, strongly negative a* -> Green, strongly
   positive b* -> Yellow, strongly negative b* -> Blue.
5. Confidence = the dominant cluster's pixel share, dampened by its
   distance to the nearest classification-boundary threshold; a tie between
   the top two clusters' shares (<8% margin) reports Other with lower
   confidence rather than guessing between them.

Eight classes, unchanged from the prior recipe round: Blue, Grey, Black,
White, Red, Green, Yellow, Other. This is a small, dependency-light,
standalone module (numpy + OpenCV only) -- intentionally separate from
`geometry.py`'s existing `dominant_color()`, which uses a different, broader
color vocabulary for the existing dashboard/material-labeling flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RECIPE_COLOR_CLASSES = ("Blue", "Grey", "Black", "White", "Red", "Green", "Yellow", "Other")

_MIN_PIXELS = 20
_DEFAULT_KMEANS_K = 4
_DEFAULT_CENTER_CROP_FRAC = 0.6

# v3 §6 step 3: "ignore low-L<20 and very-high L>95 pure-near-black-white
# pixels for cluster center only, not for assignment."
_CENTROID_EXCLUDE_LOW_L = 20.0
_CENTROID_EXCLUDE_HIGH_L = 95.0

# Achromatic decision thresholds (true L 0-100 scale, chroma = sqrt(a^2+b^2)
# on the true -127..127-ish scale). Below _CHROMA_ACHROMATIC_MAX the
# centroid is unambiguously grey-family; between that and
# _CHROMA_CHROMATIC_MIN it is too weakly saturated to call confidently and
# falls to "Other" (v3 §6 step 4's "Ambiguous/mixed -> Other").
_CHROMA_ACHROMATIC_MAX = 12.0
_CHROMA_CHROMATIC_MIN = 18.0
_BLACK_MAX_L = 35.0
_WHITE_MIN_L = 75.0

# v3 §6 step 5: "A tie (<8% margin) -> Other + lower confidence."
_TIE_MARGIN = 0.08


@dataclass(slots=True)
class ColorResult:
    label: str
    confidence: float
    lightness: float | None
    a_star: float | None
    b_star: float | None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "lightness": None if self.lightness is None else round(float(self.lightness), 2),
            "a_star": None if self.a_star is None else round(float(self.a_star), 2),
            "b_star": None if self.b_star is None else round(float(self.b_star), 2),
        }


def _center_crop_region(box: tuple[int, int, int, int], frac: float) -> tuple[int, int, int, int]:
    """The central `frac` (both axes) sub-rectangle of `box` (x1, y1, x2, y2),
    centered -- v3 §6 step 1's "central 60% crop of the Logitech object ROI"."""
    x1, y1, x2, y2 = box
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
    new_w, new_h = box_w * frac, box_h * frac
    cx, cy = x1 + box_w / 2.0, y1 + box_h / 2.0
    return (
        int(round(cx - new_w / 2.0)),
        int(round(cy - new_h / 2.0)),
        int(round(cx + new_w / 2.0)),
        int(round(cy + new_h / 2.0)),
    )


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _kmeans_lab(
    lab_pixels: np.ndarray, k: int, *, seed: int = 0, iterations: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """A small Lloyd's-algorithm k-means in LAB space, written by hand (not
    `cv2.kmeans`) because v3 §6 step 3's centroid-exclusion rule -- extreme-L
    pixels count for cluster *assignment* but not for *updating the
    centroid* -- isn't expressible through cv2's kmeans API. Deterministic
    (seeded, percentile-based init) so results are reproducible across runs
    on the same crop. Returns (labels, centroids); centroids are computed
    from non-extreme member pixels only, falling back to the full member
    mean if a cluster ends up with no non-extreme pixels at all.
    """
    n = lab_pixels.shape[0]
    rng = np.random.default_rng(seed)
    k = max(1, min(k, n))

    # k-means++-style seeding for determinism and reasonable spread, using
    # the seeded RNG rather than relying on pixel order.
    first_index = int(rng.integers(0, n))
    centroids = [lab_pixels[first_index]]
    for _ in range(1, k):
        distances = np.min(
            [np.sum((lab_pixels - c) ** 2, axis=1) for c in centroids], axis=0
        )
        total = float(distances.sum())
        if total <= 0:
            centroids.append(lab_pixels[int(rng.integers(0, n))])
            continue
        probabilities = distances / total
        next_index = int(rng.choice(n, p=probabilities))
        centroids.append(lab_pixels[next_index])
    centroids = np.stack(centroids, axis=0).astype(np.float64)

    lightness = lab_pixels[:, 0]
    extreme = (lightness < _CENTROID_EXCLUDE_LOW_L) | (lightness > _CENTROID_EXCLUDE_HIGH_L)

    labels = np.zeros(n, dtype=np.int64)
    for _iteration in range(iterations):
        distances = np.stack([np.sum((lab_pixels - c) ** 2, axis=1) for c in centroids], axis=1)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels) and _iteration > 0:
            labels = new_labels
            break
        labels = new_labels
        for cluster_index in range(k):
            member_mask = labels == cluster_index
            if not np.any(member_mask):
                continue
            centroid_mask = member_mask & ~extreme
            if np.any(centroid_mask):
                centroids[cluster_index] = lab_pixels[centroid_mask].mean(axis=0)
            else:
                centroids[cluster_index] = lab_pixels[member_mask].mean(axis=0)

    return labels, centroids


def _classify_lab_centroid(l_value: float, a_value: float, b_value: float) -> tuple[str, float]:
    """Achromatic-first decision tree (v3 §6 step 4). Returns
    (label, boundary_margin) -- `boundary_margin` is a small unitless [0, 1]
    measure of how far the centroid sits from the *nearest* threshold it was
    classified against, used to dampen confidence for a borderline call."""
    chroma = float(np.hypot(a_value, b_value))

    if chroma < _CHROMA_ACHROMATIC_MAX:
        if l_value < _BLACK_MAX_L:
            margin = min(1.0, (_BLACK_MAX_L - l_value) / _BLACK_MAX_L)
            return "Black", margin
        if l_value > _WHITE_MIN_L:
            margin = min(1.0, (l_value - _WHITE_MIN_L) / max(1.0, 100.0 - _WHITE_MIN_L))
            return "White", margin
        # Mid-lightness, low chroma -> Grey.
        span = _WHITE_MIN_L - _BLACK_MAX_L
        margin = min(l_value - _BLACK_MAX_L, _WHITE_MIN_L - l_value) / max(span, 1.0)
        return "Grey", margin

    if chroma < _CHROMA_CHROMATIC_MIN:
        # Between "clearly achromatic" and "clearly chromatic" -- v3's own
        # "Ambiguous/mixed -> Other."
        margin = (chroma - _CHROMA_ACHROMATIC_MAX) / max(_CHROMA_CHROMATIC_MIN - _CHROMA_ACHROMATIC_MAX, 1e-6)
        return "Other", 1.0 - margin

    # Clearly chromatic: dominant axis (|a*| vs |b*|) picks red/green vs
    # yellow/blue, sign picks the direction along that axis.
    if abs(a_value) >= abs(b_value):
        label = "Red" if a_value > 0 else "Green"
        margin = (abs(a_value) - abs(b_value)) / max(abs(a_value), 1.0)
    else:
        label = "Yellow" if b_value > 0 else "Blue"
        margin = (abs(b_value) - abs(a_value)) / max(abs(b_value), 1.0)
    return label, float(np.clip(margin, 0.0, 1.0))


def classify_dominant_color(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    kmeans_k: int = _DEFAULT_KMEANS_K,
    center_crop_frac: float = _DEFAULT_CENTER_CROP_FRAC,
) -> ColorResult:
    """v3 §6: dominant-color classification of one masked object crop.

    `image_bgr` is a uint8 or float BGR image (the Logitech color frame,
    per the recipe's stated camera preference); `mask` is a boolean array of
    the same height/width selecting the object's own pixels.
    """
    if image_bgr.shape[:2] != mask.shape:
        raise ValueError("image_bgr and mask must have the same height/width")

    bbox = _mask_bbox(mask.astype(bool))
    if bbox is None:
        return ColorResult("Other", 0.0, None, None, None)

    crop_box = _center_crop_region(bbox, center_crop_frac)
    height, width = mask.shape
    x1 = int(np.clip(crop_box[0], 0, width))
    y1 = int(np.clip(crop_box[1], 0, height))
    x2 = int(np.clip(crop_box[2], x1 + 1, width))
    y2 = int(np.clip(crop_box[3], y1 + 1, height))

    region_mask = np.zeros_like(mask, dtype=bool)
    region_mask[y1:y2, x1:x2] = True
    selected_mask = mask.astype(bool) & region_mask
    if np.count_nonzero(selected_mask) < _MIN_PIXELS:
        # The central 60% crop happened to contain too few of the object's
        # own pixels (e.g. a very thin/irregular mask) -- fall back to the
        # full mask rather than reporting "Other" purely from an unlucky crop.
        selected_mask = mask.astype(bool)
    pixel_count = int(np.count_nonzero(selected_mask))
    if pixel_count < _MIN_PIXELS:
        return ColorResult("Other", 0.0, None, None, None)

    import cv2

    pixels_bgr = image_bgr[selected_mask]
    if pixels_bgr.dtype != np.uint8:
        pixels_bgr = np.clip(pixels_bgr, 0, 255).astype(np.uint8)
    lab_uint8 = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    # cv2's 8-bit LAB packs L into [0,255] (true 0-100 scale) and a*/b* into
    # [0,255] offset by 128 (true roughly -127..127) -- convert to the true
    # scale the recipe's own thresholds are written against.
    lab = np.empty_like(lab_uint8)
    lab[:, 0] = lab_uint8[:, 0] * (100.0 / 255.0)
    lab[:, 1] = lab_uint8[:, 1] - 128.0
    lab[:, 2] = lab_uint8[:, 2] - 128.0

    k = max(1, min(kmeans_k, max(1, pixel_count // 5)))
    labels, centroids = _kmeans_lab(lab, k)
    cluster_sizes = np.array([np.count_nonzero(labels == i) for i in range(k)], dtype=np.float64)
    total = float(cluster_sizes.sum()) or 1.0

    order = np.argsort(cluster_sizes)[::-1]
    dominant_index = int(order[0])
    dominant_share = float(cluster_sizes[dominant_index]) / total

    if k >= 2:
        runner_up_index = int(order[1])
        runner_up_share = float(cluster_sizes[runner_up_index]) / total
        if dominant_share - runner_up_share < _TIE_MARGIN:
            centroid = centroids[dominant_index]
            return ColorResult(
                label="Other",
                confidence=float(np.clip(dominant_share * 0.5, 0.0, 1.0)),
                lightness=float(centroid[0]),
                a_star=float(centroid[1]),
                b_star=float(centroid[2]),
            )

    centroid = centroids[dominant_index]
    label, boundary_margin = _classify_lab_centroid(float(centroid[0]), float(centroid[1]), float(centroid[2]))
    confidence = float(np.clip(dominant_share * (0.6 + 0.4 * boundary_margin), 0.0, 1.0))

    return ColorResult(
        label=label,
        confidence=confidence,
        lightness=float(centroid[0]),
        a_star=float(centroid[1]),
        b_star=float(centroid[2]),
    )
