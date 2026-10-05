"""One definition of an object's colour, the same for both cameras.

`geometry.classify_color` samples only a band just inside the mask edge (so an
open bag's contents do not decide its colour) and lets any hue win once a
fifth of that band is chromatic. Together those made the answer depend on the
viewpoint: a grey bottle with a red cap seen from the side has the cap on its
silhouette, inside the band, and came out "red"; seen from above the cap sits
in the middle of the silhouette, outside the band, and came out "grey".

Here the colour is defined once:

* **dominant** -- the colour most of the object's visible surface has, by
  pixel count, over the mask with its outermost pixels, specular glare and
  deep shadow removed. Bags keep the skin band, because their middle can be
  contents rather than bag;
* **accent** -- a small, strongly coloured part distinct from the dominant
  one (a cap, a label), reported separately rather than replacing it;
* **mixed / transparent / unknown** -- when no colour holds enough of the
  surface, when the object shows mostly the floor through it, or when too few
  usable pixels remain. A low-confidence guess is not returned as a colour.

The vocabulary is `geometry`'s, so the colour-to-waste-stream sorting rules
are unchanged; the new states map to no stream, exactly like "unknown".
Nothing here touches geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .geometry import _CATEGORIES, _categorise

MIXED = "mixed"
TRANSPARENT = "transparent"
UNKNOWN = "unknown"

GLARE_VALUE, GLARE_MAX_SATURATION = 0.97, 0.10
SHADOW_VALUE = 0.12
# Glare and shadow are patches on an object. Covering most of it, they are its
# colour: a white bag, a black bag.
MAX_EXCLUDED_SHARE = 0.60
MIN_USABLE_PIXELS = 20
MIN_USABLE_FRACTION = 0.20
# The dominant colour must hold this much of the usable surface...
MIN_DOMINANT_SHARE = 0.35
# ... and a chromatic colour overrides a larger neutral one only when it is
# most of the surface's colour and the neutral part is not a majority.
CHROMATIC_SURFACE_SHARE = 0.35
CHROMATIC_AGREEMENT = 0.60
ACCENT_MIN_SHARE, ACCENT_MIN_PIXELS, ACCENT_MIN_SATURATION = 0.04, 15, 0.35
# A pixel this close to the empty-scene image is floor seen through the object.
SEE_THROUGH_DIFFERENCE = 20.0 / 255.0
SEE_THROUGH_SHARE = 0.50
NEUTRAL = frozenset({"white", "grey", "black"})
SECONDARY_MIN_SHARE = 0.20
BAG_WORDS = frozenset({"bag", "bags", "sack", "sacks", "liner", "binbag", "polythene", "pouch"})


@dataclass
class ColourEvidence:
    colour: str
    confidence: float
    accent: str | None = None
    accent_share: float = 0.0
    state: str = "solid"
    usable_pixels: int = 0
    glare_pixels: int = 0
    shadow_pixels: int = 0
    sampled_region: str = "surface"
    shares: dict | None = None
    # Per-channel (B, G, R) chroma gains applied for lighting drift since the
    # empty reference was captured; None when not applied.
    illumination_gains: list | None = None
    # Other colours that each cover a real part of the surface (a black-and-white shoe, a printed
    # box); never more than two, never the dominant one again.
    secondary: list | None = None
    secondary_note: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    try:
        import cv2

        return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=iterations).astype(bool)
    except ImportError:  # pragma: no cover - cv2 is a dependency
        out = mask.copy()
        for _ in range(iterations):
            padded = np.pad(out, 1)
            out = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                   & padded[1:-1, :-2] & padded[1:-1, 2:])
        return out


def _illumination_gains(frame_bgr: np.ndarray, background_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Chroma-only correction for a lighting/white-balance change since the
    empty reference: compare the floor just around the object now with the
    same floor in the reference. Brightness is left alone (normalised gains),
    so a shadow cast by the object does not brighten it; only a colour cast
    (warm lamp, auto white balance) is undone. Returns None when the evidence
    is too thin or the correction would be negligible."""
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return None
    m = mask.astype(np.uint8)
    ring = (cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=12) > 0) & \
        ~(cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=4) > 0)
    if int(np.count_nonzero(ring)) < 200:
        return None
    live = frame_bgr[ring].astype(np.float64)
    reference = background_bgr[ring].astype(np.float64)
    # V52: the surround must be the SAME surface as in the reference. In a bin that fills up, the
    # pixels around a bag are other bags that were not there when the reference was taken; gains from
    # them are not a lighting change (a white bag on a green pile read "purple").
    live_grey, reference_grey = live.mean(axis=1), reference.mean(axis=1)
    if live_grey.std() < 1.0 or reference_grey.std() < 1.0:
        return None
    if float(np.corrcoef(live_grey, reference_grey)[0, 1]) < SAME_SURFACE_MIN_CORRELATION:
        return None
    usable = (live.max(axis=1) < 250) & (reference.max(axis=1) < 250) & (live.min(axis=1) > 8) \
        & (reference.min(axis=1) > 8)
    if int(np.count_nonzero(usable)) < 200:
        return None
    gains = np.median(reference[usable], axis=0) / np.maximum(np.median(live[usable], axis=0), 1e-6)
    gains = gains / float(np.mean(gains))
    if float(np.max(np.abs(gains - 1.0))) < 0.02:
        return None
    return np.clip(gains, 0.8, 1.25)


# Brown is dark or desaturated orange/yellow (ISCC-NBS); HSV brightness alone called lit cardboard
# "orange". Constants are colour-naming definitions, not fitted to any test image.
BROWN_MAX_SATURATION = 0.60
BROWN_MAX_VALUE = 0.55
PINK_FROM = ("red", "orange", "brown", "white", "grey", "yellow")
PINK_MIN_A, PINK_A_OVER_B, PINK_MIN_B = 8.0, 0.8, -6.0   # Lab a* (red-green) vs b* (yellow-blue); purple is b* < 0
PINK_MAX_SATURATION, PINK_MIN_VALUE = 0.62, 0.30   # saturated red stays red; very dark stays dark


SAME_SURFACE_MIN_CORRELATION = 0.80
WHITE_REFERENCE_MIN = 0.60       # a credible white surface must be at least this bright ...
WHITE_REFERENCE_MAX_SAT = 0.15   # ... and near-neutral
WHITE_REFERENCE_MIN_PIXELS = 50


def _white_reference(frame_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """White-patch exposure reference from bright, near-neutral surfaces OUTSIDE the object. Indoors
    a white object is often at V~0.7, which absolute cuts call grey. With no such surface in view
    (or the object itself the brightest thing) there is no reference and nothing is rescaled: a mid
    grey object in a dark scene stays grey."""
    step = 4
    sample = frame_bgr[::step, ::step].astype(np.float32) / 255.0
    outside = np.ones(sample.shape[:2], bool) if mask is None else ~mask[::step, ::step]
    pixels = sample[outside]
    if pixels.size == 0:
        return 1.0
    value = pixels.max(axis=1)
    saturation = (value - pixels.min(axis=1)) / np.maximum(value, 1e-8)
    neutral_bright = value[(saturation <= WHITE_REFERENCE_MAX_SAT) & (value >= WHITE_REFERENCE_MIN)]
    if neutral_bright.size < WHITE_REFERENCE_MIN_PIXELS:
        return 1.0
    return float(np.clip(np.percentile(neutral_bright, 90), WHITE_REFERENCE_MIN, 1.0))


def _categorise_object(pixels: np.ndarray, frame_bgr: np.ndarray,
                       mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel colour classes after exposure normalisation (one gain for all channels, so hue and
    saturation are unchanged), with dark/desaturated orange-yellow named brown."""
    relative = np.clip(pixels / _white_reference(frame_bgr, mask), 0.0, 1.0)
    codes = _categorise(relative)
    value = relative.max(axis=1)
    saturation = (value - relative.min(axis=1)) / np.maximum(value, 1e-8)
    orange, yellow, brown = (_CATEGORIES.index(name) for name in ("orange", "yellow", "brown"))
    to_brown = ((codes == orange) & ((saturation < BROWN_MAX_SATURATION) | (value < BROWN_MAX_VALUE))) \
        | ((codes == yellow) & (value < BROWN_MAX_VALUE))
    codes = np.where(to_brown, brown, codes)
    # V53: pink is a light, desaturated red -- in Lab, red-green a* at least comparable to yellow-blue b*.
    # Hue bands alone called pale pink bags red/orange (warm light), or white/yellow once desaturated.
    if relative.size:
        try:
            import cv2
            lab = cv2.cvtColor(np.clip(relative * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1, 3),
                               cv2.COLOR_BGR2LAB)[:, 0, :].astype(np.float32) - np.float32([0, 128, 128])
            a_star, b_star = lab[:, 1], lab[:, 2]
            candidates = np.isin(codes, [_CATEGORIES.index(n) for n in PINK_FROM])
            pink = candidates & (a_star >= PINK_MIN_A) & (a_star >= PINK_A_OVER_B * b_star) & (b_star >= PINK_MIN_B) \
                & (saturation < PINK_MAX_SATURATION) & (value >= PINK_MIN_VALUE)
            codes = np.where(pink, _CATEGORIES.index("pink"), codes)
        except ImportError:  # pragma: no cover
            pass
    return codes, value


def _is_bag(label: str | None) -> bool:
    words = set((label or "").lower().replace("-", " ").replace("_", " ").split())
    return bool(words & BAG_WORDS) and "handbag" not in words


def describe_colour(
    frame_bgr: np.ndarray, mask: np.ndarray | None, *, label: str | None = None,
    background_bgr: np.ndarray | None = None,
) -> ColourEvidence:
    """Dominant colour, accent and confidence of one object's visible surface."""
    if mask is None or frame_bgr is None or mask.shape != frame_bgr.shape[:2]:
        return ColourEvidence(UNKNOWN, 0.0, state="no_mask")
    mask = mask.astype(bool)
    area = int(np.count_nonzero(mask))
    if area < MIN_USABLE_PIXELS:
        return ColourEvidence(UNKNOWN, 0.0, state="too_few_pixels")
    # The outermost pixels are floor at object depth: segmentation halos.
    surface = _erode(mask, 2 if area >= 400 else 1)
    if np.count_nonzero(surface) < MIN_USABLE_PIXELS:
        surface = mask
    region = "surface"
    if _is_bag(label) and area >= 120:
        # An open bag's middle is its contents; its skin is the band inside
        # the edge. Only for bags: a bottle's cap must not move in and out of
        # the sample with the viewing angle.
        radius = max(3, min(14, int(np.sqrt(area) * 0.05)))
        band = surface & ~_erode(mask, radius + 2)
        if np.count_nonzero(band) >= max(MIN_USABLE_PIXELS, int(area * 0.06)):
            surface, region = band, "bag_skin_band"

    pixels = frame_bgr[surface].astype(np.float32) / 255.0
    value = pixels.max(axis=1)
    saturation = (value - pixels.min(axis=1)) / np.maximum(value, 1e-8)
    glare = (value >= GLARE_VALUE) & (saturation <= GLARE_MAX_SATURATION)
    shadow = value <= SHADOW_VALUE
    if glare.mean() >= MAX_EXCLUDED_SHARE:
        glare = np.zeros_like(glare)
    if shadow.mean() >= MAX_EXCLUDED_SHARE:
        shadow = np.zeros_like(shadow)
    usable = ~glare & ~shadow
    evidence = ColourEvidence(UNKNOWN, 0.0, state="insufficient_unoccluded_pixels",
                              usable_pixels=int(usable.sum()), glare_pixels=int(glare.sum()),
                              shadow_pixels=int(shadow.sum()), sampled_region=region)
    if usable.sum() < max(MIN_USABLE_PIXELS, MIN_USABLE_FRACTION * pixels.shape[0]):
        return evidence
    pixels, saturation = pixels[usable], saturation[usable]
    raw_pixels = pixels

    if background_bgr is not None and background_bgr.shape == frame_bgr.shape:
        gains = _illumination_gains(frame_bgr, background_bgr, mask)
        if gains is not None:
            pixels = np.clip(pixels * gains.astype(np.float32), 0.0, 1.0)
            evidence.illumination_gains = [round(float(g), 4) for g in gains]
        behind = background_bgr[surface][usable].astype(np.float32) / 255.0
        see_through = float(np.mean(np.abs(raw_pixels - behind).max(axis=1) <= SEE_THROUGH_DIFFERENCE))
        if see_through >= SEE_THROUGH_SHARE:
            # Mostly the floor, seen through it -- or an object the colour of
            # the floor. A bag is called transparent; anything else uncertain.
            evidence.colour = TRANSPARENT if _is_bag(label) else UNKNOWN
            evidence.confidence = round(see_through, 4)
            evidence.state = "see_through_or_background_coloured"
            return evidence

    codes, _ = _categorise_object(pixels, frame_bgr, mask)
    counts = np.bincount(codes, minlength=len(_CATEGORIES)).astype(np.float64)
    shares = counts / counts.sum()
    evidence.shares = {_CATEGORIES[i]: round(float(s), 4) for i, s in enumerate(shares) if s >= 0.01}
    order = np.argsort(shares)[::-1]
    winner = int(order[0])
    chromatic = np.array([name not in NEUTRAL for name in _CATEGORIES])
    chromatic_share = float(shares[chromatic].sum())
    if _CATEGORIES[winner] in NEUTRAL and chromatic_share >= CHROMATIC_SURFACE_SHARE \
            and shares[winner] < 0.5:
        best_chromatic = int(np.argmax(np.where(chromatic, shares, -1.0)))
        if shares[best_chromatic] / max(chromatic_share, 1e-9) >= CHROMATIC_AGREEMENT:
            winner = best_chromatic
    share = float(shares[winner])
    evidence.confidence = round(share, 4)
    if share < MIN_DOMINANT_SHARE:
        second = float(shares[order[1]]) if len(order) > 1 else 0.0
        evidence.colour = MIXED if share + second >= 0.5 else UNKNOWN
        evidence.state = "no_dominant_colour"
        return evidence
    evidence.colour, evidence.state = _CATEGORIES[winner], "solid"
    evidence.secondary = [_CATEGORIES[int(i)] for i in order[:3]
                          if int(i) != winner and shares[int(i)] >= SECONDARY_MIN_SHARE][:2] or None
    # Very dark pixels are kept out of the DOMINANT colour (they may be shadow), but a large dark
    # region is real information -- the black panels of a striped hamper -- so it is reported as a
    # secondary colour, with the caveat recorded.
    if _CATEGORIES[winner] != "black" and evidence.shadow_pixels >= SECONDARY_MIN_SHARE * (
            evidence.usable_pixels + evidence.shadow_pixels) and "black" not in (evidence.secondary or []):
        evidence.secondary = ((evidence.secondary or []) + ["black"])[:2]
        evidence.secondary_note = "black: a large very dark region (black part, or deep shadow)"

    for index in order:
        name = _CATEGORIES[int(index)]
        if int(index) == winner or name in NEUTRAL:
            continue
        if shares[index] >= ACCENT_MIN_SHARE and counts[index] >= ACCENT_MIN_PIXELS \
                and float(np.mean(saturation[codes == index])) >= ACCENT_MIN_SATURATION:
            evidence.accent, evidence.accent_share = name, round(float(shares[index]), 4)
        break
    return evidence
