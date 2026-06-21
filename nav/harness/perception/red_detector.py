"""Locate the agent's red position marker on the minimap.

The unified client renders the agent as a saturated-red triangle on the
minimap. Each decision step we find its centroid to track the agent's
pixel position. Detection is HSV-threshold + largest-blob centroid, with a
nearest-HSV-pixel fallback so a frame never returns "not found" (the loop
needs *some* position estimate every step).

Detection defaults live in :mod:`nav.config` (``RED_DOT_*``); callers
normally use them via the keyword defaults here and only override for
debugging.
"""

import cv2
import numpy as np

from nav.config import (
    RED_DOT_HUE_TOL_DEG,
    RED_DOT_MIN_BLOB_AREA,
    RED_DOT_SAT_TOL_FRAC,
    RED_DOT_TARGET_HUE_DEG,
    RED_DOT_TARGET_SAT,
    RED_DOT_TARGET_VAL,
    RED_DOT_VAL_TOL_FRAC,
)


def detect_red_point(
    img_bgr,
    target_h_deg: float = RED_DOT_TARGET_HUE_DEG,
    target_s: float = RED_DOT_TARGET_SAT,
    target_v: float = RED_DOT_TARGET_VAL,
    hue_tol_deg: float = RED_DOT_HUE_TOL_DEG,
    sat_tol_frac: float = RED_DOT_SAT_TOL_FRAC,
    val_tol_frac: float = RED_DOT_VAL_TOL_FRAC,
    min_blob_area: int = RED_DOT_MIN_BLOB_AREA,
    draw: bool = True,
):
    """Detect the red agent marker in a BGR image.

    Returns ``(result_bgr, center, mask)`` where ``center`` is the ``(x, y)``
    pixel centroid (never None — falls back to the nearest-HSV pixel),
    ``result_bgr`` is a copy with the detection drawn (or None if
    ``draw=False``), and ``mask`` is the binary threshold mask used.
    """
    # Convert target HSV (human scale) to OpenCV HSV scale.
    H_t = int(round((target_h_deg % 360) / 2.0))  # 0..179
    S_t = int(round(np.clip(target_s, 0, 1) * 255))  # 0..255
    V_t = int(round(np.clip(target_v, 0, 1) * 255))  # 0..255

    h_tol = int(round(hue_tol_deg / 2.0))  # tolerance in 0..179 space
    s_tol = max(10, int(round(S_t * sat_tol_frac)))  # small floor for robustness
    v_tol = max(10, int(round(V_t * val_tol_frac)))

    def clip255(x):
        return int(np.clip(x, 0, 255))

    h_low = (H_t - h_tol) % 180
    h_high = (H_t + h_tol) % 180
    s_low, s_high = clip255(S_t - s_tol), clip255(S_t + s_tol)
    v_low, v_high = clip255(V_t - v_tol), clip255(V_t + v_tol)

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if h_low <= h_high:
        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, lower, upper)
    else:
        # Hue wrap-around: [h_low..179] ∪ [0..h_high]
        lower1 = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper1 = np.array([179, s_high, v_high], dtype=np.uint8)
        lower2 = np.array([0, s_low, v_low], dtype=np.uint8)
        upper2 = np.array([h_high, s_high, v_high], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, lower1, upper1) | cv2.inRange(img_hsv, lower2, upper2)

    # Clean up the mask.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    result = img_bgr.copy() if draw else None

    # Largest contour (blob) above the area threshold.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center = None
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        idx = int(np.argmax(areas))
        if areas[idx] >= min_blob_area:
            M = cv2.moments(contours[idx])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_center = (cx, cy)
                if draw:
                    cv2.circle(result, (cx, cy), 6, (0, 255, 0), 2)
                    cv2.drawContours(result, [contours[idx]], -1, (0, 255, 0), 1)

    # Fallback: nearest HSV pixel to the target (weighted, hue dominant).
    if best_center is None:
        h, s, v = cv2.split(img_hsv.astype(np.int16))
        dh = np.abs(h - H_t)
        dh = np.minimum(dh, 180 - dh) / 180.0
        ds = np.abs(s - S_t) / 255.0
        dv = np.abs(v - V_t) / 255.0
        D = 2.0 * dh + 0.75 * ds + 0.75 * dv
        min_loc = np.unravel_index(np.argmin(D), D.shape)  # (y, x)
        cy, cx = int(min_loc[0]), int(min_loc[1])
        best_center = (cx, cy)
        if draw:
            cv2.circle(result, (cx, cy), 6, (0, 255, 255), 2)  # yellow = fallback

    return result, best_center, mask
