from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calibrator_gui import COURT_LENGTH_M, COURT_WIDTH_M, ROOT, VideoSource


OUT_DIR = ROOT / "mapping_debug"
PX_PER_M = 200          # 200 px/m → 5600 × 3000 output (full court)
BORDER_MARGIN = 20      # ignore source pixels this close to frame edge


def load_frame(camera: dict[str, Any]) -> np.ndarray | None:
    video_path = ROOT / camera.get("video", "")
    if not video_path.exists():
        return None
    source = VideoSource(video_path)
    frame = source.read_at(float(camera.get("frame_time", 0.0)))
    source.close()
    return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)


def build_topdown(camera: dict[str, Any], px_per_m: int = PX_PER_M) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp the calibration frame to a top-down court view using cv2.warpPerspective.

    M maps each output pixel (px, py)  →  source pixel (u, v) via:
        [u·w, v·w, w] = H_world_to_image @ [px/px_per_m, py/px_per_m, 1]
    """
    out_w = int(round(COURT_LENGTH_M * px_per_m))
    out_h = int(round(COURT_WIDTH_M  * px_per_m))

    h_world_to_image = camera.get("projection", {}).get("court_homography_world_to_image")
    frame = load_frame(camera)
    if frame is None or h_world_to_image is None:
        blank = np.full((out_h, out_w, 3), 32, dtype=np.uint8)
        return blank, np.zeros((out_h, out_w), dtype=np.uint8)

    H = np.asarray(h_world_to_image, dtype=float)
    # Scale matrix: converts output pixel coords to world coords before applying H
    S_inv = np.diag([1.0 / px_per_m, 1.0 / px_per_m, 1.0])
    M = H @ S_inv  # (3×3): output-pixel → source-pixel (used with WARP_INVERSE_MAP)

    src_h, src_w = frame.shape[:2]

    # --- warp frame to top-down view ---
    output = cv2.warpPerspective(
        frame, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(32, 32, 32),
    )

    # --- validity mask: only pixels whose source falls inside the frame (minus border) ---
    src_valid = np.zeros((src_h, src_w), dtype=np.uint8)
    m = BORDER_MARGIN
    src_valid[m : src_h - m, m : src_w - m] = 255
    mask = cv2.warpPerspective(
        src_valid, M, (out_w, out_h),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return output, mask


def draw_court_overlay(img: np.ndarray, px_per_m: int = PX_PER_M) -> np.ndarray:
    """Burn FIBA court lines onto a top-down BGR image."""
    out = img.copy()
    L, W = COURT_LENGTH_M, COURT_WIDTH_M
    white = (255, 255, 255)
    th = max(2, round(px_per_m / 50))   # line thickness scales with resolution

    def px(x: float, y: float) -> tuple[int, int]:
        return (int(round(x * px_per_m)), int(round(y * px_per_m)))

    # Court outline + midcourt
    cv2.rectangle(out, px(0, 0), px(L, W), white, th)
    cv2.line(out, px(L / 2, 0), px(L / 2, W), white, th)

    # Center circle (r = 1.8 m)
    cv2.circle(out, px(L / 2, W / 2), round(1.8 * px_per_m), white, th)

    for bx, d in ((0.0, 1), (L, -1)):
        ft_x  = bx + d * 5.8          # free-throw line
        lw    = 4.9                    # lane width
        y1    = W / 2 - lw / 2
        y2    = W / 2 + lw / 2
        rim_x = bx + d * 1.575        # basket center (approx)

        # Lane boundary + FT line
        cv2.line(out, px(bx, y1),  px(ft_x, y1), white, th)
        cv2.line(out, px(bx, y2),  px(ft_x, y2), white, th)
        cv2.line(out, px(ft_x, y1), px(ft_x, y2), white, th)

        # FT semicircle (radius 1.8 m) — full circle, same as real court markings
        cv2.circle(out, px(ft_x, W / 2), round(1.8 * px_per_m), white, th)

        # Restricted area arc (r = 1.25 m)
        cv2.circle(out, px(rim_x, W / 2), round(1.25 * px_per_m), white, th)

        # 3-point corner lines (0.9 m from each sideline, up to arc)
        arc_r = 6.75
        corner_y_near = 0.9
        corner_y_far  = W - 0.9
        # Straight parts
        cv2.line(out, px(bx, corner_y_near), px(bx + d * 3.0, corner_y_near), white, th)
        cv2.line(out, px(bx, corner_y_far),  px(bx + d * 3.0, corner_y_far),  white, th)

        # Arc (polyline approximation)
        arc_pts = []
        for i in range(181):
            angle = math.radians(i)
            ax = rim_x + arc_r * math.cos(angle) * (-d)
            ay = W / 2  + arc_r * math.sin(angle)
            if corner_y_near <= ay <= corner_y_far:
                arc_pts.append(px(ax, ay))
        if len(arc_pts) >= 2:
            cv2.polylines(out, [np.array(arc_pts, dtype=np.int32)], False, white, th)

    return out


def blend_cameras(
    textures: dict[str, np.ndarray],
    masks:    dict[str, np.ndarray],
    priority: list[str] = ("cam2", "cam1"),
) -> tuple[np.ndarray, np.ndarray]:
    """Merge top-down textures; first camera in *priority* wins in overlap regions."""
    ref = textures[next(iter(textures))]
    mix      = np.full_like(ref, 32)
    mix_mask = np.zeros(ref.shape[:2], dtype=np.uint8)
    for name in priority:
        if name not in textures:
            continue
        valid = masks[name] > 128
        mix[valid]      = textures[name][valid]
        mix_mask[valid] = 255
    return mix, mix_mask


def main() -> int:
    payload = json.loads((ROOT / "basket_calibration.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(exist_ok=True)

    textures: dict[str, np.ndarray] = {}
    masks:    dict[str, np.ndarray] = {}

    for name, camera in payload.get("cameras", {}).items():
        print(f"Procesando {name}…", end=" ", flush=True)
        texture, mask = build_topdown(camera, PX_PER_M)
        textures[name] = texture
        masks[name]    = mask

        base      = OUT_DIR / f"{name}_topdown_{PX_PER_M}px"
        annotated = draw_court_overlay(texture, PX_PER_M)

        cv2.imwrite(str(base.with_suffix(".jpg")),
                    texture, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(OUT_DIR / f"{name}_annotated_{PX_PER_M}px.jpg"),
                    annotated, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(OUT_DIR / f"{name}_mask.png"), mask)

        cov = mask.mean() / 255.0
        h, w = texture.shape[:2]
        print(f"{w}×{h} px  cobertura {cov:.1%}")

    if len(textures) >= 2:
        print("Mezclando cámaras…", end=" ", flush=True)
        mix, mix_mask = blend_cameras(textures, masks)
        annotated_mix = draw_court_overlay(mix, PX_PER_M)

        cv2.imwrite(str(OUT_DIR / f"mix_topdown_{PX_PER_M}px.jpg"),
                    mix, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(OUT_DIR / f"mix_annotated_{PX_PER_M}px.jpg"),
                    annotated_mix, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(OUT_DIR / "mix_mask.png"), mix_mask)

        cov = mix_mask.mean() / 255.0
        print(f"cobertura total {cov:.1%}")

    print(f"\nArchivos guardados en: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
