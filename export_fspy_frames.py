from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
VIDEOS = {
    "cam1": ROOT / "videos" / "cam1.mov",
    "cam2": ROOT / "videos" / "cam2.mov",
}
OUT_DIR = ROOT / "fspy"


def read_calibration_times() -> dict[str, float]:
    path = ROOT / "basket_calibration.json"
    if not path.exists():
        return {name: 0.0 for name in VIDEOS}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: float(payload.get("cameras", {}).get(name, {}).get("frame_time", 0.0))
        for name in VIDEOS
    }


def export_frame(camera: str, video_path: Path, time_seconds: float) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_index = max(0, int(round(time_seconds * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer frame {frame_index} de {video_path}")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{camera}_fspy_frame.jpg"
    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    return {
        "camera": camera,
        "video": str(video_path.relative_to(ROOT)),
        "frame_path": str(out_path.relative_to(ROOT)),
        "time_seconds": time_seconds,
        "frame_index": frame_index,
        "fps": fps,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
    }


def main() -> int:
    times = read_calibration_times()
    manifest = {"frames": []}
    for camera, video_path in VIDEOS.items():
        if not video_path.exists():
            video_path = ROOT / "videos" / f"{camera}.mp4"
        manifest["frames"].append(export_frame(camera, video_path, times.get(camera, 0.0)))

    manifest_path = OUT_DIR / "fspy_frames_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frames exportados en {OUT_DIR}")
    for item in manifest["frames"]:
        print(f"{item['camera']}: {item['frame_path']} ({item['width']}x{item['height']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
