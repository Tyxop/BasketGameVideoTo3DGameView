from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None


ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "videos"


def cv2_metadata(path: Path) -> dict[str, Any]:
    if cv2 is None:
        return {"available": False, "error": "opencv-python no esta instalado"}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"available": False, "error": "OpenCV no pudo abrir el video"}
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "available": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps if fps else None,
    }


def run_json_tool(command: list[str]) -> dict[str, Any] | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding="utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout}


def external_metadata(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if which("exiftool"):
        data["exiftool"] = run_json_tool(["exiftool", "-json", "-n", str(path)])
    if which("ffprobe"):
        data["ffprobe"] = run_json_tool(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
    return data


def hachoir_metadata(path: Path) -> dict[str, Any]:
    try:
        from hachoir.metadata import extractMetadata
        from hachoir.parser import createParser
    except ImportError as error:
        return {"available": False, "error": str(error)}

    parser = createParser(str(path))
    if not parser:
        return {"available": False, "error": "hachoir no pudo crear parser"}
    with parser:
        metadata = extractMetadata(parser)
    if metadata is None:
        return {"available": False, "error": "hachoir no encontro metadata"}
    values: dict[str, Any] = {"available": True}
    for item in metadata.exportPlaintext():
        if ": " not in item:
            continue
        key, value = item.split(": ", 1)
        values[key.strip("- ")] = value
    return values


def inspect_video(path: Path) -> dict[str, Any]:
    return {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "cv2": cv2_metadata(path),
        "hachoir": hachoir_metadata(path),
        "external_tools": external_metadata(path),
    }


def main() -> int:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        paths = sorted(VIDEO_DIR.glob("cam*.*"))
    report = {path.name: inspect_video(path) for path in paths if path.exists()}
    output = ROOT / "video_metadata.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadata guardada en {output}")
    for name, item in report.items():
        cv2_data = item.get("cv2", {})
        print(name, cv2_data.get("width"), "x", cv2_data.get("height"), cv2_data.get("fps"), "fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
