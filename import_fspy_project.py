from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_fspy(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"fspy":
        raise ValueError(f"No parece un archivo .fspy valido: {path}")

    version, state_size, image_size = struct.unpack("<III", data[4:16])
    state_start = 16
    state_end = state_start + state_size
    image_end = state_end + image_size
    if image_end > len(data):
        raise ValueError("El tamano declarado en el .fspy no coincide con el archivo")

    state = json.loads(data[state_start:state_end].decode("utf-8"))
    return {
        "source_file": str(path),
        "project_file_version": version,
        "state_size": state_size,
        "image_size": image_size,
        "state": state,
        "cameraParameters": state.get("cameraParameters"),
    }


def main() -> int:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        paths = sorted((ROOT / "fspy").glob("*.fspy"))
    if not paths:
        print("No hay archivos .fspy. Guardalos en la carpeta fspy o pasalos como argumento.")
        return 1

    output: dict[str, Any] = {}
    for path in paths:
        parsed = read_fspy(path)
        output[path.stem] = parsed
        status = "OK" if parsed.get("cameraParameters") else "sin cameraParameters"
        print(f"{path.name}: {status}")

    out_path = ROOT / "fspy" / "fspy_import.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Importacion guardada en {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
