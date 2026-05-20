from __future__ import annotations

import json
import math
import sys
import tkinter as tk
from dataclasses import asdict, dataclass, field, replace as dc_replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

try:
    import cv2
except ImportError:
    cv2 = None


ROOT = Path(__file__).resolve().parent


def find_video(camera_name: str) -> Path:
    video_dir = ROOT / "videos"
    for suffix in (".mov", ".MOV", ".mp4", ".MP4"):
        candidate = video_dir / f"{camera_name}{suffix}"
        if candidate.exists():
            return candidate
    return video_dir / f"{camera_name}.mp4"


VIDEOS = {
    "cam1": find_video("cam1"),
    "cam2": find_video("cam2"),
}

COURT_LENGTH_M = 28.0   # default fallback (FIBA); use court_spec in instance methods
COURT_WIDTH_M  = 15.0


@dataclass
class CourtSpec:
    name: str
    length_m: float
    width_m: float
    ft_depth_m: float       # distancia línea tiro libre desde cada fondo
    lane_width_m: float     # ancho del carril/pintura
    circle_r_m: float = 1.80
    board_offset_m: float = 1.20   # distancia tablero–línea de fondo
    board_w_m: float = 1.83
    board_h_m: float = 1.05
    board_z_m: float = 3.05
    rim_offset_m: float = 0.15
    rim_r_m: float = 0.225


COURT_PRESETS: dict[str, CourtSpec] = {
    "FIBA":  CourtSpec("FIBA",  28.00, 15.00, 5.80, 4.90),
    "NBA":   CourtSpec("NBA",   28.65, 15.24, 5.79, 4.88, 1.83, 1.22),
    "NCAA":  CourtSpec("NCAA",  28.65, 15.24, 5.79, 3.66, 1.83, 1.22),
    "Local": CourtSpec("Local", 26.00, 14.00, 5.80, 4.90),
}


def _court_lines_for(spec: CourtSpec) -> list[tuple]:
    L, W, ft = spec.length_m, spec.width_m, spec.ft_depth_m
    hw = spec.lane_width_m / 2
    return [
        ("baseline_back",      "Fondo cercano",                  "x", 0.0,      "calibration"),
        ("baseline_front",     "Fondo lejano",                   "x", L,        "calibration"),
        ("midcourt",           "Medio campo",                    "x", L / 2,    "calibration"),
        ("ft_near",            "Tiro libre cercano",             "x", ft,       "calibration"),
        ("ft_far",             "Tiro libre lejano",              "x", L - ft,   "calibration"),
        ("midcourt_vp",        "Medio campo solo fuga",          "x", L / 2,    "vanishing"),
        ("baseline_back_vp",   "Fondo cercano solo fuga",        "x", 0.0,      "vanishing"),
        ("baseline_front_vp",  "Fondo lejano solo fuga",         "x", L,        "vanishing"),
        ("sideline_left",      "Lateral izquierdo",              "y", 0.0,      "calibration"),
        ("sideline_right",     "Lateral derecho",                "y", W,        "calibration"),
        ("sideline_left_vp",   "Lateral izq solo fuga",          "y", 0.0,      "vanishing"),
        ("sideline_right_vp",  "Lateral der solo fuga",          "y", W,        "vanishing"),
        ("ft_lane_left",       f"Carril izq (y={W/2-hw:.2f}m)", "y", W/2 - hw, "calibration"),
        ("ft_lane_right",      f"Carril der (y={W/2+hw:.2f}m)", "y", W/2 + hw, "calibration"),
        ("ft_lane_left_vp",    "Carril izq solo fuga",           "y", W/2 - hw, "vanishing"),
        ("ft_lane_right_vp",   "Carril der solo fuga",           "y", W/2 + hw, "vanishing"),
    ]

COURT_LINE_COLORS = {
    "baseline_back": "#ff5f57",
    "baseline_front": "#ff9f43",
    "midcourt": "#f8e71c",
    "ft_near": "#e040fb",
    "ft_far": "#b000d0",
    "midcourt_vp": "#d6c900",
    "baseline_back_vp": "#b94742",
    "baseline_front_vp": "#c2772e",
    "sideline_left": "#33d6a6",
    "sideline_right": "#3fa7ff",
    "sideline_left_vp": "#278f72",
    "sideline_right_vp": "#2d77b6",
    "ft_lane_left": "#00e5ff",
    "ft_lane_right": "#00bcd4",
    "ft_lane_left_vp": "#0097a7",
    "ft_lane_right_vp": "#006978",
}

GRID_COLOR = "#9ed8ff"


def _build_court_3d_lines_for(spec: CourtSpec) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    L, W = spec.length_m, spec.width_m
    lines: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    N = 24

    # Court boundary
    corners = [(0.0, 0.0, 0.0), (L, 0.0, 0.0), (L, W, 0.0), (0.0, W, 0.0)]
    for i in range(4):
        lines.append((corners[i], corners[(i + 1) % 4]))

    # Midcourt line
    lines.append(((L / 2, 0.0, 0.0), (L / 2, W, 0.0)))

    CR = spec.circle_r_m
    for i in range(N):
        a1 = 2 * math.pi * i / N
        a2 = 2 * math.pi * (i + 1) / N
        lines.append((
            (L / 2 + CR * math.cos(a1), W / 2 + CR * math.sin(a1), 0.0),
            (L / 2 + CR * math.cos(a2), W / 2 + CR * math.sin(a2), 0.0),
        ))

    bo = spec.board_offset_m
    BW, BH, BZ = spec.board_w_m, spec.board_h_m, spec.board_z_m
    for bx, direction in ((bo, 1.0), (L - bo, -1.0)):
        cy = W / 2.0

        # Backboard rectangle
        bb = [
            (bx, cy - BW / 2, BZ - BH / 2),
            (bx, cy + BW / 2, BZ - BH / 2),
            (bx, cy + BW / 2, BZ + BH / 2),
            (bx, cy - BW / 2, BZ + BH / 2),
        ]
        for i in range(4):
            lines.append((bb[i], bb[(i + 1) % 4]))

        # Inner target square on backboard
        SW, SH = 0.59, 0.45
        sq = [
            (bx, cy - SW / 2, BZ - SH / 2),
            (bx, cy + SW / 2, BZ - SH / 2),
            (bx, cy + SW / 2, BZ + SH / 2),
            (bx, cy - SW / 2, BZ + SH / 2),
        ]
        for i in range(4):
            lines.append((sq[i], sq[(i + 1) % 4]))

        # Support post
        lines.append(((bx, cy, BZ - BH / 2), (bx, cy, 0.0)))

        # Rim
        RX = bx + direction * spec.rim_offset_m
        RR = spec.rim_r_m
        RN = 20
        for i in range(RN):
            a1 = 2 * math.pi * i / RN
            a2 = 2 * math.pi * (i + 1) / RN
            lines.append((
                (RX + RR * math.cos(a1), cy + RR * math.sin(a1), BZ),
                (RX + RR * math.cos(a2), cy + RR * math.sin(a2), BZ),
            ))

        # Free throw lane
        LW = spec.lane_width_m
        FTX = bx + direction * spec.ft_depth_m
        baseline = 0.0 if direction > 0 else L
        lines += [
            ((baseline, cy - LW / 2, 0.0), (FTX, cy - LW / 2, 0.0)),
            ((baseline, cy + LW / 2, 0.0), (FTX, cy + LW / 2, 0.0)),
            ((FTX, cy - LW / 2, 0.0), (FTX, cy + LW / 2, 0.0)),
        ]

        # Free throw semicircle
        for i in range(N):
            a1 = 2 * math.pi * i / N
            a2 = 2 * math.pi * (i + 1) / N
            lines.append((
                (FTX + CR * math.cos(a1), cy + CR * math.sin(a1), 0.0),
                (FTX + CR * math.cos(a2), cy + CR * math.sin(a2), 0.0),
            ))

    return lines


@dataclass
class LensProfile:
    name: str
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    cx_offset_px: float = 0.0
    cy_offset_px: float = 0.0
    focal_mm: float = 0.0   # full-frame equiv. focal length; 0 = auto (use FOV hint)

    def has_distortion(self) -> bool:
        return any(abs(v) > 1e-9 for v in (self.k1, self.k2, self.k3, self.p1, self.p2))

    def dist_array(self) -> Any:
        import numpy as _np
        return _np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=float)

    def focal_px_for(self, img_w: int) -> float | None:
        """Pixel focal length from focal_mm (full-frame equiv). None if unknown (focal_mm==0)."""
        if self.focal_mm > 0.1:
            return (img_w / 2) * self.focal_mm / 18.0  # 18 = half of 36 mm FF sensor
        return None


LENS_PRESETS: dict[str, LensProfile] = {
    "Sin corrección":        LensProfile("Sin corrección"),
    "iPhone 1x (std)":       LensProfile("iPhone 1x (std)",       k1=-0.045, k2=0.012,              focal_mm=26.0),
    "iPhone 0.5x (ultra)":  LensProfile("iPhone 0.5x (ultra)",   k1=-0.280, k2=0.090, k3=-0.010,  focal_mm=13.0),
    "iPhone 2x (tele)":     LensProfile("iPhone 2x (tele)",      k1=-0.020, k2=0.005,              focal_mm=52.0),
    "Android 1x (std)":     LensProfile("Android 1x (std)",      k1=-0.055, k2=0.018,              focal_mm=27.0),
    "Android ultra-wide":   LensProfile("Android ultra-wide",    k1=-0.220, k2=0.070,              focal_mm=13.0),
    "GoPro wide":           LensProfile("GoPro wide",            k1=-0.330, k2=0.150, k3=-0.030,  focal_mm=14.0),
}

FOCAL_MM_PRESETS: list[tuple[str, float]] = [
    ("Auto (estimado por FOV)", 0.0),
    ("12 mm  (ultra gran angular)", 12.0),
    ("14 mm", 14.0),
    ("16 mm", 16.0),
    ("20 mm", 20.0),
    ("24 mm", 24.0),
    ("26 mm  (iPhone 1x)", 26.0),
    ("28 mm", 28.0),
    ("35 mm", 35.0),
    ("50 mm  (normal)", 50.0),
    ("52 mm  (iPhone 2x)", 52.0),
    ("85 mm  (retrato)", 85.0),
    ("105 mm", 105.0),
]


CAMERA_RIG_POSITION = (COURT_LENGTH_M / 2, COURT_WIDTH_M + 2.0, 1.1)
CAMERA_RIG_BASELINE_M = 0.45
# cam1 = iPhone SE 25 mm equiv (~71.5° H-FOV); cam2 = 13 mm equiv (~108.4° H-FOV)
CAMERA_FOV_HINTS: dict[str, float] = {
    "cam1": 2 * math.degrees(math.atan(18.0 / 25.0)),  # ≈ 71.5°
    "cam2": 2 * math.degrees(math.atan(18.0 / 13.0)),  # ≈ 108.4°
}
DEFAULT_CAMERA_FOV_DEG = 70.0
DEFAULT_IMAGE_SIZE = (1920, 1080)
CAMERA_HEIGHT_M = 1.1
CAMERA_TARGETS = {
    "cam1": (COURT_LENGTH_M / 4, COURT_WIDTH_M / 2, 0.0),
    "cam2": (3 * COURT_LENGTH_M / 4, COURT_WIDTH_M / 2, 0.0),
}

WALL_PLANES = [
    ("left_wall", "Pared izquierda"),
    ("right_wall", "Pared derecha"),
    ("back_wall", "Pared fondo"),
    ("front_wall", "Pared frontal"),
    ("custom_wall", "Pared personalizada"),
]


@dataclass
class MarkPoint:
    id: str
    type: str
    image: dict[str, float]
    frame_time: float
    label: str
    name: str
    world: dict[str, float] | None = None
    plane: str | None = None
    line: str | None = None


@dataclass
class CameraCalibration:
    video: str
    frame_time: float = 0.0
    points: list[MarkPoint] = field(default_factory=list)


class VideoSource:
    def __init__(self, path: Path) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV no esta instalado. Instala requirements.txt para leer video MP4.")
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.duration = self.frame_count / self.fps if self.fps and self.frame_count else 0.0

    def read_at(self, time_seconds: float) -> Image.Image:
        frame_index = max(0, int(round(time_seconds * self.fps)))
        if self.frame_count:
            frame_index = min(frame_index, self.frame_count - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"No se pudo leer frame {frame_index} de {self.path.name}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def close(self) -> None:
        self.capture.release()


class CalibratorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("3D Basket Match - Calibrador")
        self.geometry("1320x820")
        self.minsize(1100, 680)

        self.tool = tk.StringVar(value="court")
        self.active_camera = tk.StringVar(value="cam1")
        self.show_3d_overlay = tk.BooleanVar(value=False)
        self.show_court_lines = tk.BooleanVar(value=True)
        self.camera_heights: dict[str, tk.DoubleVar] = {
            "cam1": tk.DoubleVar(value=CAMERA_HEIGHT_M),
            "cam2": tk.DoubleVar(value=CAMERA_HEIGHT_M),
        }
        self.legend_expanded = tk.BooleanVar(value=False)
        self.court_spec: CourtSpec = dc_replace(COURT_PRESETS["FIBA"])
        self.lens_profiles: dict[str, LensProfile] = {
            "cam1": dc_replace(LENS_PRESETS["Sin corrección"]),
            "cam2": dc_replace(LENS_PRESETS["Sin corrección"]),
        }
        self.selected_id: str | None = None
        self.drag_id: str | None = None
        self.sources: dict[str, VideoSource] = {}
        self.current_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.image_rect = (0, 0, 1, 1)
        self.view_zoom = 1.0
        self.view_pan = [0.0, 0.0]
        self.pan_start: tuple[int, int, float, float] | None = None

        self.cameras = {
            camera: CameraCalibration(video=str(path.relative_to(ROOT)))
            for camera, path in VIDEOS.items()
        }

        self._build_ui()
        self._load_sources()
        self._autoload_default_json()
        self._switch_camera("cam1")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(12, 10))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text="Calibrador de marcas, tablero y paredes", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(top, text="Pulsa sobre el frame para crear puntos. Arrastra un punto para recolocarlo.").grid(row=1, column=0, sticky="w")
        ttk.Button(top, text="Guardar JSON", command=self._save_default_json).grid(row=0, column=1, rowspan=2, padx=4)
        ttk.Button(top, text="Importar JSON", command=self._import_json).grid(row=0, column=2, rowspan=2, padx=4)
        ttk.Button(top, text="Exportar JSON", command=self._export_json).grid(row=0, column=3, rowspan=2, padx=4)
        ttk.Button(top, text="Vista 3D", command=self._open_3d_view).grid(row=0, column=4, rowspan=2, padx=4)
        ttk.Checkbutton(top, text="Líneas campo", variable=self.show_court_lines, command=self._render).grid(row=0, column=5, rowspan=2, padx=(12, 4))
        ttk.Checkbutton(top, text="Superponer 3D", variable=self.show_3d_overlay, command=self._render).grid(row=0, column=6, rowspan=2, padx=(0, 4))
        ttk.Label(top, text="Alt. cam (m)").grid(row=0, column=7, sticky="s", padx=(4, 2))
        self.height_spin = ttk.Spinbox(top, from_=0.5, to=20.0, increment=0.1, textvariable=self.camera_heights["cam1"], width=6)
        self.height_spin.grid(row=1, column=7, sticky="n", padx=(4, 2))
        for var in self.camera_heights.values():
            var.trace_add("write", lambda *_: self.after_idle(self._render))

        ttk.Label(top, text="Cancha").grid(row=0, column=8, sticky="s", padx=(8, 2))
        self.preset_var = tk.StringVar(value="FIBA")
        preset_combo = ttk.Combobox(top, state="readonly", textvariable=self.preset_var,
                                    values=list(COURT_PRESETS.keys()) + ["Personalizar..."], width=12)
        preset_combo.grid(row=1, column=8, sticky="n", padx=(8, 2))
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        main = ttk.Frame(self, padding=(12, 0, 12, 12))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        camera_bar = ttk.Frame(main)
        camera_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for camera in ("cam1", "cam2"):
            ttk.Radiobutton(
                camera_bar,
                text=camera.upper(),
                value=camera,
                variable=self.active_camera,
                command=lambda c=camera: self._switch_camera(c),
                style="Toolbutton",
            ).pack(side="left", padx=(0, 4))
        self.lens_btn = ttk.Button(camera_bar, text="Lente…", command=lambda: self._open_lens_config(self.active_camera.get()))
        self.lens_btn.pack(side="left", padx=(8, 0))

        viewer = ttk.Frame(main)
        viewer.grid(row=1, column=0, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(viewer, bg="#050608", highlightthickness=1, highlightbackground="#3a4352")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<Button-1>", self._on_pointer_down)
        self.canvas.bind("<B1-Motion>", self._on_pointer_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _event: self._end_drag())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1.12))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, 1 / 1.12))
        self.canvas.bind("<Button-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan_view)
        self.canvas.bind("<ButtonRelease-2>", lambda _event: self._end_pan())

        controls = ttk.Frame(viewer, padding=(0, 8, 0, 0))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Tiempo").grid(row=0, column=0, padx=(0, 8))
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = ttk.Scale(controls, from_=0, to=1, variable=self.time_var, command=self._on_time_scale)
        self.time_scale.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.time_entry = ttk.Entry(controls, width=8)
        self.time_entry.grid(row=0, column=2, padx=(0, 8))
        self.time_entry.insert(0, "0.00")
        self.time_entry.bind("<Return>", lambda _event: self._seek_from_entry())
        ttk.Button(controls, text="Cargar frame", command=self._seek_from_entry).grid(row=0, column=3)
        ttk.Button(controls, text="Reset vista", command=self._reset_view).grid(row=0, column=4, padx=(8, 0))

        side = ttk.Frame(main, width=360)
        side.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(12, 0))
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)

        self._build_tools(side)
        self._build_point_list(side)
        self._build_projection_panel(side)

    def _build_tools(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Herramienta", padding=10)
        box.grid(row=0, column=0, sticky="ew")
        for index, (value, text) in enumerate((("court", "Campo"), ("backboard", "Tablero"), ("wall", "Pared"), ("select", "Mover"))):
            ttk.Radiobutton(box, text=text, value=value, variable=self.tool, command=self._refresh_tool_options, style="Toolbutton").grid(
                row=index // 2, column=index % 2, sticky="ew", padx=3, pady=3
            )
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=1)

        self.options = ttk.LabelFrame(parent, text="Datos del punto", padding=10)
        self.options.grid(row=1, column=0, sticky="ew", pady=10)
        self._refresh_tool_options()

    # ── Court spec helpers ──────────────────────────────────────────────────

    def _court_line_defs(self) -> list[tuple]:
        return _court_lines_for(self.court_spec)

    def _court_3d_lines(self) -> list[tuple]:
        return _build_court_3d_lines_for(self.court_spec)

    def _on_preset_selected(self, _event: tk.Event | None = None) -> None:
        name = self.preset_var.get()
        if name == "Personalizar...":
            self.preset_var.set(self.court_spec.name if self.court_spec.name in COURT_PRESETS else "FIBA")
            self._open_court_custom()
            return
        self.court_spec = dc_replace(COURT_PRESETS[name])
        self._refresh_tool_options()
        self._render()

    def _open_court_custom(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Medidas personalizadas")
        dlg.resizable(False, False)
        dlg.grab_set()

        spec = self.court_spec
        fields = [
            ("Largo cancha (m)",       "length_m",       spec.length_m),
            ("Ancho cancha (m)",       "width_m",        spec.width_m),
            ("Dist. tiro libre (m)",   "ft_depth_m",     spec.ft_depth_m),
            ("Ancho carril (m)",       "lane_width_m",   spec.lane_width_m),
            ("Radio círculo centro (m)","circle_r_m",    spec.circle_r_m),
            ("Tablero a fondo (m)",    "board_offset_m", spec.board_offset_m),
        ]
        vars_: dict[str, tk.DoubleVar] = {}
        for row, (label, key, val) in enumerate(fields):
            ttk.Label(dlg, text=label).grid(row=row, column=0, sticky="w", padx=12, pady=4)
            v = tk.DoubleVar(value=val)
            vars_[key] = v
            ttk.Spinbox(dlg, from_=0.1, to=99.0, increment=0.01, textvariable=v, width=8).grid(row=row, column=1, padx=12, pady=4)

        def _apply() -> None:
            try:
                new_spec = dc_replace(
                    self.court_spec,
                    name="Personalizado",
                    **{k: float(v.get()) for k, v in vars_.items()},
                )
            except Exception:
                messagebox.showerror("Error", "Valor no válido.", parent=dlg)
                return
            self.court_spec = new_spec
            self.preset_var.set("Personalizado" if "Personalizado" not in COURT_PRESETS else self.court_spec.name)
            self._refresh_tool_options()
            self._render()
            dlg.destroy()

        btn_row = len(fields)
        ttk.Button(dlg, text="Aplicar", command=_apply).grid(row=btn_row, column=0, pady=12, padx=12, sticky="ew")
        ttk.Button(dlg, text="Cancelar", command=dlg.destroy).grid(row=btn_row, column=1, pady=12, padx=12, sticky="ew")

    # ── Lens profile ─────────────────────────────────────────────────────────

    def _open_lens_config(self, camera: str) -> None:
        dlg = tk.Toplevel(self)
        dlg.title(f"Lente — {camera.upper()}")
        dlg.resizable(False, False)
        dlg.grab_set()

        prof = self.lens_profiles[camera]
        preset_var = tk.StringVar(value=prof.name if prof.name in LENS_PRESETS else "Sin corrección")

        ttk.Label(dlg, text="Preset:").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        preset_combo = ttk.Combobox(dlg, state="readonly", textvariable=preset_var,
                                    values=list(LENS_PRESETS.keys()), width=22)
        preset_combo.grid(row=0, column=1, columnspan=2, padx=12, pady=(12, 4), sticky="ew")

        # ── Focal length row ─────────────────────────────────────────────────
        focal_row = 1
        ttk.Label(dlg, text="Focal equiv. FF (mm):").grid(row=focal_row, column=0, sticky="w", padx=12, pady=(8, 3))
        focal_var = tk.DoubleVar(value=prof.focal_mm)
        focal_spin = ttk.Spinbox(dlg, from_=0.0, to=500.0, increment=1.0, textvariable=focal_var,
                                 width=8, format="%.1f")
        focal_spin.grid(row=focal_row, column=1, padx=(12, 4), pady=(8, 3), sticky="w")

        focal_quick_var = tk.StringVar()
        focal_quick = ttk.Combobox(dlg, state="readonly", textvariable=focal_quick_var,
                                   values=[label for label, _ in FOCAL_MM_PRESETS], width=22)
        focal_quick.grid(row=focal_row, column=2, padx=(0, 12), pady=(8, 3), sticky="ew")

        def _sync_focal_quick(*_: Any) -> None:
            val = focal_var.get()
            for label, mm in FOCAL_MM_PRESETS:
                if abs(mm - val) < 0.1:
                    focal_quick_var.set(label)
                    return
            focal_quick_var.set("Personalizado")

        def _focal_quick_selected(*_: Any) -> None:
            idx = focal_quick.current()
            if idx >= 0:
                focal_var.set(FOCAL_MM_PRESETS[idx][1])

        focal_spin.bind("<FocusOut>", _sync_focal_quick)
        focal_spin.bind("<Return>", _sync_focal_quick)
        focal_quick.bind("<<ComboboxSelected>>", _focal_quick_selected)
        _sync_focal_quick()

        ttk.Label(dlg, text="0 = automático (estimado desde FOV de cámara)", font=("Segoe UI", 8),
                  foreground="#888").grid(row=focal_row + 1, column=0, columnspan=3, padx=12, pady=(0, 6), sticky="w")

        ttk.Separator(dlg, orient="horizontal").grid(row=focal_row + 2, column=0, columnspan=3,
                                                     sticky="ew", padx=8, pady=4)

        # ── Distortion coefficients ──────────────────────────────────────────
        dist_start = focal_row + 3
        fields = [
            ("k1  (radial 1)",   "k1",           prof.k1),
            ("k2  (radial 2)",   "k2",           prof.k2),
            ("k3  (radial 3)",   "k3",           prof.k3),
            ("p1  (tangencial)", "p1",           prof.p1),
            ("p2  (tangencial)", "p2",           prof.p2),
            ("cx offset (px)",   "cx_offset_px", prof.cx_offset_px),
            ("cy offset (px)",   "cy_offset_px", prof.cy_offset_px),
        ]
        vars_: dict[str, tk.DoubleVar] = {}
        for row_idx, (label, key, val) in enumerate(fields, start=dist_start):
            ttk.Label(dlg, text=label).grid(row=row_idx, column=0, sticky="w", padx=12, pady=3)
            v = tk.DoubleVar(value=val)
            vars_[key] = v
            sp = ttk.Spinbox(dlg, from_=-5.0, to=5.0, increment=0.001, textvariable=v, width=12, format="%.4f")
            sp.grid(row=row_idx, column=1, padx=12, pady=3, columnspan=2, sticky="w")

        def _load_preset(*_: Any) -> None:
            name = preset_var.get()
            p = LENS_PRESETS.get(name)
            if p is None:
                return
            focal_var.set(p.focal_mm)
            _sync_focal_quick()
            for key, sp_var in vars_.items():
                sp_var.set(getattr(p, key))

        preset_combo.bind("<<ComboboxSelected>>", _load_preset)

        hint_row = dist_start + len(fields)
        ttk.Label(dlg, text="(k1 negativo = distorsión barril;  0 = sin corrección)",
                  font=("Segoe UI", 8), foreground="#888").grid(
            row=hint_row, column=0, columnspan=3, padx=12, pady=(4, 8), sticky="w")

        def _apply() -> None:
            try:
                new_prof = LensProfile(
                    name=preset_var.get(),
                    focal_mm=float(focal_var.get()),
                    **{k: float(v.get()) for k, v in vars_.items()},
                )
            except Exception:
                messagebox.showerror("Error", "Valor no válido.", parent=dlg)
                return
            self.lens_profiles[camera] = new_prof
            self._render()
            dlg.destroy()

        btn_row = hint_row + 1
        ttk.Button(dlg, text="Aplicar", command=_apply).grid(row=btn_row, column=0, pady=12, padx=12, sticky="ew")
        ttk.Button(dlg, text="Cancelar", command=dlg.destroy).grid(row=btn_row, column=1, pady=12, padx=12, sticky="ew")

    def _focal_px(self, camera: str, img_w: int) -> float:
        """Pixel focal length: uses focal_mm from lens profile if set, else FOV hint fallback."""
        prof = self.lens_profiles.get(camera)
        if prof is not None:
            fp = prof.focal_px_for(img_w)
            if fp is not None:
                return fp
        fov = CAMERA_FOV_HINTS.get(camera, DEFAULT_CAMERA_FOV_DEG)
        return (img_w / 2) / math.tan(math.radians(fov) / 2)

    def _undistort_image(self, image: Image.Image, camera: str) -> Image.Image:
        if cv2 is None:
            return image
        prof = self.lens_profiles.get(camera)
        if prof is None or not prof.has_distortion():
            return image
        w, h = image.size
        f = self._focal_px(camera, w)
        cx = w / 2 + prof.cx_offset_px
        cy = h / 2 + prof.cy_offset_px
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = prof.dist_array()
        img_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        undistorted = cv2.undistort(img_np, K, dist)
        return Image.fromarray(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))

    # ── Tool options ─────────────────────────────────────────────────────────

    def _refresh_tool_options(self) -> None:
        for child in self.options.winfo_children():
            child.destroy()

        if self.tool.get() == "court":
            self.court_line = tk.StringVar(value=self._court_line_defs()[0][0])
            line_combo = ttk.Combobox(self.options, state="readonly", values=[label for _, label, *_ in self._court_line_defs()])
            line_combo.current(0)
            line_combo.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
            line_combo.bind("<<ComboboxSelected>>", lambda _event: self._set_court_line(line_combo.current()))
            self.line_combo = line_combo
            ttk.Label(
                self.options,
                text="Marca dos puntos sobre la misma linea visible. Cada linea tiene su color; con 4 intersecciones la rejilla es exacta, con 3 lineas se muestra una rejilla provisional.",
                wraplength=320,
            ).grid(row=1, column=0, columnspan=3, sticky="w")
            legend_wrapper = ttk.Frame(self.options)
            legend_wrapper.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))

            legend_list = ttk.Frame(legend_wrapper)
            for idx, (code, label, _axis, _value, usage) in enumerate(self._court_line_defs()):
                tk.Label(legend_list, text="■", fg=COURT_LINE_COLORS[code], font=("Segoe UI", 10)).grid(row=idx, column=0, sticky="w")
                suffix = " (solo fuga)" if usage == "vanishing" else ""
                ttk.Label(legend_list, text=f"{label}{suffix}").grid(row=idx, column=1, sticky="w", padx=(4, 0))

            def _toggle_legend(btn=None, lst=legend_list) -> None:
                expanded = self.legend_expanded.get()
                if expanded:
                    lst.pack(fill="x")
                    if btn:
                        btn.configure(text="▲ Ocultar leyenda")
                else:
                    lst.pack_forget()
                    if btn:
                        btn.configure(text="▼ Mostrar leyenda")

            toggle_btn = ttk.Button(legend_wrapper, text="▼ Mostrar leyenda", command=lambda: (
                self.legend_expanded.set(not self.legend_expanded.get()),
                _toggle_legend(toggle_btn),
            ))
            toggle_btn.pack(anchor="w")
        elif self.tool.get() == "wall":
            self.wall_plane = tk.StringVar(value=WALL_PLANES[0][0])
            plane = ttk.Combobox(self.options, state="readonly", values=[label for _, label in WALL_PLANES])
            plane.current(0)
            plane.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
            plane.bind("<<ComboboxSelected>>", lambda _event: self.wall_plane.set(WALL_PLANES[plane.current()][0]))
            self.wall_vars = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)]
            for index, label in enumerate(("X m", "Y m", "Z m")):
                ttk.Label(self.options, text=label).grid(row=1, column=index, sticky="w")
                ttk.Entry(self.options, textvariable=self.wall_vars[index], width=8).grid(row=2, column=index, sticky="ew", padx=2)
        else:
            ttk.Label(self.options, text="Pulsa o arrastra puntos ya creados.").grid(row=0, column=0, sticky="w")

        for col in range(3):
            self.options.columnconfigure(col, weight=1)

    def _build_point_list(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Puntos", padding=10)
        box.grid(row=2, column=0, sticky="nsew")
        parent.rowconfigure(2, weight=1)
        buttons = ttk.Frame(box)
        buttons.pack(fill="x", pady=(0, 8))
        ttk.Button(buttons, text="Deshacer", command=self._undo).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Borrar", command=self._delete_selected).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Limpiar camara", command=self._clear_camera).pack(side="left")

        list_frame = ttk.Frame(box)
        list_frame.pack(fill="both", expand=True)
        self.point_list = tk.Listbox(list_frame, height=12, activestyle="none")
        self.point_list.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.point_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.point_list.configure(yscrollcommand=scrollbar.set)
        self.point_list.bind("<<ListboxSelect>>", self._select_from_list)

    def _build_projection_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Proyeccion", padding=10)
        box.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.projection_label = ttk.Label(box, text="", wraplength=320)
        self.projection_label.pack(fill="x")
        self.top_view = tk.Canvas(box, width=320, height=180, bg="#10151d", highlightthickness=1, highlightbackground="#3a4352")
        self.top_view.pack(fill="x", pady=(8, 0))
        ttk.Button(box, text="Abrir vista 3D", command=self._open_3d_view).pack(fill="x", pady=(8, 0))

    def _load_sources(self) -> None:
        if cv2 is None:
            messagebox.showerror(
                "Falta OpenCV",
                "La GUI esta creada, pero para leer MP4 hace falta OpenCV.\n\nEjecuta:\npython -m pip install -r requirements.txt",
            )
            return
        for camera, path in VIDEOS.items():
            if path.exists():
                self.sources[camera] = VideoSource(path)

    def _autoload_default_json(self) -> None:
        path = ROOT / "basket_calibration.json"
        if not path.exists():
            return
        try:
            self._apply_payload(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            return

    def _switch_camera(self, camera: str) -> None:
        self.active_camera.set(camera)
        self.selected_id = None
        self.height_spin.configure(textvariable=self.camera_heights[camera])
        source = self.sources.get(camera)
        if source:
            self.time_scale.configure(to=max(source.duration, 1.0))
        self.time_var.set(self.cameras[camera].frame_time)
        self._load_frame(self.cameras[camera].frame_time)

    def _load_frame(self, time_seconds: float) -> None:
        camera = self.active_camera.get()
        source = self.sources.get(camera)
        self.cameras[camera].frame_time = float(time_seconds)
        self.time_entry.delete(0, "end")
        self.time_entry.insert(0, f"{time_seconds:.2f}")
        if not source:
            self.current_image = None
            self._render()
            return
        try:
            self.current_image = source.read_at(time_seconds)
        except RuntimeError as error:
            messagebox.showerror("Error de video", str(error))
            self.current_image = None
        self._render()

    def _on_time_scale(self, raw_value: str) -> None:
        value = float(raw_value)
        self.after_cancel(getattr(self, "_seek_after_id", "after#0")) if hasattr(self, "_seek_after_id") else None
        self._seek_after_id = self.after(80, lambda: self._load_frame(value))

    def _seek_from_entry(self) -> None:
        try:
            value = float(self.time_entry.get().replace(",", "."))
        except ValueError:
            return
        self.time_var.set(value)
        self._load_frame(value)

    def _set_court_line(self, index: int) -> None:
        self.court_line.set(self._court_line_defs()[index][0])

    def _active_points(self) -> list[MarkPoint]:
        return self.cameras[self.active_camera.get()].points

    def _on_pointer_down(self, event: tk.Event) -> None:
        hit = self._hit_test(event.x, event.y)
        if hit:
            self.selected_id = hit.id
            self.drag_id = hit.id
            self._render()
            return
        if self.tool.get() == "select":
            return
        image_point = self._screen_to_image(event.x, event.y)
        if image_point is None:
            return
        self._add_point(image_point)

    def _on_pointer_drag(self, event: tk.Event) -> None:
        if not self.drag_id:
            return
        image_point = self._screen_to_image(event.x, event.y)
        if image_point is None:
            return
        for point in self._active_points():
            if point.id == self.drag_id:
                point.image = {"x": round(image_point[0], 2), "y": round(image_point[1], 2)}
                break
        self._render()

    def _end_drag(self) -> None:
        self.drag_id = None

    def _add_point(self, image_point: tuple[float, float]) -> None:
        camera = self.active_camera.get()
        points = self._active_points()
        point_type = self.tool.get()
        point_id = f"{camera}_{len(points) + 1}_{int(self.cameras[camera].frame_time * 1000)}"
        frame_time = round(self.cameras[camera].frame_time, 3)
        image = {"x": round(image_point[0], 2), "y": round(image_point[1], 2)}

        if point_type == "court":
            line_index = self.line_combo.current()
            code, label, axis, value, usage = self._court_line_defs()[line_index]
            count = 1 + sum(1 for item in points if item.type == "court" and item.line == code)
            world = {"axis": axis, "value": value, "z": 0.0, "usage": usage}
            point = MarkPoint(point_id, point_type, image, frame_time, f"{code}_{count}", f"{label} punto {count}", world, line=code)
        elif point_type == "backboard":
            count = 1 + sum(1 for point in points if point.type == "backboard")
            point = MarkPoint(point_id, point_type, image, frame_time, f"backboard_square_{count}", f"Tablero esquina {count}")
        else:
            plane_label = dict(WALL_PLANES).get(self.wall_plane.get(), self.wall_plane.get())
            count = 1 + sum(1 for point in points if point.type == "wall" and point.plane == self.wall_plane.get())
            world = {"x": self.wall_vars[0].get(), "y": self.wall_vars[1].get(), "z": self.wall_vars[2].get()}
            point = MarkPoint(point_id, point_type, image, frame_time, f"{self.wall_plane.get()}_{count}", f"{plane_label} punto {count}", world, self.wall_plane.get())

        points.append(point)
        self.selected_id = point.id
        self._render()

    def _hit_test(self, x: float, y: float) -> MarkPoint | None:
        for point in reversed(self._active_points()):
            sx, sy = self._image_to_screen(point.image["x"], point.image["y"])
            if math.hypot(sx - x, sy - y) <= 10:
                return point
        return None

    def _screen_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        ix, iy, iw, ih = self.image_rect
        if not self.current_image or x < ix or y < iy or x > ix + iw or y > iy + ih:
            return None
        return ((x - ix) / iw * self.current_image.width, (y - iy) / ih * self.current_image.height)

    def _image_to_screen(self, x: float, y: float) -> tuple[float, float]:
        ix, iy, iw, ih = self.image_rect
        if not self.current_image:
            return ix, iy
        return ix + x / self.current_image.width * iw, iy + y / self.current_image.height * ih

    def _on_mousewheel(self, event: tk.Event) -> None:
        factor = 1.12 if event.delta > 0 else 1 / 1.12
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, screen_x: float, screen_y: float, factor: float) -> None:
        if not self.current_image:
            return
        before = self._screen_to_image(screen_x, screen_y)
        self.view_zoom = max(0.35, min(8.0, self.view_zoom * factor))
        if before is not None:
            width = max(self.canvas.winfo_width(), 1)
            height = max(self.canvas.winfo_height(), 1)
            fit = min(width / self.current_image.width, height / self.current_image.height)
            draw_w = self.current_image.width * fit * self.view_zoom
            draw_h = self.current_image.height * fit * self.view_zoom
            center_x = (width - draw_w) / 2
            center_y = (height - draw_h) / 2
            self.view_pan[0] = screen_x - center_x - before[0] / self.current_image.width * draw_w
            self.view_pan[1] = screen_y - center_y - before[1] / self.current_image.height * draw_h
        self._render()

    def _start_pan(self, event: tk.Event) -> None:
        self.pan_start = (event.x, event.y, self.view_pan[0], self.view_pan[1])
        self.canvas.configure(cursor="fleur")

    def _pan_view(self, event: tk.Event) -> None:
        if not self.pan_start:
            return
        start_x, start_y, pan_x, pan_y = self.pan_start
        self.view_pan[0] = pan_x + event.x - start_x
        self.view_pan[1] = pan_y + event.y - start_y
        self._render()

    def _end_pan(self) -> None:
        self.pan_start = None
        self.canvas.configure(cursor="")

    def _reset_view(self) -> None:
        self.view_zoom = 1.0
        self.view_pan = [0.0, 0.0]
        self._render()

    def _render(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)

        if self.current_image:
            scale = min(width / self.current_image.width, height / self.current_image.height) * self.view_zoom
            draw_w = max(1, int(self.current_image.width * scale))
            draw_h = max(1, int(self.current_image.height * scale))
            x = int((width - draw_w) / 2 + self.view_pan[0])
            y = int((height - draw_h) / 2 + self.view_pan[1])
            self.image_rect = (x, y, draw_w, draw_h)
            resized = self.current_image.resize((draw_w, draw_h), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized)
            self.canvas.create_image(x, y, image=self.tk_image, anchor="nw")
            self.canvas.create_rectangle(x, y, x + draw_w, y + draw_h, outline="#ffffff", width=1)
        else:
            self.canvas.create_text(
                width // 2,
                height // 2,
                text="No hay frame cargado.\nInstala OpenCV o revisa los videos.",
                fill="#f3f4f6",
                justify="center",
                font=("Segoe UI", 14),
            )

        if self.show_court_lines.get():
            self._draw_plane_grid()
            self._draw_connections()
        self._draw_points()
        self._draw_3d_court_overlay()
        self._refresh_point_list()
        self._draw_top_view()
        self._refresh_projection_status()

    def _draw_connections(self) -> None:
        line_usage = {code: usage for code, _label, _axis, _value, usage in self._court_line_defs()}
        hide_vp = self.show_3d_overlay.get()
        court_lines = self._court_line_groups()
        for code, points in court_lines.items():
            if hide_vp and line_usage.get(code) == "vanishing":
                continue
            self._draw_court_line(points, COURT_LINE_COLORS.get(code, "#42d392"))

        backboard = [point for point in self._active_points() if point.type == "backboard"]
        self._draw_path(backboard, "#57a7ff", closed=len(backboard) >= 3)

        planes: dict[str, list[MarkPoint]] = {}
        for point in self._active_points():
            if point.type == "wall" and point.plane:
                planes.setdefault(point.plane, []).append(point)
        for points in planes.values():
            self._draw_path(points, "#f7bd52", closed=len(points) >= 3)

    def _draw_court_line(self, points: list[MarkPoint], color: str) -> None:
        if len(points) < 2:
            return
        p1, p2 = points[0], points[1]
        x1, y1 = self._image_to_screen(p1.image["x"], p1.image["y"])
        x2, y2 = self._image_to_screen(p2.image["x"], p2.image["y"])
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=3)

    def _draw_path(self, points: list[MarkPoint], color: str, closed: bool) -> None:
        if len(points) < 2:
            return
        coords: list[float] = []
        for point in points:
            coords.extend(self._image_to_screen(point.image["x"], point.image["y"]))
        if closed:
            coords.extend(coords[:2])
        self.canvas.create_line(*coords, fill=color, width=2)

    def _draw_plane_grid(self) -> None:
        homography = self._court_homography_world_to_image()
        if homography is None:
            return
        h = np.asarray(homography, dtype=float)

        def project(world_x: float, world_y: float) -> tuple[float, float] | None:
            point = h @ np.asarray([world_x, world_y, 1.0])
            if abs(point[2]) < 1e-9:
                return None
            return self._image_to_screen(float(point[0] / point[2]), float(point[1] / point[2]))

        L, W = self.court_spec.length_m, self.court_spec.width_m
        for x in np.arange(0.0, L + 0.001, 1.0):
            a = project(float(x), 0.0)
            b = project(float(x), W)
            if a and b:
                color = "#ffffff" if abs(x % 7.0) < 1e-6 else GRID_COLOR
                self.canvas.create_line(a[0], a[1], b[0], b[1], fill=color, width=1, dash=(2, 6))

        for y in np.arange(0.0, W + 0.001, 1.0):
            a = project(0.0, float(y))
            b = project(L, float(y))
            if a and b:
                color = "#ffffff" if y in (0.0, W) else GRID_COLOR
                self.canvas.create_line(a[0], a[1], b[0], b[1], fill=color, width=1, dash=(2, 6))

    def _draw_points(self) -> None:
        colors = {"court": "#42d392", "backboard": "#57a7ff", "wall": "#f7bd52"}
        for point in self._active_points():
            x, y = self._image_to_screen(point.image["x"], point.image["y"])
            radius = 7 if point.id == self.selected_id else 5
            color = COURT_LINE_COLORS.get(point.line, colors.get(point.type, "#ffffff"))
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="#10151d", width=2)
            self.canvas.create_text(x + 10, y - 10, text=point.label, fill="#ffffff", anchor="w", font=("Segoe UI", 9))

    def _refresh_point_list(self) -> None:
        current_selection = self.selected_id
        self.point_list.delete(0, "end")
        for point in self._active_points():
            self.point_list.insert("end", f"{point.type}: {point.name} ({point.image['x']:.0f}, {point.image['y']:.0f})")
            if point.id == current_selection:
                self.point_list.selection_clear(0, "end")
                self.point_list.selection_set("end")
                self.point_list.see("end")

    def _select_from_list(self, _event: tk.Event) -> None:
        selection = self.point_list.curselection()
        if not selection:
            return
        points = self._active_points()
        if selection[0] < len(points):
            self.selected_id = points[selection[0]].id
            self._render()

    def _draw_top_view(self) -> None:
        self.top_view.delete("all")
        width = int(self.top_view["width"])
        height = int(self.top_view["height"])
        pad = 18
        L, W = self.court_spec.length_m, self.court_spec.width_m
        scale = min((width - pad * 2) / L, (height - pad * 2) / W)
        ox = (width - L * scale) / 2
        oy = (height - W * scale) / 2
        self.top_view.create_rectangle(ox, oy, ox + L * scale, oy + W * scale, outline="#5f6b7a", width=2)
        self.top_view.create_line(ox + L / 2 * scale, oy, ox + L / 2 * scale, oy + W * scale, fill="#5f6b7a")
        for code, _label, axis, value, usage in self._court_line_defs():
            if usage != "calibration":
                continue
            color = COURT_LINE_COLORS[code]
            if axis == "x":
                x = ox + value * scale
                self.top_view.create_line(x, oy, x, oy + W * scale, fill=color, width=2)
            else:
                y = oy + value * scale
                self.top_view.create_line(ox, y, ox + L * scale, y, fill=color, width=2)
        for item in self._court_intersections():
            x = ox + item["world"]["x"] * scale
            y = oy + item["world"]["y"] * scale
            self.top_view.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#42d392", outline="")

    def _refresh_projection_status(self) -> None:
        line_count = sum(1 for points in self._court_line_groups().values() if len(points) >= 2)
        intersection_count = len(self._court_intersections())
        vanishing = self._court_vanishing_points()
        board_count = sum(1 for point in self._active_points() if point.type == "backboard")
        wall_count = sum(1 for point in self._active_points() if point.type == "wall")
        homography = self._court_homography_image_to_world()
        if homography is None:
            text = f"Lineas de campo listas: {line_count}. Intersecciones: {intersection_count}/4. Marca 3 lineas compatibles para rejilla provisional o 4 intersecciones para plano exacto."
        elif len(self._court_intersections()) < 4:
            text = f"Rejilla provisional con 3 lineas. Ajusta con zoom y confirma con una cuarta linea cuando se vea. Tablero: {board_count}/4. Paredes: {wall_count}."
        else:
            text = f"Plano con rejilla listo. Lineas: {line_count}. Intersecciones usadas: {intersection_count}. Tablero: {board_count}/4. Paredes: {wall_count}."
        vp_text = ", ".join(axis.upper() for axis, value in vanishing.items() if value) or "ninguno"
        text = f"{text} Puntos de fuga: {vp_text}."
        self.projection_label.configure(text=text)

    def _court_line_groups(self, points: list[MarkPoint] | None = None) -> dict[str, list[MarkPoint]]:
        groups: dict[str, list[MarkPoint]] = {}
        source_points = points if points is not None else self._active_points()
        for point in source_points:
            if point.type == "court" and point.line:
                groups.setdefault(point.line, []).append(point)
        return groups

    def _court_intersections(self, points: list[MarkPoint] | None = None) -> list[dict[str, Any]]:
        line_meta = {code: {"label": label, "axis": axis, "value": value, "usage": usage} for code, label, axis, value, usage in self._court_line_defs()}
        groups = self._court_line_groups(points)
        image_lines = {
            code: self._image_line_from_points(points[:2])
            for code, points in groups.items()
            if len(points) >= 2 and code in line_meta and line_meta[code]["usage"] == "calibration"
        }
        intersections = []
        for code_a, line_a in image_lines.items():
            meta_a = line_meta[code_a]
            for code_b, line_b in image_lines.items():
                meta_b = line_meta[code_b]
                if meta_a["axis"] != "x" or meta_b["axis"] != "y":
                    continue
                image = self._line_intersection(line_a, line_b)
                if image is None:
                    continue
                intersections.append(
                    {
                        "label": f"{code_a}__{code_b}",
                        "image": {"x": round(image[0], 2), "y": round(image[1], 2)},
                        "world": {"x": float(meta_a["value"]), "y": float(meta_b["value"]), "z": 0.0},
                    }
                )
        return intersections

    def _court_vanishing_points(self, points: list[MarkPoint] | None = None) -> dict[str, dict[str, float] | None]:
        line_meta = {code: {"axis": axis, "value": value, "usage": usage} for code, _label, axis, value, usage in self._court_line_defs()}
        groups = self._court_line_groups(points)
        image_lines = [
            (code, self._image_line_from_points(group[:2]))
            for code, group in groups.items()
            if len(group) >= 2 and code in line_meta
        ]
        result: dict[str, dict[str, float] | None] = {"x": None, "y": None}
        for axis in ("x", "y"):
            parallel_lines = [line for code, line in image_lines if line is not None and line_meta[code]["axis"] == axis]
            intersections = []
            for i, line_a in enumerate(parallel_lines):
                for line_b in parallel_lines[i + 1 :]:
                    point = self._line_intersection(line_a, line_b)
                    if point is not None:
                        intersections.append(point)
            if intersections:
                result[axis] = {
                    "x": round(sum(point[0] for point in intersections) / len(intersections), 2),
                    "y": round(sum(point[1] for point in intersections) / len(intersections), 2),
                    "pairs": len(intersections),
                }
        return result

    def _undistort_pt(self, camera: str, x: float, y: float) -> tuple[float, float]:
        if cv2 is None:
            return x, y
        prof = self.lens_profiles.get(camera)
        if prof is None or not prof.has_distortion():
            return x, y
        img = self.current_image
        w = img.width if img else DEFAULT_IMAGE_SIZE[0]
        h = img.height if img else DEFAULT_IMAGE_SIZE[1]
        f = self._focal_px(camera, w)
        cx = w / 2 + prof.cx_offset_px
        cy = h / 2 + prof.cy_offset_px
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        pts = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.undistortPoints(pts, K, prof.dist_array(), P=K)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def _image_line_from_points(self, points: list[MarkPoint]) -> tuple[float, float, float] | None:
        if len(points) < 2:
            return None
        p1, p2 = points[0], points[1]
        camera = self.active_camera.get()
        x1, y1 = self._undistort_pt(camera, p1.image["x"], p1.image["y"])
        x2, y2 = self._undistort_pt(camera, p2.image["x"], p2.image["y"])
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        norm = math.hypot(a, b)
        if norm < 1e-9:
            return None
        return a / norm, b / norm, c / norm

    def _line_intersection(self, line_a: tuple[float, float, float] | None, line_b: tuple[float, float, float] | None) -> tuple[float, float] | None:
        if line_a is None or line_b is None:
            return None
        a1, b1, c1 = line_a
        a2, b2, c2 = line_b
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-9:
            return None
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det
        return x, y

    def _court_homography_image_to_world(self, points: list[MarkPoint] | None = None) -> list[list[float]] | None:
        intersections = self._court_intersections(points)
        correspondences = self._court_correspondences(points, intersections)
        if len(correspondences) < 4:
            return None
        source = [item[0] for item in correspondences]
        target = [item[1] for item in correspondences]
        return self._compute_homography(source, target)

    def _court_homography_world_to_image(self, points: list[MarkPoint] | None = None) -> list[list[float]] | None:
        intersections = self._court_intersections(points)
        correspondences = self._court_correspondences(points, intersections)
        if len(correspondences) < 4:
            return None
        source = [item[1] for item in correspondences]
        target = [item[0] for item in correspondences]
        return self._compute_homography(source, target)

    def _court_correspondences(
        self,
        points: list[MarkPoint] | None,
        intersections: list[dict[str, Any]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        exact = [
            ((item["image"]["x"], item["image"]["y"]), (item["world"]["x"], item["world"]["y"]))
            for item in intersections
        ]
        if len(exact) >= 4:
            return exact

        provisional = self._provisional_three_line_correspondences(points, exact)
        return provisional if len(provisional) >= 4 else exact

    def _provisional_three_line_correspondences(
        self,
        points: list[MarkPoint] | None,
        exact: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        groups = self._court_line_groups(points)
        meta = {code: {"axis": axis, "value": value, "usage": usage} for code, _label, axis, value, usage in self._court_line_defs()}
        x_lines = [code for code, group in groups.items() if len(group) >= 2 and meta.get(code, {}).get("axis") == "x" and meta.get(code, {}).get("usage") == "calibration"]
        y_lines = [code for code, group in groups.items() if len(group) >= 2 and meta.get(code, {}).get("axis") == "y" and meta.get(code, {}).get("usage") == "calibration"]

        if len(y_lines) >= 2 and len(x_lines) >= 1:
            return self._provisional_from_two_y_one_x(groups, meta, y_lines[:2], x_lines[0], exact)
        if len(x_lines) >= 2 and len(y_lines) >= 1:
            return self._provisional_from_two_x_one_y(groups, meta, x_lines[:2], y_lines[0], exact)
        return exact

    def _provisional_from_two_y_one_x(
        self,
        groups: dict[str, list[MarkPoint]],
        meta: dict[str, dict[str, float | str]],
        y_lines: list[str],
        x_line: str,
        exact: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        line_x = float(meta[x_line]["value"])
        start = {world[1]: image for image, world in exact if abs(world[0] - line_x) < 1e-6}
        if len(start) < 2:
            return exact

        correspondences = list(exact)
        for y_code in y_lines:
            y_value = float(meta[y_code]["value"])
            anchor = start.get(y_value)
            if anchor is None:
                continue
            p1, p2 = groups[y_code][0], groups[y_code][1]
            q1 = (p1.image["x"], p1.image["y"])
            q2 = (p2.image["x"], p2.image["y"])
            sample = q2 if self._distance(q2, anchor) > self._distance(q1, anchor) else q1
            direction = (sample[0] - anchor[0], sample[1] - anchor[1])
            L = self.court_spec.length_m
            target_x = L if line_x <= L / 2 else 0.0
            scale = abs(target_x - line_x) / max(L, 1e-9)
            synthetic = (anchor[0] + direction[0] / max(scale, 1e-6), anchor[1] + direction[1] / max(scale, 1e-6))
            correspondences.append((synthetic, (target_x, y_value)))
        return correspondences

    def _provisional_from_two_x_one_y(
        self,
        groups: dict[str, list[MarkPoint]],
        meta: dict[str, dict[str, float | str]],
        x_lines: list[str],
        y_line: str,
        exact: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        line_y = float(meta[y_line]["value"])
        start = {world[0]: image for image, world in exact if abs(world[1] - line_y) < 1e-6}
        if len(start) < 2:
            return exact

        correspondences = list(exact)
        for x_code in x_lines:
            x_value = float(meta[x_code]["value"])
            anchor = start.get(x_value)
            if anchor is None:
                continue
            p1, p2 = groups[x_code][0], groups[x_code][1]
            q1 = (p1.image["x"], p1.image["y"])
            q2 = (p2.image["x"], p2.image["y"])
            sample = q2 if self._distance(q2, anchor) > self._distance(q1, anchor) else q1
            direction = (sample[0] - anchor[0], sample[1] - anchor[1])
            W = self.court_spec.width_m
            target_y = W if line_y <= W / 2 else 0.0
            scale = abs(target_y - line_y) / max(W, 1e-9)
            synthetic = (anchor[0] + direction[0] / max(scale, 1e-6), anchor[1] + direction[1] / max(scale, 1e-6))
            correspondences.append((synthetic, (x_value, target_y)))
        return correspondences

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _compute_homography(self, source: list[tuple[float, float]], target: list[tuple[float, float]]) -> list[list[float]] | None:
        if len(source) < 4 or len(target) < 4:
            return None
        rows = []
        for (u, v), (x, y) in zip(source, target):
            rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * x, v * x, x])
            rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * y, v * y, y])
        matrix = np.asarray(rows, dtype=float)
        _, _, vh = np.linalg.svd(matrix)
        h = vh[-1, :].reshape(3, 3)
        if abs(h[2, 2]) < 1e-9:
            return None
        h = h / h[2, 2]
        return h.round(8).tolist()

    def _fspy_like_intrinsics(
        self,
        image_size: dict[str, int],
        vanishing_points: dict[str, dict[str, float] | None],
    ) -> dict[str, Any]:
        width = int(image_size["width"])
        height = int(image_size["height"])
        camera = self.active_camera.get()
        lens = self.lens_profiles.get(camera, LensProfile("Sin corrección"))
        principal = np.asarray([width / 2 + lens.cx_offset_px, height / 2 + lens.cy_offset_px], dtype=float)
        vp_x = vanishing_points.get("x")
        vp_y = vanishing_points.get("y")
        source = "fallback_fov"
        focal = self._focal_px(camera, width)
        quality = "needs_two_orthogonal_vanishing_points"

        if vp_x and vp_y:
            a = np.asarray([float(vp_x["x"]), float(vp_x["y"])], dtype=float) - principal
            b = np.asarray([float(vp_y["x"]), float(vp_y["y"])], dtype=float) - principal
            focal_sq = -float(a @ b)
            if focal_sq > 1e-6:
                focal = math.sqrt(focal_sq)
                source = "orthogonal_vanishing_points"
                quality = "ok"
            else:
                quality = "invalid_vanishing_geometry_fallback_fov"

        intrinsics = np.asarray(
            [
                [focal, 0.0, principal[0]],
                [0.0, focal, principal[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        horizontal_fov = math.degrees(2 * math.atan((width / 2) / focal))
        return {
            "matrix": intrinsics,
            "source": source,
            "quality": quality,
            "focal_px": focal,
            "horizontal_fov_deg": horizontal_fov,
            "principal_point_px": {"x": float(principal[0]), "y": float(principal[1])},
        }

    def _estimate_camera_from_homography(
        self,
        homography_world_to_image: list[list[float]] | None,
        image_size: dict[str, int],
        vanishing_points: dict[str, dict[str, float] | None],
    ) -> dict[str, Any] | None:
        if homography_world_to_image is None:
            return None
        calibration = self._fspy_like_intrinsics(image_size, vanishing_points)
        intrinsics = calibration["matrix"]
        h = np.asarray(homography_world_to_image, dtype=float)
        k_inv = np.linalg.inv(intrinsics)

        def decompose(sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            b = k_inv @ (h * sign)
            scale = 2.0 / (np.linalg.norm(b[:, 0]) + np.linalg.norm(b[:, 1]))
            r1 = b[:, 0] * scale
            r2 = b[:, 1] * scale
            t_candidate = b[:, 2] * scale
            r3 = np.cross(r1, r2)
            rotation_guess = np.column_stack((r1, r2, r3))
            u, _s, vh = np.linalg.svd(rotation_guess)
            rotation_candidate = u @ vh
            if np.linalg.det(rotation_candidate) < 0:
                rotation_candidate[:, 2] *= -1
            center_candidate = -rotation_candidate.T @ t_candidate
            return rotation_candidate, t_candidate, center_candidate

        candidates = [decompose(1.0), decompose(-1.0)]
        rotation, t, center = max(candidates, key=lambda item: item[2][2])
        return {
            "status": f"estimated_from_court_homography_{calibration['source']}",
            "image_size": image_size,
            "intrinsics_source": calibration["source"],
            "quality": calibration["quality"],
            "focal_px": round(float(calibration["focal_px"]), 6),
            "horizontal_fov_deg": round(float(calibration["horizontal_fov_deg"]), 6),
            "principal_point_px": calibration["principal_point_px"],
            "intrinsics": intrinsics.round(6).tolist(),
            "rotation_world_to_camera": rotation.round(8).tolist(),
            "translation_world_to_camera": t.round(8).tolist(),
            "camera_center_world_m": {
                "x": round(float(center[0]), 6),
                "y": round(float(center[1]), 6),
                "z": round(float(center[2]), 6),
            },
            "note": "Sistema tipo fSpy: focal desde dos puntos de fuga ortogonales si existen; si falta una direccion, usa FOV fallback y marca quality como incompleta.",
        }

    def _match_backboard_to_3d(
        self,
        image_pts: list[tuple[float, float]],
        basket_x: float,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> list[tuple[list[float], list[float]]]:
        """Nearest-neighbour matching of 4 image backboard points to their 3D corners."""
        cy = self.court_spec.width_m / 2.0
        BW, BH, BZ = self.court_spec.board_w_m, self.court_spec.board_h_m, self.court_spec.board_z_m
        world_corners: list[list[float]] = [
            [basket_x, cy - BW / 2, BZ - BH / 2],
            [basket_x, cy + BW / 2, BZ - BH / 2],
            [basket_x, cy + BW / 2, BZ + BH / 2],
            [basket_x, cy - BW / 2, BZ + BH / 2],
        ]
        projected: list[np.ndarray] = []
        for p3d in world_corners:
            Pc = R @ np.array(p3d, dtype=float) + t
            if Pc[2] >= -0.05:
                return []
            ph = K @ Pc
            projected.append(np.array([ph[0] / ph[2], ph[1] / ph[2]]))
        available = list(range(4))
        result: list[tuple[list[float], list[float]]] = []
        for img_pt in image_pts:
            img_arr = np.array(img_pt, dtype=float)
            best = min(available, key=lambda i: float(np.sum((img_arr - projected[i]) ** 2)))
            result.append((list(img_pt), world_corners[best]))
            available.remove(best)
        return result

    def _estimate_camera_pnp(self, camera_name: str | None = None) -> dict[str, Any] | None:
        """Refine camera pose with solvePnP using floor intersections + backboard corners."""
        if cv2 is None:
            return None
        cam = camera_name or self.active_camera.get()
        calibration = self.cameras[cam]
        source = self.sources.get(cam)
        image_size = {
            "width": source.width if source else DEFAULT_IMAGE_SIZE[0],
            "height": source.height if source else DEFAULT_IMAGE_SIZE[1],
        }
        vanishing_pts = self._court_vanishing_points(calibration.points)
        h_w2i = self._court_homography_world_to_image(calibration.points)
        approx = self._estimate_camera_from_homography(h_w2i, image_size, vanishing_pts)
        if approx is None:
            return None

        K = np.asarray(approx["intrinsics"], dtype=float)
        R_approx = np.asarray(approx["rotation_world_to_camera"], dtype=float)
        t_approx = np.asarray(approx["translation_world_to_camera"], dtype=float)

        intersections = self._court_intersections(calibration.points)
        obj_pts: list[list[float]] = [[i["world"]["x"], i["world"]["y"], 0.0] for i in intersections]
        img_pts: list[list[float]] = [[i["image"]["x"], i["image"]["y"]] for i in intersections]

        backboard_pts = [p for p in calibration.points if p.type == "backboard"]
        if len(backboard_pts) >= 4:
            target = CAMERA_TARGETS.get(cam)
            L = self.court_spec.length_m
            basket_x = self.court_spec.board_offset_m if (target and target[0] < L / 2) else L - self.court_spec.board_offset_m
            image_bb = [(p.image["x"], p.image["y"]) for p in backboard_pts[:4]]
            matches = self._match_backboard_to_3d(image_bb, basket_x, K, R_approx, t_approx)
            for img_pt, world_pt in matches:
                img_pts.append(img_pt)
                obj_pts.append(world_pt)

        if len(obj_pts) < 6:
            return None

        obj_arr = np.array(obj_pts, dtype=np.float32)
        img_arr = np.array(img_pts, dtype=np.float32)
        K_f = K.astype(np.float32)

        if len(obj_pts) >= 8:
            ok, rvec, tvec, _inliers = cv2.solvePnPRansac(
                obj_arr, img_arr, K_f, None,
                reprojectionError=10.0, confidence=0.99,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(obj_arr, img_arr, K_f, None)

        if not ok:
            return None

        R_cv, _ = cv2.Rodrigues(rvec)
        t_cv = tvec.flatten()

        # Convert OpenCV convention (P_cam[2]>0 visible) → our flip-Z convention
        flip_z = np.diag([1.0, 1.0, -1.0])
        R = flip_z @ R_cv
        t = flip_z @ t_cv
        center = -R.T @ t

        rx, ry, rz = CAMERA_RIG_POSITION
        cx, cy_val, cz = float(center[0]), float(center[1]), float(center[2])
        if abs(cx - rx) > 20 or abs(cy_val - ry) > 12 or cz < -1 or cz > 20:
            return None

        proj, _ = cv2.projectPoints(obj_arr, rvec, tvec, K_f, None)
        err = float(np.mean(np.linalg.norm(img_arr - proj.reshape(-1, 2), axis=1)))

        horizontal_fov = math.degrees(2 * math.atan((image_size["width"] / 2) / float(K[0, 0])))
        return {
            "status": "pnp_floor_and_backboard",
            "image_size": image_size,
            "intrinsics_source": approx["intrinsics_source"],
            "quality": "ok" if err < 12 else "poor",
            "focal_px": round(float(K[0, 0]), 4),
            "horizontal_fov_deg": round(horizontal_fov, 4),
            "principal_point_px": approx["principal_point_px"],
            "intrinsics": K.round(4).tolist(),
            "rotation_world_to_camera": R.round(8).tolist(),
            "translation_world_to_camera": t.round(8).tolist(),
            "camera_center_world_m": {"x": round(cx, 4), "y": round(cy_val, 4), "z": round(cz, 4)},
            "reprojection_error_px": round(err, 3),
            "n_points": len(obj_pts),
        }

    def _get_overlay_pose(self) -> dict[str, Any] | None:
        # Try PnP (backboard + floor) first; fall back to floor-homography only
        pose = self._estimate_camera_pnp() if cv2 is not None else None

        if pose is None:
            camera = self.active_camera.get()
            calibration = self.cameras[camera]
            source = self.sources.get(camera)
            image_size = {
                "width": source.width if source else DEFAULT_IMAGE_SIZE[0],
                "height": source.height if source else DEFAULT_IMAGE_SIZE[1],
            }
            vanishing_points = self._court_vanishing_points(calibration.points)
            h_world_to_image = self._court_homography_world_to_image(calibration.points)
            pose = self._estimate_camera_from_homography(h_world_to_image, image_size, vanishing_points)

        if pose is None:
            return None

        # Apply manual height override from the spinner (if set and PnP not available)
        if pose.get("status", "").startswith("estimated_from_court"):
            try:
                override_z = float(self.camera_heights[self.active_camera.get()].get())
            except (tk.TclError, ValueError):
                override_z = 0.0
            if override_z > 0.0:
                center = pose["camera_center_world_m"]
                new_center = np.asarray([center["x"], center["y"], override_z], dtype=float)
                R = np.asarray(pose["rotation_world_to_camera"], dtype=float)
                t_new = (-R @ new_center).tolist()
                pose = dict(pose)
                pose["translation_world_to_camera"] = t_new
                pose["camera_center_world_m"] = {"x": float(new_center[0]), "y": float(new_center[1]), "z": override_z}
        return pose

    def _project_world_to_screen(
        self,
        world_point: tuple[float, float, float],
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        dist: np.ndarray | None = None,
    ) -> tuple[float, float] | None:
        P_cam = R @ np.asarray(world_point, dtype=float) + t
        # Decomposition uses camera-looks-along-minus-Z convention:
        # visible points have P_cam[2] < 0; reject anything at or behind the image plane.
        if P_cam[2] >= -0.05:
            return None
        p_h = K @ P_cam
        u = float(p_h[0] / p_h[2])
        v = float(p_h[1] / p_h[2])
        if dist is not None:
            # Re-apply forward distortion so the overlay lands on the distorted displayed image.
            # The homography was built from undistorted coords; projection gives undistorted pixels.
            f  = float(K[0, 0])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            x_n = (u - cx) / f
            y_n = (v - cy) / f
            r2 = x_n * x_n + y_n * y_n
            # dist order: [k1, k2, p1, p2, k3]
            k1, k2, p1_d, p2_d, k3 = (float(dist[i]) for i in range(5))
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x_d = x_n * radial + 2.0 * p1_d * x_n * y_n + p2_d * (r2 + 2.0 * x_n * x_n)
            y_d = y_n * radial + p1_d * (r2 + 2.0 * y_n * y_n) + 2.0 * p2_d * x_n * y_n
            u = f * x_d + cx
            v = f * y_d + cy
        return self._image_to_screen(u, v)

    def _draw_3d_court_overlay(self) -> None:
        if not self.show_3d_overlay.get():
            return
        pose = self._get_overlay_pose()
        if pose is None:
            return
        K = np.asarray(pose["intrinsics"], dtype=float)
        R = np.asarray(pose["rotation_world_to_camera"], dtype=float)
        t = np.asarray(pose["translation_world_to_camera"], dtype=float)
        camera = self.active_camera.get()
        prof = self.lens_profiles.get(camera)
        dist = prof.dist_array() if (prof and prof.has_distortion()) else None

        for start, end in self._court_3d_lines():
            s = self._project_world_to_screen(start, K, R, t, dist=dist)
            e = self._project_world_to_screen(end, K, R, t, dist=dist)
            if s is None or e is None:
                continue
            z_avg = (start[2] + end[2]) / 2.0
            is_rim = z_avg > 2.8 and abs(start[1] - self.court_spec.width_m / 2) < 1.5 and abs(end[1] - self.court_spec.width_m / 2) < 1.5
            if is_rim:
                color, width, dash = "#ff8c00", 2, ()
            elif z_avg < 0.01:
                color, width, dash = "#00d4ff", 1, (4, 5)
            else:
                color, width, dash = "#8be8ff", 2, ()
            self.canvas.create_line(s[0], s[1], e[0], e[1], fill=color, width=width, dash=dash)

    def _undo(self) -> None:
        if self._active_points():
            self._active_points().pop()
        self.selected_id = None
        self._render()

    def _delete_selected(self) -> None:
        if not self.selected_id:
            return
        camera = self.active_camera.get()
        self.cameras[camera].points = [point for point in self._active_points() if point.id != self.selected_id]
        self.selected_id = None
        self._render()

    def _clear_camera(self) -> None:
        if messagebox.askyesno("Limpiar camara", f"Borrar todos los puntos de {self.active_camera.get().upper()}?"):
            self.cameras[self.active_camera.get()].points.clear()
            self.selected_id = None
            self._render()

    def _export_payload(self) -> dict[str, Any]:
        cameras: dict[str, Any] = {}
        for camera, calibration in self.cameras.items():
            data = asdict(calibration)
            intersections = self._court_intersections(calibration.points)
            homography_image_to_world = self._court_homography_image_to_world(calibration.points)
            homography_world_to_image = self._court_homography_world_to_image(calibration.points)
            source = self.sources.get(camera)
            image_size = {
                "width": source.width if source else DEFAULT_IMAGE_SIZE[0],
                "height": source.height if source else DEFAULT_IMAGE_SIZE[1],
            }
            data["image_size"] = image_size
            data["camera_height_m"] = round(self.camera_heights[camera].get(), 3)
            lens = self.lens_profiles.get(camera, LensProfile("Sin corrección"))
            data["lens"] = asdict(lens)
            vanishing_points = self._court_vanishing_points(calibration.points)
            floor_pose = self._estimate_camera_from_homography(homography_world_to_image, image_size, vanishing_points)
            pnp_pose = self._estimate_camera_pnp(camera) if cv2 is not None else None
            data["projection"] = {
                "court_intersections": intersections,
                "vanishing_points": vanishing_points,
                "court_homography_image_to_world": homography_image_to_world,
                "court_homography_world_to_image": homography_world_to_image,
                "camera_pose": floor_pose,
                "camera_pose_pnp": pnp_pose,
                "note": "camera_pose: homografia suelo + fugas ortogonales. camera_pose_pnp: solvePnP suelo + esquinas tablero (mas preciso en Z).",
            }
            cameras[camera] = data
        return {
            "schema": "basket-match-calibration-v1",
            "court": {"name": self.court_spec.name, "length_m": self.court_spec.length_m, "width_m": self.court_spec.width_m, "ft_depth_m": self.court_spec.ft_depth_m, "lane_width_m": self.court_spec.lane_width_m},
            "camera_rig": {
                "type": "stereo_midcourt_opposite_side_approx",
                "position_world_m": {"x": CAMERA_RIG_POSITION[0], "y": CAMERA_RIG_POSITION[1], "z": CAMERA_RIG_POSITION[2]},
                "baseline_m": CAMERA_RIG_BASELINE_M,
                "targets_world_m": {
                    name: {"x": target[0], "y": target[1], "z": target[2]}
                    for name, target in CAMERA_TARGETS.items()
                },
                "note": "Aproximacion visual: rig en la linea de medio campo, al lateral opuesto, con camaras proximas rotadas hacia tableros opuestos.",
            },
            "cameras": cameras,
        }

    def _export_json(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Exportar calibracion",
            initialdir=ROOT,
            initialfile="basket_calibration.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self._export_payload(), indent=2), encoding="utf-8")

    def _save_default_json(self) -> None:
        path = ROOT / "basket_calibration.json"
        path.write_text(json.dumps(self._export_payload(), indent=2), encoding="utf-8")
        messagebox.showinfo("JSON guardado", f"Calibracion guardada en:\n{path}")

    def _open_3d_view(self) -> None:
        Calibration3DWindow(self, self._export_payload, self._save_default_json)

    def _import_json(self) -> None:
        path = filedialog.askopenfilename(title="Importar calibracion", initialdir=ROOT, filetypes=[("JSON", "*.json")])
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._apply_payload(payload)
        self._switch_camera(self.active_camera.get())

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        for camera, data in payload.get("cameras", {}).items():
            if camera not in self.cameras:
                continue
            self.cameras[camera].frame_time = float(data.get("frame_time", 0.0))
            self.cameras[camera].points = [MarkPoint(**point) for point in data.get("points", [])]
            if "camera_height_m" in data and camera in self.camera_heights:
                self.camera_heights[camera].set(float(data["camera_height_m"]))
            if "lens" in data and camera in self.lens_profiles:
                try:
                    self.lens_profiles[camera] = LensProfile(**data["lens"])
                except TypeError:
                    pass

    def _on_close(self) -> None:
        for source in self.sources.values():
            source.close()
        self.destroy()


class Calibration3DWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, payload_provider: Any, save_callback: Any | None = None) -> None:
        super().__init__(parent)
        self.title("Vista 3D - 3D Basket Match")
        self.geometry("1100x760")
        self.minsize(900, 620)
        self._app = parent
        self.payload_provider = payload_provider
        self.save_callback = save_callback
        self.payload = payload_provider()
        self.yaw = math.radians(0)
        self.pitch = math.radians(55)
        self.zoom = 26.0
        self.view_distance = 42.0
        self.pan = [0.0, 0.0]
        self.show_estimated_pose = tk.BooleanVar(value=False)
        self.show_camera_mapping = tk.BooleanVar(value=False)
        self.mapping_camera = tk.StringVar(value="cam2")
        self.mapping_cache: dict[str, Any] = {}
        self._floor_tk_image: Any = None
        self.drag_start: tuple[int, int, float, float] | None = None
        self.pan_start: tuple[int, int, float, float] | None = None

        self._build_ui()
        self._draw()

    @property
    def court_spec(self) -> CourtSpec:
        return getattr(self._app, "court_spec", COURT_PRESETS["FIBA"])

    def _court_line_defs(self) -> list[tuple]:
        return _court_lines_for(self.court_spec)

    def _court_3d_lines(self) -> list[tuple]:
        return _build_court_3d_lines_for(self.court_spec)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        bar = ttk.Frame(self, padding=(10, 8))
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="Cargar JSON", command=self._load_json).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Recalcular", command=self._recalculate_from_calibrator).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Reset vista", command=self._reset_view).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Arriba", command=self._top_view).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Lateral", command=self._side_view).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(bar, text="Mostrar pose estimada", variable=self.show_estimated_pose, command=self._draw).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(bar, text="Camera mapping", variable=self.show_camera_mapping, command=self._draw).pack(side="left", padx=(0, 8))
        self.mapping_combo = ttk.Combobox(bar, state="readonly", width=7, textvariable=self.mapping_camera, values=["cam2", "cam1", "mix"])
        self.mapping_combo.pack(side="left", padx=(0, 8))
        self.mapping_combo.bind("<<ComboboxSelected>>", lambda _event: self._draw())
        ttk.Label(
            bar,
            text="Arrastre izquierdo: rotar | rueda: zoom | boton central: paneo",
        ).pack(side="left")

        self.status = ttk.Label(self, padding=(10, 0), text="")
        self.status.grid(row=2, column=0, sticky="ew")

        self.canvas = tk.Canvas(self, bg="#0b0f16", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._draw())
        self.canvas.bind("<Button-1>", self._start_rotate)
        self.canvas.bind("<B1-Motion>", self._rotate_view)
        self.canvas.bind("<ButtonRelease-1>", lambda _event: self._end_rotate())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda _event: self._zoom(1.12))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom(1 / 1.12))
        self.canvas.bind("<Button-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan_view)
        self.canvas.bind("<ButtonRelease-2>", lambda _event: self._end_pan())

    def _load_json(self) -> None:
        path = filedialog.askopenfilename(title="Cargar calibracion", initialdir=ROOT, filetypes=[("JSON", "*.json")])
        if not path:
            return
        self.payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.mapping_cache.clear()
        self._draw()

    def _recalculate_from_calibrator(self) -> None:
        self.payload = self.payload_provider()
        self.mapping_cache.clear()
        self._draw()

    def _reset_view(self) -> None:
        self.yaw = math.radians(0)
        self.pitch = math.radians(40)
        self.zoom = 26.0
        self.view_distance = 42.0
        self.pan = [0.0, 0.0]
        self._draw()

    def _top_view(self) -> None:
        self.yaw = math.radians(0)
        self.pitch = math.radians(90)
        self.zoom = 30.0
        self.view_distance = 50.0
        self.pan = [0.0, 0.0]
        self._draw()

    def _side_view(self) -> None:
        self.yaw = math.radians(0)
        self.pitch = math.radians(0)
        self.zoom = 28.0
        self.view_distance = 42.0
        self.pan = [0.0, 0.0]
        self._draw()

    def _start_rotate(self, event: tk.Event) -> None:
        self.drag_start = (event.x, event.y, self.yaw, self.pitch)

    def _rotate_view(self, event: tk.Event) -> None:
        if not self.drag_start:
            return
        sx, sy, yaw, pitch = self.drag_start
        self.yaw = yaw + (event.x - sx) * 0.008
        self.pitch = pitch + (event.y - sy) * 0.008
        self._draw()

    def _end_rotate(self) -> None:
        self.drag_start = None

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._zoom(1.12 if event.delta > 0 else 1 / 1.12)

    def _zoom(self, factor: float) -> None:
        self.zoom = max(8.0, min(90.0, self.zoom * factor))
        self.view_distance = max(16.0, min(120.0, self.view_distance / factor))
        self._draw()

    def _start_pan(self, event: tk.Event) -> None:
        self.pan_start = (event.x, event.y, self.pan[0], self.pan[1])
        self.canvas.configure(cursor="fleur")

    def _pan_view(self, event: tk.Event) -> None:
        if not self.pan_start:
            return
        sx, sy, px, py = self.pan_start
        self.pan[0] = px + event.x - sx
        self.pan[1] = py + event.y - sy
        self._draw()

    def _end_pan(self) -> None:
        self.pan_start = None
        self.canvas.configure(cursor="")

    def _project(self, point: tuple[float, float, float]) -> tuple[float, float]:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        x = point[0] - self.court_spec.length_m / 2
        y = point[1] - self.court_spec.width_m / 2
        z = point[2]

        cosy, siny = math.cos(self.yaw), math.sin(self.yaw)
        cosp, sinp = math.cos(self.pitch), math.sin(self.pitch)

        right = x * cosy - y * siny
        forward = x * siny + y * cosy
        up = z
        screen_y_axis = forward * sinp - up * cosp
        depth_axis = forward * cosp + up * sinp
        camera_z = self.view_distance - depth_axis

        perspective = self.view_distance / max(4.0, camera_z)
        sx = width / 2 + self.pan[0] + right * self.zoom * perspective
        sy = height / 2 + self.pan[1] + screen_y_axis * self.zoom * perspective
        return sx, sy

    def _draw_line(self, a: tuple[float, float, float], b: tuple[float, float, float], color: str, width: int = 1, dash: tuple[int, int] | None = None) -> None:
        ax, ay = self._project(a)
        bx, by = self._project(b)
        self.canvas.create_line(ax, ay, bx, by, fill=color, width=width, dash=dash)

    def _draw_poly(self, points: list[tuple[float, float, float]], outline: str, fill: str = "", width: int = 1) -> None:
        coords: list[float] = []
        for point in points:
            coords.extend(self._project(point))
        self.canvas.create_polygon(*coords, outline=outline, fill=fill, width=width)

    def _draw_text(self, point: tuple[float, float, float], text: str, color: str = "#d7dee9") -> None:
        x, y = self._project(point)
        self.canvas.create_text(x, y, text=text, fill=color, font=("Segoe UI", 9))

    def _draw(self) -> None:
        self.canvas.delete("all")
        self._draw_floor()
        self._draw_walls()
        self._draw_backboards()
        self._draw_cameras()
        self._draw_mark_summary()

    def _draw_floor(self) -> None:
        L, W = self.court_spec.length_m, self.court_spec.width_m
        if self.show_camera_mapping.get():
            self._draw_camera_mapped_floor()
        else:
            self._draw_poly(
                [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0)],
                outline="#d7dee9",
                fill="#17212d",
                width=2,
            )
        for x in np.arange(0.0, L + 0.001, 1.0):
            color = "#5b6b7d" if abs(x % 7.0) > 1e-6 else "#d7dee9"
            self._draw_line((float(x), 0, 0.02), (float(x), W, 0.02), color, 1)
        for y in np.arange(0.0, W + 0.001, 1.0):
            color = "#5b6b7d" if y not in (0.0, W) else "#d7dee9"
            self._draw_line((0, float(y), 0.02), (L, float(y), 0.02), color, 1)

        for code, label, axis, value, usage in self._court_line_defs():
            if usage != "calibration":
                continue
            color = COURT_LINE_COLORS[code]
            if axis == "x":
                self._draw_line((value, 0, 0.06), (value, W, 0.06), color, 4)
                self._draw_text((value, W + 0.8, 0.2), label, color)
            else:
                self._draw_line((0, value, 0.06), (L, value, 0.06), color, 4)
                self._draw_text((L + 1.2, value, 0.2), label, color)

        self._draw_text((L / 2, W / 2, 0.3), f"CANCHA {L:.1f}m x {W:.1f}m ({self.court_spec.name})", "#ffffff")

    def _camera_world_pos(self, camera_name: str) -> tuple[float, float] | None:
        proj = self.payload.get("cameras", {}).get(camera_name, {}).get("projection", {})
        for pose_key in ("camera_pose_pnp", "camera_pose"):
            pose = proj.get(pose_key) or {}
            center = pose.get("camera_center_world_m")
            if center:
                return float(center.get("x", 0)), float(center.get("y", 0))
        return None

    def _undistorted_frame(self, camera_name: str) -> np.ndarray | None:
        """Return a downscaled + undistorted camera frame (cached)."""
        key = f"frame_ds_{camera_name}"
        if key in self.mapping_cache:
            return self.mapping_cache[key]
        raw = self._mapping_texture(camera_name)
        if raw is None:
            self.mapping_cache[key] = None
            return None
        frame: np.ndarray = raw["image"].copy()
        lens_data = self.payload.get("cameras", {}).get(camera_name, {}).get("lens")
        if lens_data is not None and cv2 is not None:
            try:
                prof = LensProfile(**lens_data)
                if prof.has_distortion():
                    h_f, w_f = frame.shape[:2]
                    f_px = prof.focal_px_for(w_f)
                    if f_px is None:
                        fov = CAMERA_FOV_HINTS.get(camera_name, DEFAULT_CAMERA_FOV_DEG)
                        f_px = (w_f / 2) / math.tan(math.radians(fov) / 2)
                    K_mat = np.float64([
                        [f_px, 0, w_f / 2 + prof.cx_offset_px],
                        [0, f_px, h_f / 2 + prof.cy_offset_px],
                        [0, 0, 1],
                    ])
                    frame = cv2.undistort(frame, K_mat, prof.dist_array())
            except Exception:
                pass
        # Downscale for speed: target ≤ 1200 px wide
        h_f, w_f = frame.shape[:2]
        ds = max(1, w_f // 1200)
        if ds > 1:
            frame = cv2.resize(frame, (w_f // ds, h_f // ds), interpolation=cv2.INTER_AREA)
        self.mapping_cache[key] = (frame, ds)
        return self.mapping_cache[key]

    def _project_camera_to_canvas(
        self,
        camera_name: str,
        H_screen_to_world: np.ndarray,
        canvas_w: int,
        canvas_h: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Warp camera image to canvas space by composing screen→world→image.
        Returns (warped_bgr, valid_mask_uint8) or None."""
        raw = self._mapping_texture(camera_name)
        if raw is None:
            return None
        # Use the raw (distorted) frame — the homography was calibrated on it.
        frame_full: np.ndarray = raw["image"]
        h_full, w_full = frame_full.shape[:2]
        ds = max(1, w_full // 1200)
        if ds > 1:
            frame_key = f"frame_raw_ds_{camera_name}"
            if frame_key not in self.mapping_cache:
                self.mapping_cache[frame_key] = cv2.resize(
                    frame_full, (w_full // ds, h_full // ds), interpolation=cv2.INTER_AREA
                )
            frame = self.mapping_cache[frame_key]
        else:
            frame = frame_full
        H_world_to_image = np.asarray(raw["homography"], dtype=np.float64)

        # H_s2i: canvas pixel → image pixel (backward map).
        # H_world_to_image maps world→image; H_screen_to_world maps canvas→world.
        H_s2i = H_world_to_image @ H_screen_to_world
        # Adjust for downscale: image coords = H_s2i @ screen_px, then divide by ds
        S_ds = np.float64([[1.0 / ds, 0, 0], [0, 1.0 / ds, 0], [0, 0, 1]])
        H_s2i_ds = S_ds @ H_s2i

        # WARP_INVERSE_MAP: H_s2i_ds is already the backward map (canvas→frame),
        # so skip the internal inversion that warpPerspective would otherwise do.
        warped = cv2.warpPerspective(
            frame, H_s2i_ds, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
        h_f, w_f = frame.shape[:2]
        ones = np.ones((h_f, w_f), dtype=np.uint8) * 255
        valid = cv2.warpPerspective(
            ones, H_s2i_ds, (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return warped, valid

    def _draw_camera_mapped_floor(self) -> None:
        if cv2 is None:
            return
        mode = self.mapping_camera.get()
        L, W = self.court_spec.length_m, self.court_spec.width_m
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)

        # Homography: world floor (X,Y) ↔ screen pixel (via 4 floor corners)
        world_corners = [(0.0, 0.0), (L, 0.0), (L, W), (0.0, W)]
        screen_corners = np.float32([self._project((x, y, 0.0)) for x, y in world_corners])
        world_pts = np.float32(world_corners)
        try:
            H_w2s = cv2.getPerspectiveTransform(world_pts, screen_corners)
            H_s2w = np.linalg.inv(H_w2s.astype(np.float64))
        except (cv2.error, np.linalg.LinAlgError):
            return

        # Clip mask: only draw inside the floor quad
        floor_clip = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(floor_clip, [screen_corners.astype(np.int32)], 255)

        cam_list = ["cam2", "cam1"] if mode == "mix" else [mode]
        layers: list[tuple[str, np.ndarray, np.ndarray]] = []
        for cam in cam_list:
            result = self._project_camera_to_canvas(cam, H_s2w, canvas_w, canvas_h)
            if result is not None:
                layers.append((cam, result[0], result[1]))

        if not layers:
            self._draw_poly([(0,0,0),(L,0,0),(L,W,0),(0,W,0)], outline="#d7dee9", fill="#17212d", width=2)
            return

        if mode == "mix" and len(layers) == 2:
            # Blend by world X: cam1 covers its aimed half, cam2 covers its other half.
            # Compute world X for every canvas pixel.
            px_x, px_y = np.meshgrid(np.arange(canvas_w, dtype=np.float64),
                                      np.arange(canvas_h, dtype=np.float64))
            ones = np.ones((canvas_h, canvas_w), dtype=np.float64)
            scr_pts = np.stack([px_x, px_y, ones]).reshape(3, -1)  # (3, N)
            w_pts = H_s2w @ scr_pts
            world_x = (w_pts[0] / (w_pts[2] + 1e-12)).reshape(canvas_h, canvas_w)

            # Decide split: camera aimed at the smaller X target covers the left half
            tgt = CAMERA_TARGETS
            cam_a, warped_a, mask_a = layers[0]  # cam2
            cam_b, warped_b, mask_b = layers[1]  # cam1
            tgt_x_a = tgt.get(cam_a, (L * 0.75, W / 2, 0))[0]
            tgt_x_b = tgt.get(cam_b, (L * 0.25, W / 2, 0))[0]
            # cam with higher target X covers right half (world_x > L/2)
            if tgt_x_a >= tgt_x_b:
                # a=cam2=right, b=cam1=left
                blend_a = np.clip((world_x - L / 2) / (L * 0.12) + 0.5, 0.0, 1.0)
            else:
                blend_a = np.clip((L / 2 - world_x) / (L * 0.12) + 0.5, 0.0, 1.0)

            va = (mask_a > 127).astype(np.float64)
            vb = (mask_b > 127).astype(np.float64)
            wa = blend_a * va
            wb = (1.0 - blend_a) * vb
            total = wa + wb + 1e-12
            blended = (warped_a.astype(np.float64) * wa[:, :, None] +
                       warped_b.astype(np.float64) * wb[:, :, None]) / total[:, :, None]
            result_bgr = np.clip(blended, 0, 255).astype(np.uint8)
            valid = ((wa + wb) > 0.05) & (floor_clip > 127)
        else:
            _, warped, mask = layers[0]
            result_bgr = warped
            valid = (mask > 127) & (floor_clip > 127)

        BG = np.full((canvas_h, canvas_w, 3), (11, 15, 22), dtype=np.uint8)
        alpha = valid[:, :, None].astype(np.float32)
        out_bgr = (BG * (1.0 - alpha) + result_bgr * alpha).astype(np.uint8)
        rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        self._floor_tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor="nw", image=self._floor_tk_image)

    def _sample_mapping_color(self, textures: list[dict[str, Any]], world_x: float, world_y: float) -> str:
        for texture in textures:
            if not self._world_point_allowed_for_texture(texture, world_x, world_y):
                continue
            image = texture["image"]
            homography = np.asarray(texture["homography"], dtype=float)
            height, width = image.shape[:2]
            sample = homography @ np.asarray([world_x, world_y, 1.0], dtype=float)
            if abs(sample[2]) < 1e-9:
                continue
            u = int(round(sample[0] / sample[2]))
            v = int(round(sample[1] / sample[2]))
            margin = 12
            if margin <= u < width - margin and margin <= v < height - margin:
                b, g, r = image[v, u]
                return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        return "#202833"

    def _world_point_allowed_for_texture(self, texture: dict[str, Any], world_x: float, world_y: float) -> bool:
        bounds = texture.get("world_bounds")
        if bounds is None:
            return True
        return bounds["min_x"] <= world_x <= bounds["max_x"] and bounds["min_y"] <= world_y <= bounds["max_y"]

    def _mapping_texture(self, camera_name: str) -> dict[str, Any] | None:
        if camera_name in self.mapping_cache:
            return self.mapping_cache[camera_name]
        camera = self.payload.get("cameras", {}).get(camera_name)
        if not camera:
            return None
        homography = camera.get("projection", {}).get("court_homography_world_to_image")
        if homography is None:
            return None
        video_path = ROOT / camera.get("video", f"videos/{camera_name}.mov")
        if not video_path.exists():
            return None
        try:
            source = VideoSource(video_path)
            frame = source.read_at(float(camera.get("frame_time", 0.0)))
            source.close()
        except RuntimeError:
            return None
        image = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        texture = {"image": image, "homography": homography, "world_bounds": self._mapping_world_bounds(camera)}
        self.mapping_cache[camera_name] = texture
        return texture

    def _mapping_world_bounds(self, camera: dict[str, Any]) -> dict[str, float] | None:
        intersections = camera.get("projection", {}).get("court_intersections") or []
        L, W = self.court_spec.length_m, self.court_spec.width_m
        if len(intersections) >= 4:
            return {"min_x": 0.0, "max_x": L, "min_y": 0.0, "max_y": W}
        if not intersections:
            return None
        xs = [float(item["world"]["x"]) for item in intersections]
        ys = [float(item["world"]["y"]) for item in intersections]
        line_points = [point for point in camera.get("points", []) if point.get("type") == "court" and point.get("world", {}).get("usage") == "calibration"]
        x_values = {float(point.get("world", {}).get("value")) for point in line_points if point.get("world", {}).get("axis") == "x"}
        y_values = {float(point.get("world", {}).get("value")) for point in line_points if point.get("world", {}).get("axis") == "y"}
        pad_x = 5.0 if len(x_values) <= 1 else L
        pad_y = 3.0 if len(y_values) <= 1 else W
        return {
            "min_x": max(0.0, min(xs) - pad_x),
            "max_x": min(L, max(xs) + pad_x),
            "min_y": max(0.0, min(ys) - pad_y),
            "max_y": min(W, max(ys) + pad_y),
        }

    def _draw_walls(self) -> None:
        L, W = self.court_spec.length_m, self.court_spec.width_m
        wall_planes = self._wall_planes_present()
        if "left_wall" in wall_planes:
            self._draw_wall_frame([(0, W, 0), (L, W, 0), (L, W, 5), (0, W, 5)], "Pared fondo")
        if "right_wall" in wall_planes:
            self._draw_wall_frame([(0, 0, 0), (L, 0, 0), (L, 0, 5), (0, 0, 5)], "Pared camaras")
        if "back_wall" in wall_planes:
            self._draw_wall_frame([(L, 0, 0), (L, W, 0), (L, W, 5), (L, 0, 5)], "Pared fondo")
        if "front_wall" in wall_planes:
            self._draw_wall_frame([(0, 0, 0), (0, W, 0), (0, W, 5), (0, 0, 5)], "Pared fondo")

    def _draw_wall_frame(self, points: list[tuple[float, float, float]], label: str) -> None:
        loop = points + [points[0]]
        for start, end in zip(loop, loop[1:]):
            self._draw_line(start, end, "#f7bd52", 2)
        self._draw_line(points[0], points[2], "#f7bd52", 1, (4, 5))
        self._draw_line(points[1], points[3], "#f7bd52", 1, (4, 5))
        center = tuple(sum(point[index] for point in points) / len(points) for index in range(3))
        self._draw_text((center[0], center[1], center[2] + 0.4), label, "#f7bd52")

    def _wall_planes_present(self) -> set[str]:
        planes: set[str] = set()
        for camera in self.payload.get("cameras", {}).values():
            for point in camera.get("points", []):
                if point.get("type") == "wall" and point.get("plane"):
                    planes.add(point["plane"])
        return planes

    def _draw_backboards(self) -> None:
        if not any(self._camera_points_of_type("backboard")):
            return
        bo = self.court_spec.board_offset_m
        for x, label in ((bo, "Tablero A"), (self.court_spec.length_m - bo, "Tablero B")):
            y = self.court_spec.width_m / 2
            z = 3.05
            w = 1.8
            h = 1.05
            self._draw_poly([(x, y - w / 2, z - h / 2), (x, y + w / 2, z - h / 2), (x, y + w / 2, z + h / 2), (x, y - w / 2, z + h / 2)], "#57a7ff", "#132941", 2)
            self._draw_line((x, y, z - h / 2), (x, y, 0), "#57a7ff", 1, (4, 4))
            self._draw_text((x, y, z + 0.9), label, "#57a7ff")

    def _camera_points_of_type(self, point_type: str) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for camera in self.payload.get("cameras", {}).values():
            points.extend([point for point in camera.get("points", []) if point.get("type") == point_type])
        return points

    def _draw_cameras(self) -> None:
        self._draw_stereo_rig_base()
        names = sorted(self.payload.get("cameras", {}).keys())
        bo, L, W = self.court_spec.board_offset_m, self.court_spec.length_m, self.court_spec.width_m
        fallback_targets = {
            "cam1": (bo, W / 2 - bo, 3.05),
            "cam2": (L - bo, W / 2 + bo, 3.05),
        }
        for index, name in enumerate(names):
            offset = (index - (len(names) - 1) / 2) * CAMERA_RIG_BASELINE_M
            rig_position = (CAMERA_RIG_POSITION[0] + offset, CAMERA_RIG_POSITION[1], CAMERA_RIG_POSITION[2])
            target = CAMERA_TARGETS.get(name, fallback_targets.get(name, (self.court_spec.board_offset_m, self.court_spec.width_m / 2, 3.05)))
            color = "#ff6b6b" if name == "cam1" else "#8e7dff"
            self._draw_camera_marker(rig_position, target, name.upper(), color)
            if self.show_estimated_pose.get():
                estimated = self._camera_position_from_payload(name, fallback=None)
                if estimated is not None:
                    self._draw_camera_marker(estimated, target, f"{name.upper()} EST", "#bbbbbb", diagnostic=True)

    def _camera_position_from_payload(self, name: str, fallback: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
        camera = self.payload.get("cameras", {}).get(name, {})
        center = camera.get("projection", {}).get("camera_pose", {}).get("camera_center_world_m")
        if not center:
            return fallback
        default = fallback or (0.0, 0.0, 0.0)
        x = float(center.get("x", default[0]))
        y = float(center.get("y", default[1]))
        z = float(center.get("z", default[2]))
        if not all(math.isfinite(value) for value in (x, y, z)):
            return fallback
        # Reject estimates far outside the physical rig range
        rx, ry, rz = CAMERA_RIG_POSITION
        if abs(x - rx) > 20 or abs(y - ry) > 12 or z < -1 or z > 20:
            return fallback
        return x, y, z

    def _draw_stereo_rig_base(self) -> None:
        x, y, z = CAMERA_RIG_POSITION
        half = CAMERA_RIG_BASELINE_M
        self._draw_line((x - half, y, z), (x + half, y, z), "#e6edf7", 3)
        self._draw_line((x, y, z), (x, y, 0), "#e6edf7", 1, (4, 4))
        self._draw_text((x, y, z + 0.7), "RIG ESTEREO", "#e6edf7")

    def _draw_camera_marker(self, position: tuple[float, float, float], target: tuple[float, float, float], label: str, color: str, diagnostic: bool = False) -> None:
        x, y, z = position
        size = 0.28 if diagnostic else 0.4
        fill = "" if diagnostic else color
        dash = (2, 5) if diagnostic else (5, 5)
        self._draw_poly([(x - size, y - size, z), (x + size, y - size, z), (x + size, y + size, z), (x - size, y + size, z)], color, fill, 1)
        self._draw_line(position, target, color, 1, dash)
        self._draw_text((x, y, z + 0.8), label, color)

    def _draw_mark_summary(self) -> None:
        cameras = self.payload.get("cameras", {})
        lines = []
        for name, camera in cameras.items():
            projection = camera.get("projection", {})
            points = camera.get("points", [])
            line_count = len({point.get("line") for point in points if point.get("type") == "court" and point.get("line")})
            intersections = len(projection.get("court_intersections") or [])
            vp = projection.get("vanishing_points") or {}
            vp_count = sum(1 for value in vp.values() if value)
            has_h = projection.get("court_homography_world_to_image") is not None
            pose = projection.get("camera_pose") or {}
            center = pose.get("camera_center_world_m")
            pose_text = "pose NO"
            if center:
                quality = pose.get("quality", "?")
                fov = float(pose.get("horizontal_fov_deg", 0.0))
                pose_text = f"pose z={float(center.get('z', 0.0)):.2f}m fov={fov:.1f} {quality}"
            coverage = self._mapping_coverage(name)
            lines.append(f"{name}: {line_count} lineas, {intersections} intersecciones, {vp_count} fugas, cobertura {coverage:.0%}, {pose_text}")
        self.status.configure(
            text=" | ".join(lines) + " | Mapping mix: prioridad cam2 y relleno cam1. Vista principal usa rig fisico."
        )

    def _mapping_coverage(self, camera_name: str) -> float:
        texture = self._mapping_texture(camera_name)
        if texture is None:
            return 0.0
        valid = 0
        total = 0
        image = texture["image"]
        homography = np.asarray(texture["homography"], dtype=float)
        height, width = image.shape[:2]
        for x in np.arange(0.5, self.court_spec.length_m, 1.0):
            for y in np.arange(0.5, self.court_spec.width_m, 1.0):
                total += 1
                if not self._world_point_allowed_for_texture(texture, float(x), float(y)):
                    continue
                sample = homography @ np.asarray([x, y, 1.0], dtype=float)
                if abs(sample[2]) < 1e-9:
                    continue
                u = int(round(sample[0] / sample[2]))
                v = int(round(sample[1] / sample[2]))
                if 12 <= u < width - 12 and 12 <= v < height - 12:
                    valid += 1
        return valid / total if total else 0.0


def main() -> int:
    app = CalibratorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
