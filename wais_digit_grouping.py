"""
WAIS Digit Symbol Coding — Grid Digit Grouping & Review Tool

Usage:
    Double-click the script, or run:
        python wais_digit_grouping.py [image_path]

    If no path is given, a menu pops up:
      1 → Camera (live preview, press any key to capture)
      2 → Select file via dialog

Workflow:
  1. Menu → choose input source
  2. Image appears → click 4 corners of the entry grid (TL→TR→BR→BL)
     Right-click to undo.  Then 9 draggable control points appear;
     drag to warp, press Enter/Space to confirm (homography optimised).
  3. Perspective‑correct the grid region
  4. Divide corrected grid into 7×20 cell_groups (first 7 = samples skipped)
     using the known digit layout.  Each cell is enlarged by 15 %.
     Extract the symbol region below the digit.
  5. Interactive review popup:
     • Left‑click a cell → select (green highlight)
     • Right‑click a cell → deselect
     • Drag to select multiple cells at once
     • Selected‑cell counter at bottom
  6. Enter/Space → continue with next image (same input mode)
     Q / Esc / close → quit entirely

Dependencies: opencv-python, numpy, matplotlib
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# ── Cross‑platform matplotlib backend ──────────────────────────
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, CloseEvent, MouseEvent
from matplotlib.patches import Rectangle


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
GRID_ROWS = 7
GRID_COLS = 20

# Symbol extraction boundaries (fractions of cell_group height)
SYMBOL_TOP_FRAC = 0.38
SYMBOL_BOT_FRAC = 0.92

# Cell expansion factor (fraction of cell size added to each side)
CELL_ENLARGE = 0.15

# Number of sample/demo cells at the start of row 1 (skipped during extraction)
SAMPLE_COUNT = 7

# ── Known WAIS digit layout (7 rows × 20 columns) ────────────
DIGIT_GRID: list[list[int]] = [
    [2, 1, 3, 7, 2, 4, 8, 2, 1, 3, 2, 1, 4, 2, 3, 5, 2, 3, 1, 4],
    [5, 6, 3, 1, 4, 1, 5, 4, 2, 7, 6, 3, 5, 7, 2, 8, 5, 4, 6, 3],
    [7, 2, 8, 1, 9, 5, 8, 4, 7, 3, 6, 2, 5, 1, 9, 2, 8, 3, 7, 4],
    [6, 5, 9, 4, 8, 3, 7, 2, 6, 1, 5, 4, 6, 3, 7, 9, 2, 8, 1, 7],
    [9, 4, 6, 8, 5, 9, 7, 1, 8, 5, 2, 9, 4, 8, 6, 3, 7, 9, 8, 6],
    [2, 7, 3, 6, 5, 1, 9, 8, 4, 5, 7, 3, 1, 4, 8, 7, 9, 1, 4, 5],
    [7, 1, 8, 2, 9, 3, 6, 7, 2, 8, 5, 2, 3, 1, 4, 8, 4, 2, 7, 6],
]


# ══════════════════════════════════════════════════════════════
# PHASE 1 — Load image & let user place 4 corners,
#            then fine-tune with 9 draggable control points
# ══════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Cannot load image from '{path}'")
        sys.exit(1)
    return img


def _optimize_corners_from_3x3(pts: list[tuple[float, float]]
                               ) -> list[tuple[int, int]]:
    """
    Given 9 control points arranged as a 3×3 grid in reading order:

        TL  top‑mid  TR
        left‑mid  centre  right‑mid
        BL  bot‑mid  BR

    fit a homography from a regular unit 3×3 grid to these points,
    then extract the 4 corners.  This "rectangularises" the user's
    warped grid into the best perspective rectangle.
    """
    src = np.array([
        [0, 0], [0.5, 0], [1, 0],
        [0, 0.5], [0.5, 0.5], [1, 0.5],
        [0, 1], [0.5, 1], [1, 1],
    ], dtype=np.float32)
    dst = np.array(pts, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    if H is None:
        # fallback — just use the four corners as-is
        return [(int(round(pts[i][0])), int(round(pts[i][1])))
                for i in (0, 2, 8, 6)]

    # Project the 4 corners of the unit square through H
    unit_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]],
                            dtype=np.float32)
    warped = cv2.perspectiveTransform(unit_corners.reshape(-1, 1, 2), H)
    warped = warped.reshape(-1, 2)

    return [(int(round(p[0])), int(round(p[1]))) for p in warped]


def select_four_corners(img: np.ndarray, filename: str
                        ) -> list[tuple[int, int]]:
    """
    Two‑stage corner selection:

    **Stage 1** — click 4 corners (TL → TR → BR → BL),
                   right‑click to undo.

    **Stage 2** — the system auto‑generates 5 extra control points
                   (4 edge midpoints + centre) forming a 3×3 grid.
                   All 9 dots are **draggable**; the grid wireframe
                   updates live.  Press **Enter** to apply a smart
                   homography fit that returns the optimal 4 corners
                   of a perspective rectangle.

    Returns the four optimised corner coordinates.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(img_rgb)
    ax.axis("on")

    # ── Stage-1 state ────────────────────────────────────
    corners: list[tuple[int, int]] = []
    colors = ["red", "lime", "cyan", "magenta"]
    labels = ["TL", "TR", "BR", "BL"]
    artists: list = []

    # ── Stage-2 state ────────────────────────────────────
    # 9 control points in reading order: TL..BR (3×3)
    ctrl: list[tuple[float, float]] | None = None
    ctrl_artists: list = []        # plotted dots
    grid_lines: list = []          # plotted wireframe segments
    drag_idx: int | None = None    # which control point is being dragged
    stage2_active = False

    # ── Common helpers ───────────────────────────────────

    def _clear_all():
        nonlocal stage2_active, ctrl, drag_idx
        stage2_active = False
        ctrl = None
        drag_idx = None
        for a in artists + ctrl_artists + grid_lines:
            a.remove()
        artists.clear()
        ctrl_artists.clear()
        grid_lines.clear()
        ax.set_title(
            "Click the 4 corners of the ENTRY GRID:\n"
            "Left-click: TL → TR → BR → BL     "
            "Right-click: undo last corner\n"
            f"File: {filename}",
            fontsize=11, fontweight="bold",
        )

    def _redraw_stage1():
        _clear_all()
        for i, (x, y) in enumerate(corners):
            (pt,) = ax.plot(x, y, "o", color=colors[i], markersize=10)
            artists.append(pt)
            ann = ax.annotate(
                f"{labels[i]}  ({x}, {y})",
                (x, y), fontsize=10, color=colors[i],
                fontweight="bold",
                xytext=(8, 8), textcoords="offset pixels",
            )
            artists.append(ann)
        if len(corners) == 4:
            poly = plt.Polygon(corners, fill=False,
                               edgecolor="yellow", linewidth=2, linestyle="--")
            ax.add_patch(poly)
            artists.append(poly)
        fig.canvas.draw()

    def _make_ctrl() -> list[tuple[float, float]]:
        """Generate the 9 control points from the 4 corners."""
        def _mid(pa, pb):
            return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)

        top_mid = _mid(corners[0], corners[1])
        bot_mid = _mid(corners[3], corners[2])
        left_mid = _mid(corners[0], corners[3])
        right_mid = _mid(corners[1], corners[2])
        centre = _mid(top_mid, bot_mid)

        return [
            corners[0], top_mid, corners[1],
            left_mid, centre, right_mid,
            corners[3], bot_mid, corners[2],
        ]

    def _draw_grid():
        """Draw the 3×3 wireframe + 9 control dots on top of the image."""
        nonlocal stage2_active
        # Remove previous grid/dots
        for a in ctrl_artists + grid_lines:
            a.remove()
        ctrl_artists.clear()
        grid_lines.clear()

        if ctrl is None:
            return

        stage2_active = True
        xs = [p[0] for p in ctrl]
        ys = [p[1] for p in ctrl]

        # ── Wireframe (3×3 grid) ─────────────────────────
        # Horizontal lines
        for row in range(3):
            i0 = row * 3
            line, = ax.plot([xs[i0], xs[i0 + 2]], [ys[i0], ys[i0 + 2]],
                           color="yellow", linewidth=1.5, linestyle="-")
            grid_lines.append(line)
        # Vertical lines
        for col in range(3):
            line, = ax.plot([xs[col], xs[col + 6]], [ys[col], ys[col + 6]],
                           color="yellow", linewidth=1.5, linestyle="-")
            grid_lines.append(line)

        # ── Control dots ─────────────────────────────────
        for i in range(9):
            colours = ["red", "lime", "cyan", "magenta", "orange",
                       "cyan", "magenta", "lime", "red"]
            (dot,) = ax.plot(xs[i], ys[i], "o",
                             color=colours[i], markersize=6, zorder=5)
            ctrl_artists.append(dot)

        ax.set_title(
            "Drag any dot to adjust the grid  |  "
            "Press Enter to confirm",
            fontsize=11, fontweight="bold",
        )
        fig.canvas.draw()

    def _ctrl_at(x: float, y: float, radius: float = 12
                 ) -> int | None:
        """Return the index of the control point nearest (x,y) within radius."""
        if ctrl is None:
            return None
        best, best_dist = None, radius
        for i, (cx, cy) in enumerate(ctrl):
            d = np.hypot(x - cx, y - cy)
            if d < best_dist:
                best_dist = d
                best = i
        return best

    # ── Event handlers ───────────────────────────────────

    def on_press(event):
        nonlocal drag_idx, stage2_active
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Right-click → undo in stage 1
        if event.button == 3:
            if not stage2_active and corners:
                corners.pop()
                _redraw_stage1()
            return

        if event.button != 1:
            return

        if stage2_active and ctrl is not None:
            # Try to pick up a control point
            idx = _ctrl_at(event.xdata, event.ydata)
            if idx is not None:
                drag_idx = idx
            return

        # Stage 1: left-click → add corner
        if len(corners) < 4:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            corners.append((x, y))
            _redraw_stage1()
            # If we just got the 4th corner, auto-enter stage 2
            if len(corners) == 4:
                ctrl = _make_ctrl()
                _draw_grid()

    def on_motion(event):
        nonlocal drag_idx
        if drag_idx is None or ctrl is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes != ax:
            return
        # Clamp to image bounds
        x = max(0, min(event.xdata, w_img - 1))
        y = max(0, min(event.ydata, h_img - 1))
        ctrl[drag_idx] = (x, y)
        _draw_grid()

    def on_release(event):
        nonlocal drag_idx
        if event.button == 1:
            drag_idx = None

    def on_key(event):
        if event.key in ("enter", "space"):
            if ctrl is not None:
                plt.close(fig)
        elif event.key in ("escape", "q", "Q"):
            sys.exit(0)

    def on_close(_event):
        sys.exit(0)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    plt.show()

    if ctrl is not None:
        print("       Optimising corners from 3×3 control grid …")
        return _optimize_corners_from_3x3(ctrl)
    elif len(corners) == 4:
        # User pressed Escape before dragging — use raw corners
        return corners
    else:
        print("ERROR: You must click exactly 4 corners.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# PHASE 2 — Perspective correction
# ══════════════════════════════════════════════════════════════

def perspective_correct(
    img: np.ndarray, src_corners: list[tuple[int, int]]
) -> np.ndarray:
    """Warp the quadrilateral into a rectangle.  Returns warped image."""
    src = np.array(src_corners, dtype=np.float32)

    def _dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    w = int(round(max(_dist(src[0], src[1]), _dist(src[3], src[2]))))
    h = int(round(max(_dist(src[0], src[3]), _dist(src[1], src[2]))))

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════════════
# PHASE 3 — Extract symbols grouped by digit
# ══════════════════════════════════════════════════════════════

def extract_symbols(warped: np.ndarray) -> dict[int, list[np.ndarray]]:
    """
    Divide the perspective-corrected grid into 7×20 equal cell_groups.
    Each cell is enlarged by ``CELL_ENLARGE × 100 %`` on each side
    (so neighbouring cells overlap slightly — no gap is missed).

    Each cell_group has a known digit (from DIGIT_GRID).
    Extract the symbol region (lower portion) of each cell_group
    and group by digit 1-9.

    Returns {digit: [list_of_symbol_grayscale_arrays]}.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    row_h = h_img / GRID_ROWS
    col_w = w_img / GRID_COLS

    results: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 10)}

    for ri in range(GRID_ROWS):
        # Cell row boundaries, enlarged by CELL_ENLARGE on each side
        y1 = max(0, int(ri * row_h - row_h * CELL_ENLARGE))
        y2 = min(h_img, int((ri + 1) * row_h + row_h * CELL_ENLARGE))
        cell_h = y2 - y1

        sym_y1 = y1 + int(cell_h * SYMBOL_TOP_FRAC)
        sym_y2 = y1 + int(cell_h * SYMBOL_BOT_FRAC)

        # Skip sample cells in the first row
        col_start = SAMPLE_COUNT if ri == 0 else 0

        for ci in range(col_start, GRID_COLS):
            # Cell column boundaries, enlarged by CELL_ENLARGE on each side
            x1 = max(0, int(ci * col_w - col_w * CELL_ENLARGE))
            x2 = min(w_img, int((ci + 1) * col_w + col_w * CELL_ENLARGE))
            digit = DIGIT_GRID[ri][ci]

            symbol = gray[sym_y1:sym_y2, x1:x2]
            if symbol.size < 10:
                continue
            results[digit].append(symbol)

    return results


# ══════════════════════════════════════════════════════════════
# PHASE 4 — Interactive review popup with selection
# ══════════════════════════════════════════════════════════════

# Layout constants for the review image
_CELL_W = 70
_ROW_H = 50
_TITLE_H = 45
_GAP = 3
_LABEL_W = 26


def _build_review_canvas(
    results: dict[int, list[np.ndarray]],
    filename: str,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int], list[int]]:
    """
    Build the review canvas and return cell metadata for interactivity.

    Returns
    -------
    canvas : np.ndarray
        Grayscale review image.
    rects : list[tuple[x, y, w, h]]
        Bounding box of each symbol on the canvas (pixel coords).
    digits : list[int]
        Digit (1-9) for each symbol.
    indices : list[int]
        Index within that digit's entry list for each symbol.
    """
    max_entries = max(len(v) for v in results.values())
    display_n = min(max_entries, 50)

    total_w = display_n * (_CELL_W + _GAP) + _GAP + _LABEL_W + 10
    total_h = _TITLE_H + 9 * (_ROW_H + _GAP) + _GAP + 40  # +40 for counter bar

    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Title
    cv2.putText(canvas, f"File: {filename}",
                (5, _TITLE_H - 8), font, 0.6, 0, 2, cv2.LINE_AA)
    cv2.line(canvas, (0, _TITLE_H - 1), (total_w, _TITLE_H - 1), 180, 1)

    rects: list[tuple[int, int, int, int]] = []
    digits: list[int] = []
    indices: list[int] = []

    for d in range(1, 10):
        row_y = _TITLE_H + _GAP + (d - 1) * (_ROW_H + _GAP)
        entries = results.get(d, [])

        # Label
        cv2.putText(canvas, f"{d}", (3, row_y + _ROW_H - 8),
                    font, 0.5, 50, 1, cv2.LINE_AA)
        cv2.line(canvas, (_LABEL_W - 4, row_y),
                 (_LABEL_W - 4, row_y + _ROW_H), 200, 1)

        x_pos = _LABEL_W + _GAP
        for idx, sym in enumerate(entries[:display_n]):
            resized = cv2.resize(sym, (_CELL_W, _ROW_H),
                                 interpolation=cv2.INTER_AREA)
            canvas[row_y:row_y + _ROW_H, x_pos:x_pos + _CELL_W] = resized

            rects.append((x_pos, row_y, _CELL_W, _ROW_H))
            digits.append(d)
            indices.append(idx)

            x_pos += _CELL_W + _GAP

    return canvas, rects, digits, indices


def show_review_popup(results: dict[int, list[np.ndarray]],
                      filename: str) -> None:
    """
    Interactive review popup with selection (blitting‑accelerated).

    • Left‑click a cell  →  select / highlight green
    • Right‑click a cell →  deselect
    • Drag               →  select all cells inside the drag rectangle
    • Counter at bottom shows how many cells are selected
    • Q / Enter / close  →  quit
    """
    canvas, rects, _digits, _indices = _build_review_canvas(results, filename)
    n_cells = len(rects)

    fig, ax = plt.subplots(figsize=(14, 9.5))

    ax.imshow(canvas, cmap="gray", vmin=0, vmax=255,
              interpolation="nearest")
    ax.set_title(
        "WAIS Digit Symbol Coding — Select cells  "
        "(Left=select  Right=deselect  Drag=box  Q=quit)",
        fontsize=11, fontweight="bold",
    )
    ax.axis("off")

    # ── Text elements ─────────────────────────────────────
    counter_text = ax.text(
        0.5, -0.03, "",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        ha="center", va="top", color="green",
    )

    # ── One reusable drag‑rectangle patch ─────────────────
    drag_rect = Rectangle(
        (0, 0), 0, 0,
        linewidth=1.5, edgecolor="cyan", facecolor="cyan",
        alpha=0.12, linestyle="--", visible=False,
    )
    ax.add_patch(drag_rect)

    # Maps cell_index → Rectangle patch for O(1) add/remove
    sel_patches: dict[int, Rectangle] = {}

    # State
    drag_origin: tuple[float, float] | None = None
    _bg: object | None = None  # saved blit background, refreshed on resize
    _resizing = False  # guard against recursive resize_draw() calls

    # ── Blit helpers ──────────────────────────────────────

    _ANIMATED_ARTISTS: list = []

    def _blit() -> None:
        nonlocal _bg
        if _bg is None:
            return
        fig.canvas.restore_region(_bg)  # type: ignore[arg-type]
        for a in _ANIMATED_ARTISTS:
            ax.draw_artist(a)
        if drag_rect.get_visible():
            ax.draw_artist(drag_rect)
        for p in sel_patches.values():
            ax.draw_artist(p)
        fig.canvas.blit(fig.bbox)

    def _full_refresh() -> None:
        """Full redraw + re‑capture background (used on window resize).
        Temporarily strips animated artists so they never bake into _bg."""
        nonlocal _bg
        # Remember state
        old_text = counter_text.get_text()
        old_patches = dict(sel_patches)
        # Strip
        counter_text.set_text("")
        for p in old_patches.values():
            p.remove()
        sel_patches.clear()
        fig.canvas.draw()
        _bg = fig.canvas.copy_from_bbox(fig.bbox)
        # Restore
        counter_text.set_text(old_text)
        for idx in old_patches:
            _select(idx)
        _blit()

    # ── Cell ops (dict‑backed, O(1)) ──────────────────────

    def _select(idx: int) -> None:
        if idx in sel_patches:
            return
        x, y, w, h = rects[idx]
        patch = Rectangle((x, y), w, h,
                          linewidth=0, facecolor="lime", alpha=0.25,
                          edgecolor=None)
        ax.add_patch(patch)
        sel_patches[idx] = patch

    def _deselect(idx: int) -> None:
        patch = sel_patches.pop(idx, None)
        if patch is None:
            return
        patch.remove()

    def _cell_at(x: float, y: float) -> int | None:
        for i, (cx, cy, cw, ch) in enumerate(rects):
            if cx <= x <= cx + cw and cy <= y <= cy + ch:
                return i
        return None

    def _cells_in_rect(x1: float, y1: float,
                       x2: float, y2: float) -> list[int]:
        x_lo, x_hi = min(x1, x2), max(x1, x2)
        y_lo, y_hi = min(y1, y2), max(y1, y2)
        return [i for i, (cx, cy, cw, ch) in enumerate(rects)
                if x_lo <= cx + cw / 2 <= x_hi and
                   y_lo <= cy + ch / 2 <= y_hi]

    # ── Event handlers ───────────────────────────────────

    def on_press(event: MouseEvent) -> None:
        nonlocal drag_origin
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            drag_origin = (event.xdata, event.ydata)
        elif event.button == 3:
            idx = _cell_at(event.xdata, event.ydata)
            if idx is not None:
                _deselect(idx)
                counter_text.set_text(
                    f"Selected: {len(sel_patches)} / {n_cells}")
                _blit()

    def on_motion(event: MouseEvent) -> None:
        if event.inaxes != ax or drag_origin is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, y0 = drag_origin
        x1, y1 = event.xdata, event.ydata
        x_lo, x_hi = min(x0, x1), max(x0, x1)
        y_lo, y_hi = min(y0, y1), max(y0, y1)

        drag_rect.set_bounds(x_lo, y_lo, x_hi - x_lo, y_hi - y_lo)
        if not drag_rect.get_visible():
            drag_rect.set_visible(True)
        _blit()

    def on_release(event: MouseEvent) -> None:
        nonlocal drag_origin
        if event.button != 1 or drag_origin is None:
            return

        x0, y0 = drag_origin
        x1, y1 = (event.xdata, event.ydata) if (
            event.xdata is not None and event.ydata is not None
        ) else (x0, y0)

        drag_origin = None
        drag_rect.set_visible(False)

        dist = np.hypot(x1 - x0, y1 - y0)
        changed = False
        if dist > 5:
            for idx in _cells_in_rect(x0, y0, x1, y1):
                if idx not in sel_patches:
                    _select(idx)
                    changed = True
        else:
            idx = _cell_at(x1, y1)
            if idx is not None and idx not in sel_patches:
                _select(idx)
                changed = True

        if changed:
            counter_text.set_text(
                f"Selected: {len(sel_patches)} / {n_cells}")
        _blit()

    def on_key(event: KeyEvent) -> None:
        if event.key in ("enter", "space"):
            plt.close(fig)  # return to caller → continue
        elif event.key in ("q", "Q", "escape"):
            sys.exit(0)      # quit entirely

    def on_close(_event: CloseEvent) -> None:
        sys.exit(0)

    def on_resize(_event) -> None:
        """Re‑render everything cleanly when the window is resized."""
        nonlocal _resizing
        if _resizing:
            return
        _resizing = True
        _full_refresh()
        _resizing = False

    # ── Connect events ───────────────────────────────────
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("resize_event", on_resize)

    # ── Initial full render + save background (WITHOUT text) ──
    plt.tight_layout()
    fig.canvas.draw()
    _bg = fig.canvas.copy_from_bbox(fig.bbox)

    # Text lives purely as an animated overlay (never baked into _bg)
    counter_text.set_text(f"Selected: 0 / {n_cells}")
    _ANIMATED_ARTISTS.extend([counter_text])
    _blit()  # render the initial text on screen

    plt.show()
    # If we get here, Enter/Space was pressed → return to caller for re-loop
    return


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def _pick_image_via_dialog() -> str | None:
    """Open a native file‑chooser dialog and return the selected path."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    path = filedialog.askopenfilename(
        title="Select a WAIS form image",
        filetypes=[("Image files",
                    "*.jpg *.jpeg *.png *.bmp *.tiff"),
                   ("All files", "*.*")],
    )
    root.destroy()
    return path if path else None


def _capture_from_camera() -> tuple[np.ndarray, str]:
    """
    Open the default camera, show live preview in an OpenCV window.
    Press **any key** to capture the current frame and proceed.
    ESC quits entirely.

    Returns (image, label) where label is a short descriptive name.
    """
    print("       Opening camera (press any key to capture, ESC to quit) …")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("       ERROR: Could not open camera.  Exiting.")
        sys.exit(1)

    # Allow a moment for the camera to warm up
    for _ in range(10):
        cap.read()

    captured: np.ndarray | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("       Camera read failed.  Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)

        # Mirror horizontally for intuitive left-right movement
        display = cv2.flip(frame, 1)

        # Overlay instruction text
        cv2.putText(display, "Position the WAIS form, then press ANY KEY to capture",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "ESC to quit",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        cv2.imshow("WAIS Camera Capture", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            print("       Capture cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        if key != 255:  # any key pressed (not a timeout)
            # Capture without the mirror & overlay
            captured = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    label = "camera_capture.jpg"
    print(f"       Captured {captured.shape[1]}×{captured.shape[0]} px")
    return captured, label


def show_menu() -> int:
    """
    Show a menu window with two wide rectangular buttons using tkinter.
      Button 1 → Camera
      Button 2 → Select File
    Q / Esc → quit.

    Returns the choice (1 or 2).
    """
    import tkinter as tk
    from tkinter import font as tkfont

    root = tk.Tk()
    root.title("WAIS Digit Symbol Coding")
    root.configure(bg="#f0f0f0")
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", sys.exit)

    # Center the window
    win_w, win_h = 380, 260
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - win_w) // 2
    y = (sh - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    # Bind keys
    def _on_key(event):
        if event.keysym == "1":
            _choose(1)
        elif event.keysym == "2":
            _choose(2)
        elif event.keysym in ("q", "Q", "Escape"):
            sys.exit(0)

    root.bind("<Key>", _on_key)

    choice: list[int] = []

    def _choose(val: int) -> None:
        choice.append(val)
        root.destroy()
        root.quit()

    # Title
    title_font = tkfont.Font(family="Helvetica", size=14, weight="bold")
    tk.Label(root, text="WAIS Digit Symbol Coding", font=title_font,
             bg="#f0f0f0", fg="#333").pack(pady=(20, 18))

    # Button 1: Camera
    btn1 = tk.Button(
        root, text="1    Camera", font=("Helvetica", 12, "bold"),
        bg="#4CAF50", fg="white", activebackground="#388E3C",
        activeforeground="white", relief="raised", bd=3,
        cursor="hand2", width=30, height=2, command=lambda: _choose(1),
    )
    btn1.pack(pady=(0, 10))

    # Button 2: Select File
    btn2 = tk.Button(
        root, text="2    Select File", font=("Helvetica", 12, "bold"),
        bg="#2196F3", fg="white", activebackground="#1976D2",
        activeforeground="white", relief="raised", bd=3,
        cursor="hand2", width=30, height=2, command=lambda: _choose(2),
    )
    btn2.pack(pady=(0, 10))

    # Focus so keyboard works
    root.focus_set()
    btn1.focus_set()

    root.mainloop()

    if not choice:
        sys.exit(0)
    return choice[0]


def _run_pipeline(img: np.ndarray, filename: str) -> None:
    """Run the full pipeline on one image.
    Returns normally → Enter/Space was pressed (caller should re-loop).
    Never returns on Q/Esc/X (calls sys.exit in popups).
    """
    # ── 1. Load & corners ───────────────────────────────
    print("\n[1/4] Loading input image …")
    print("       Stage 1 — Click 4 grid corners (right-click to undo).")
    print("       Stage 2 — Drag any of the 9 control points to warp.")
    print("                 Press Enter/Space to confirm and optimise.")
    corners = select_four_corners(img, filename)

    # ── 2. Perspective correction ───────────────────────
    print("\n[2/4] Applying perspective correction …")
    warped = perspective_correct(img, corners)
    h_crop, w_crop = warped.shape[:2]
    print(f"       Corrected grid size: {w_crop}×{h_crop} px")

    # ── 3. Extract symbols ──────────────────────────────
    print("\n[3/4] Extracting symbol entries …")
    results = extract_symbols(warped)

    total = sum(len(v) for v in results.values())
    print(f"       Total cell_groups processed: {total} / {GRID_ROWS * GRID_COLS - SAMPLE_COUNT}")
    for d in range(1, 10):
        print(f"         Digit {d}: {len(results[d])} entries")

    # ── 4. Show review popup ────────────────────────────
    print("\n[4/4] Displaying review popup …")
    print("       Enter/Space → continue with next  |  Q/Esc → quit")
    show_review_popup(results, filename)


def main() -> None:
    print(f"\n{'=' * 60}")
    print("WAIS Digit Symbol Coding — Grouping Tool")
    print(f"{'=' * 60}")

    if len(sys.argv) >= 2:
        # CLI arg: process once and exit
        input_path = sys.argv[1]
        img = load_image(input_path)
        filename = os.path.basename(input_path)
        print(f"Input: {input_path}")
        _run_pipeline(img, filename)
        return

    # No CLI arg: show menu and loop
    while True:
        menu_choice = show_menu()
        print(f"       You chose: {'Camera' if menu_choice == 1 else 'Select File'}")

        if menu_choice == 1:
            img, filename = _capture_from_camera()
            print(f"Input: camera capture ({filename})")
        else:
            input_path = _pick_image_via_dialog()
            if not input_path:
                print("       No file selected.  Returning to menu.")
                continue
            img = load_image(input_path)
            filename = os.path.basename(input_path)
            print(f"Input: {input_path}")

        _run_pipeline(img, filename)
        # If we get here, user pressed Enter/Space in review popup → loop
        print("       Continuing (Enter/Space).  Close or press Q/Esc to quit.")
        # Final review popup returns → loop back with same menu choice


if __name__ == "__main__":
    main()
