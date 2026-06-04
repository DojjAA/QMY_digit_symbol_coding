"""
WAIS Digit Symbol Coding — Grid Digit Grouping & Review Tool

Usage:
    Double-click the script, or run:
        python wais_digit_grouping.py [image_path]

    If no path is given, a file-open dialog appears.

Workflow:
  1. Open image (dialog or CLI arg) → display in popup
  2. User clicks 4 corners of the entry grid (TL→TR→BR→BL)
     Right-click to undo the last corner click.
  3. Perspective‑correct the grid region
  4. Divide corrected grid into 7×20 cell_groups using the
     known digit layout (hardcoded from the WAIS key).
     Each cell is enlarged by 10 % on each side for overlap.
     Extract the symbol region below the digit.
  5. Interactive review popup:
     • Left‑click a cell → select (green highlight)
     • Right‑click a cell → deselect
     • Drag to select multiple cells at once
     • Selected‑cell counter at bottom
  6. Press Q / Enter / close popup → quit

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
CELL_ENLARGE = 0.1

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
# PHASE 1 — Load image & let user click four corners
# ══════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Cannot load image from '{path}'")
        sys.exit(1)
    return img


def select_four_corners(img: np.ndarray, filename: str
                        ) -> list[tuple[int, int]]:
    """
    Display the image.  Left-click TL → TR → BR → BL.
    Right-click to undo the last click.
    Returns list of four (x, y) tuples.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(img_rgb)
    ax.set_title(
        "Click the 4 corners of the ENTRY GRID:\n"
        "Left-click: TL → TR → BR → BL     "
        "Right-click: undo last corner\n"
        f"File: {filename}",
        fontsize=11, fontweight="bold",
    )
    ax.axis("on")

    corners: list[tuple[int, int]] = []
    colors = ["red", "lime", "cyan", "magenta"]
    labels = ["TL", "TR", "BR", "BL"]
    # Store plot artists so we can remove them on undo
    artists: list = []

    def _redraw():
        """Clear and redraw all markers/polygon from scratch."""
        for a in artists:
            a.remove()
        artists.clear()
        ax.set_title(
            "Click the 4 corners of the ENTRY GRID:\n"
            "Left-click: TL → TR → BR → BL     "
            "Right-click: undo last corner\n"
            f"File: {filename}",
            fontsize=11, fontweight="bold",
        )
        for i, (x, y) in enumerate(corners):
            (pt,) = ax.plot(x, y, "o", color=colors[i], markersize=10)
            artists.append(pt)
            ann = ax.annotate(
                f"{labels[i]}  ({x}, {y})",
                (x, y),
                fontsize=10, color=colors[i], fontweight="bold",
                xytext=(8, 8), textcoords="offset pixels",
            )
            artists.append(ann)
        if len(corners) == 4:
            poly = plt.Polygon(corners, fill=False,
                               edgecolor="yellow", linewidth=2, linestyle="--")
            ax.add_patch(poly)
            artists.append(poly)
        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax:
            return
        # Right-click → undo
        if event.button == 3:
            if corners:
                corners.pop()
                _redraw()
            return
        # Left-click → add corner
        if event.button != 1 or len(corners) >= 4:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        corners.append((x, y))
        _redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if len(corners) != 4:
        print("ERROR: You must click exactly 4 corners.")
        sys.exit(1)
    return corners


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

        for ci in range(GRID_COLS):
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
        0.5, -0.03, "Selected: 0 / 0",
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

    # ── Blit helpers ──────────────────────────────────────

    _ANIMATED_ARTISTS: list = []  # filled after initial draw

    def _blit() -> None:
        """Redraw only animated artists on the saved background."""
        fig.canvas.restore_region(_bg)  # type: ignore[name-defined]
        for a in _ANIMATED_ARTISTS:
            ax.draw_artist(a)
        if drag_rect.get_visible():
            ax.draw_artist(drag_rect)
        for p in sel_patches.values():
            ax.draw_artist(p)
        fig.canvas.blit(fig.bbox)

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
        else:
            # Still remove drag rect visually
            fig.canvas.restore_region(_bg)  # type: ignore[name-defined]
            for p in sel_patches.values():
                ax.draw_artist(p)
            fig.canvas.blit(fig.bbox)

    def on_key(event: KeyEvent) -> None:
        if event.key in ("q", "Q", "enter", "escape"):
            plt.close(fig)

    def on_close(_event: CloseEvent) -> None:
        pass

    # ── Connect events ───────────────────────────────────
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    # ── Initial full render + save background ────────────
    plt.tight_layout()
    fig.canvas.draw()
    _bg = fig.canvas.copy_from_bbox(fig.bbox)  # type: ignore[name-defined]
    _ANIMATED_ARTISTS.extend([counter_text])

    plt.show()
    sys.exit(0)


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


def main() -> None:
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    else:
        print("No image given — opening file dialog …")
        input_path = _pick_image_via_dialog()
        if not input_path:
            print("No file selected.  Exiting.")
            sys.exit(0)

    filename = os.path.basename(input_path)

    print(f"\n{'=' * 60}")
    print("WAIS Digit Symbol Coding — Grouping Tool")
    print(f"{'=' * 60}")
    print(f"Input: {input_path}")

    # ── 1. Load & corners ───────────────────────────────
    print("\n[1/4] Loading input image …")
    img = load_image(input_path)
    print("       Left-click 4 grid corners (right-click to undo).")
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
    print(f"       Total cell_groups processed: {total} / {GRID_ROWS * GRID_COLS}")
    for d in range(1, 10):
        print(f"         Digit {d}: {len(results[d])} entries")

    # ── 4. Show review popup ────────────────────────────
    print("\n[4/4] Displaying review popup …")
    print("       Press Q or Enter to quit.  Close window also quits.")
    show_review_popup(results, filename)


if __name__ == "__main__":
    main()
