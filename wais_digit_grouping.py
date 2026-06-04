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
     For each cell_group: extract the symbol region below the digit.
  5. Show ONE popup: file name at top + 9 rows (one per digit),
     each row containing all participant entries for that digit.
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
from matplotlib.backend_bases import KeyEvent, CloseEvent

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
GRID_ROWS = 7
GRID_COLS = 20

# Symbol extraction boundaries (fractions of cell_group height)
SYMBOL_TOP_FRAC = 0.38
SYMBOL_BOT_FRAC = 0.92

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
    The 4-corner perspective correction already handles most paper
    warp; equal division of the corrected rectangle is the most
    robust approach for forms with faint grid lines.

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
        y1 = int(ri * row_h)
        y2 = int((ri + 1) * row_h)
        cell_h = y2 - y1

        sym_y1 = y1 + int(cell_h * SYMBOL_TOP_FRAC)
        sym_y2 = y1 + int(cell_h * SYMBOL_BOT_FRAC)

        for ci in range(GRID_COLS):
            x1 = int(ci * col_w)
            x2 = int((ci + 1) * col_w)
            digit = DIGIT_GRID[ri][ci]

            symbol = gray[sym_y1:sym_y2, x1:x2]
            if symbol.size < 10:
                continue
            results[digit].append(symbol)

    return results


# ══════════════════════════════════════════════════════════════
# PHASE 4 — Display grouped review popup
# ══════════════════════════════════════════════════════════════

def build_review_image(
    results: dict[int, list[np.ndarray]],
    filename: str,
    cell_w: int = 70,
    row_h: int = 50,
    title_h: int = 45,
    gap: int = 3,
) -> np.ndarray:
    """
    Build a single grayscale image:
      • Title bar at top (file name)
      • 9 rows, one per digit, showing all extracted symbol entries
    """
    max_entries = max(len(v) for v in results.values())
    display_n = min(max_entries, 50)

    total_w = display_n * (cell_w + gap) + gap + 30
    total_h = title_h + 9 * (row_h + gap) + gap

    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(canvas, f"File: {filename}",
                (5, title_h - 8), font, 0.6, 0, 2, cv2.LINE_AA)
    cv2.line(canvas, (0, title_h - 1), (total_w, title_h - 1), 180, 1)

    # Rows
    for d in range(1, 10):
        row_y = title_h + gap + (d - 1) * (row_h + gap)
        entries = results.get(d, [])

        # Label
        cv2.putText(canvas, f"{d}", (3, row_y + row_h - 8),
                    font, 0.5, 50, 1, cv2.LINE_AA)
        cv2.line(canvas, (22, row_y), (22, row_y + row_h), 200, 1)

        x_pos = 26
        for sym in entries[:display_n]:
            resized = cv2.resize(sym, (cell_w, row_h),
                                 interpolation=cv2.INTER_AREA)
            canvas[row_y:row_y + row_h, x_pos:x_pos + cell_w] = resized
            x_pos += cell_w + gap

    return canvas


def show_review_popup(results: dict[int, list[np.ndarray]],
                      filename: str) -> None:
    """Display the review image.  Q / Enter / close → quit."""
    img = build_review_image(results, filename)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        "WAIS Digit Symbol Coding — Review  (press Q or Enter to quit)",
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")

    def _quit(*_args) -> None:
        plt.close(fig)

    def on_key(event: KeyEvent) -> None:
        if event.key in ("q", "Q", "enter", "escape"):
            _quit()

    def on_close(_event: CloseEvent) -> None:
        _quit()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)
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
