#!/usr/bin/env python3
"""
WAIS Digit Symbol Coding — Grid Digit Grouping & Review Tool

Usage:
    python wais_digit_grouping.py <input_image>

Workflow:
  1. Load input image → display in popup
  2. User clicks 4 corners of the entry grid (TL→TR→BR→BL)
  3. Load digit references from template.jpg key boxes
  4. Perspective‑correct the grid region
  5. Divide corrected grid into 7×20 equal cells.
     For each cell: binary‑match digit (1‑9) against template key digits,
     then extract participant's drawn symbol from the lower portion.
  6. Show ONE popup: file name at top + 9 rows (one per digit),
     each row containing all participant entries for that digit.
  7. Press Q / Enter / close popup → quit

Dependencies: opencv-python, numpy, matplotlib
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
GRID_ROWS     = 7
GRID_COLS     = 20
DIGIT_COUNT   = 9

# Fraction of cell height used for digit matching
DIGIT_HEIGHT_FRAC = 0.35

# Symbol extraction boundaries (fractions of cell height)
SYMBOL_TOP_FRAC    = 0.38
SYMBOL_BOT_FRAC    = 0.92


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
    Display the image.  User clicks TL → TR → BR → BL.
    Returns list of four (x, y) tuples.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(img_rgb)
    ax.set_title(
        f"Click the 4 corners of the ENTRY GRID:\n"
        f"1=Top‑Left  2=Top‑Right  3=Bottom‑Right  4=Bottom‑Left\n"
        f"File: {filename}",
        fontsize=11, fontweight="bold",
    )
    ax.axis("on")

    corners: list[tuple[int, int]] = []
    colors   = ["red", "lime", "cyan", "magenta"]
    labels   = ["TL", "TR", "BR", "BL"]

    def on_click(event):
        if event.inaxes != ax or len(corners) >= 4:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        corners.append((x, y))
        idx = len(corners) - 1
        ax.plot(x, y, "o", color=colors[idx], markersize=10)
        ax.annotate(
            f"{labels[idx]}  ({x}, {y})",
            (x, y),
            fontsize=10, color=colors[idx], fontweight="bold",
            xytext=(8, 8), textcoords="offset pixels",
        )
        fig.canvas.draw()
        if len(corners) == 4:
            poly = plt.Polygon(corners, fill=False,
                               edgecolor="yellow", linewidth=2, linestyle="--")
            ax.add_patch(poly)
            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if len(corners) != 4:
        print("ERROR: You must click exactly 4 corners.")
        sys.exit(1)
    return corners


# ══════════════════════════════════════════════════════════════
# PHASE 2 — Load digit references from template.jpg key boxes
#           (cleanest source for the printed digit patterns)
# ══════════════════════════════════════════════════════════════

def load_digit_references() -> list[np.ndarray]:
    """
    Extract the 9 digit reference images from the key‑box area of
    *template.jpg* (upper ~45 % of each key box).

    Returns a list of 9 grayscale images (CLAHE‑enhanced).
    """
    template_path = Path(__file__).parent / "template.jpg"
    if not template_path.exists():
        print(f"ERROR: template.jpg not found at {template_path}")
        sys.exit(1)

    gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    top = enhanced[:400, :]
    _, bw = cv2.threshold(top, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in cts:
        x, y, w, h = cv2.boundingRect(cnt)
        if 3000 < w * h < 8000 and 0.4 < w / h < 1.0:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    boxes = boxes[:9]

    refs = []
    for x, y, w, h in boxes:
        roi = enhanced[y: y + int(h * 0.45), x: x + w]
        refs.append(roi)

    print(f"  Loaded {len(refs)} digit references from template.jpg")
    return refs


# ══════════════════════════════════════════════════════════════
# PHASE 3 — Perspective correction
# ══════════════════════════════════════════════════════════════

def perspective_correct(
    img: np.ndarray, src_corners: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the quadrilateral into a rectangle.
    Returns (warped_image, homography_matrix)."""
    src = np.array(src_corners, dtype=np.float32)

    def _dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    w = int(round(max(_dist(src[0], src[1]), _dist(src[3], src[2]))))
    h = int(round(max(_dist(src[0], src[3]), _dist(src[1], src[2]))))

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M


# ══════════════════════════════════════════════════════════════
# PHASE 4 — Grid processing (digit identification + extraction)
# ══════════════════════════════════════════════════════════════

def _prepare_binary_refs(
    digit_refs: list[np.ndarray],
    std_w: int = 20,
    std_h: int = 30,
) -> list[np.ndarray]:
    """Convert grayscale digit references to standardised binary images."""
    binary_refs = []
    for ref in digit_refs:
        _, bin_ref = cv2.threshold(ref, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_refs.append(cv2.resize(bin_ref, (std_w, std_h)))
    return binary_refs


def _best_digit_match(
    cell_digit_roi: np.ndarray,
    binary_refs: list[np.ndarray],
    threshold: float = 0.28,
    min_fg_px: int = 8,
) -> int:
    """
    Match *cell_digit_roi* (grayscale) against 9 standardised binary digit
    references using IoU (overlap) + pixel agreement.

    *threshold*     — minimum combined score to accept a match.
    *min_fg_px*     — minimum foreground pixels in binarised ROI
                      (cells with less content are skipped as noise).

    Returns digit 1…9, or 0 if no reliable match.
    """
    h, w = cell_digit_roi.shape
    if h < 5 or w < 5:
        return 0

    # Binarise the cell ROI
    _, bin_cell = cv2.threshold(cell_digit_roi, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Skip cells with negligible content (likely blank / noise)
    if np.sum(bin_cell > 0) < min_fg_px:
        return 0

    # Resize to common standard size
    std_w, std_h = binary_refs[0].shape[1], binary_refs[0].shape[0]
    query = cv2.resize(bin_cell, (std_w, std_h))

    best_digit = 0
    best_score = -1.0

    for d_idx, bin_ref in enumerate(binary_refs):
        # Intersection over Union of foreground (white) pixels
        fg_and = np.sum((query > 0) & (bin_ref > 0))
        fg_or  = np.sum((query > 0) | (bin_ref > 0))
        iou = fg_and / max(fg_or, 1)

        # Pixel‑wise agreement (both foreground or both background)
        agree = np.sum(query == bin_ref) / query.size

        score = 0.6 * iou + 0.4 * agree

        if score > best_score:
            best_score = score
            best_digit = d_idx + 1

    return best_digit if best_score >= threshold else 0


def process_grid(
    warped: np.ndarray,
    digit_refs: list[np.ndarray],
) -> dict[int, list[np.ndarray]]:
    """
    Divide the perspective-corrected grid into 7×20 cells.
    For each cell: identify its pre‑printed digit using the digit
    references extracted from the input's own key section, then
    extract the participant's symbol from the lower portion.

    Returns {digit_1…9: [list_of_symbol_images_as_grayscale_arrays]}.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # Pre‑compute binary digit references (standardised size)
    binary_refs = _prepare_binary_refs(digit_refs)

    # Divide the corrected rectangle into equal 7×20 cells.
    # The entire warped image IS the grid (user clicked its corners).
    row_height = h_img / GRID_ROWS
    col_width  = w_img / GRID_COLS

    results: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 10)}

    for ri in range(GRID_ROWS):
        y1 = int(ri * row_height)
        y2 = int((ri + 1) * row_height)
        cell_h = y2 - y1

        # Digit region: top ~35 % of cell
        digit_split = int(cell_h * DIGIT_HEIGHT_FRAC)
        # Symbol region: middle band (below the printed digit)
        sym_y1 = y1 + int(cell_h * SYMBOL_TOP_FRAC)
        sym_y2 = y1 + int(cell_h * SYMBOL_BOT_FRAC)

        for ci in range(GRID_COLS):
            x1 = int(ci * col_width)
            x2 = int((ci + 1) * col_width)

            # ── Digit ROI ──────────────────────────────
            digit_roi = gray[y1: y1 + digit_split, x1:x2]
            if digit_roi.size < 20:
                continue

            digit = _best_digit_match(digit_roi, binary_refs)
            if digit == 0:
                # Retry with a slightly larger region
                alt = gray[y1: y1 + int(cell_h * 0.50), x1:x2]
                digit = _best_digit_match(alt, binary_refs)
            if digit == 0:
                continue

            # ── Symbol ROI ─────────────────────────────
            symbol = gray[sym_y1:sym_y2, x1:x2]
            if symbol.size < 20:
                continue

            results[digit].append(symbol)

    # Log counts
    for d in range(1, 10):
        print(f"  Digit {d}: {len(results[d])} entries")

    return results


# ══════════════════════════════════════════════════════════════
# PHASE 5 — Display grouped review popup
# ══════════════════════════════════════════════════════════════

def build_review_image(
    results: dict[int, list[np.ndarray]],
    filename: str,
    max_cell_w: int = 70,
    row_h: int = 50,
    title_h: int = 45,
    gap: int = 3,
) -> np.ndarray:
    """
    Build a single grayscale image:
      • Title bar at top (file name)
      • 9 rows, one per digit, showing all participant symbol entries
    """
    max_entries = max(len(v) for v in results.values())
    display_n = min(max_entries, 50)

    total_w = display_n * (max_cell_w + gap) + gap + 30  # +30 for label
    total_h = title_h + 9 * (row_h + gap) + gap

    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # ── Title ───────────────────────────────────────────
    cv2.putText(canvas, f"File: {filename}",
                (5, title_h - 8), font, 0.6, 0, 2, cv2.LINE_AA)
    cv2.line(canvas, (0, title_h - 1), (total_w, title_h - 1), 180, 1)

    # ── Rows ────────────────────────────────────────────
    for d in range(1, 10):
        row_y = title_h + gap + (d - 1) * (row_h + gap)
        entries = results.get(d, [])

        # Label
        cv2.putText(canvas, f"{d}", (3, row_y + row_h - 8),
                    font, 0.5, 50, 1, cv2.LINE_AA)
        cv2.line(canvas, (22, row_y), (22, row_y + row_h), 200, 1)

        x_pos = 26
        for sym in entries[:display_n]:
            resized = cv2.resize(sym, (max_cell_w, row_h),
                                 interpolation=cv2.INTER_AREA)
            canvas[row_y:row_y + row_h, x_pos:x_pos + max_cell_w] = resized
            x_pos += max_cell_w + gap

    return canvas


def show_review_popup(
    results: dict[int, list[np.ndarray]],
    filename: str,
) -> None:
    """Display the review image.  Q / Enter / close → quit."""
    img = build_review_image(results, filename)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        "WAIS Digit Symbol Coding — Review  (press Q or Enter to quit)",
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")

    quit_flag = [False]

    def _quit(*_args) -> None:
        quit_flag[0] = True
        plt.close(fig)

    def on_key(event: KeyEvent) -> None:
        if event.key in ("q", "Q", "enter", "escape"):
            _quit()

    def on_close(_event: CloseEvent) -> None:
        quit_flag[0] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()

    sys.exit(0)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python wais_digit_grouping.py <input_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    filename = os.path.basename(input_path)

    print(f"\n{'=' * 60}")
    print("WAIS Digit Symbol Coding — Grouping Tool")
    print(f"{'=' * 60}")
    print(f"Input: {input_path}")

    # ── 1. Load & corners ───────────────────────────────
    print("\n[1/5] Loading input image …")
    img = load_image(input_path)
    print("       Click the 4 corners of the grid in the popup.")
    corners = select_four_corners(img, filename)

    # ── 2. Load digit references from template.jpg ──────
    print("\n[2/5] Loading digit references from template.jpg …")
    digit_refs = load_digit_references()

    # ── 3. Perspective correction ───────────────────────
    print("\n[3/5] Applying perspective correction …")
    warped, _ = perspective_correct(img, corners)
    h_crop, w_crop = warped.shape[:2]
    print(f"       Corrected grid size: {w_crop}×{h_crop} px")

    # ── 4. Process grid ─────────────────────────────────
    print("\n[4/5] Processing grid cells …")
    results = process_grid(warped, digit_refs)

    total = sum(len(v) for v in results.values())
    print(f"       Total cells processed: {total} / {GRID_ROWS * GRID_COLS}")

    if total == 0:
        print("ERROR: No cells were successfully processed.")
        print("       Check that the input image shows a valid WAIS form.")
        sys.exit(1)

    # ── 5. Show review popup ────────────────────────────
    print("\n[5/5] Displaying review popup …")
    print("       Press Q or Enter to quit.  Close window also quits.")
    show_review_popup(results, filename)


if __name__ == "__main__":
    main()
