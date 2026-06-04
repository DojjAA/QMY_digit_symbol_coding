# QMY_digit_symbol_coding — Full Codebase Audit

> **Audit Date:** 2026-06-04
> **Auditor:** OpenHands AI Agent
> **Repository:** `DojjAA/QMY_digit_symbol_coding` (GitHub)
> **Branch:** `main`

---

## 1. Repository Overview

| Property | Value |
|---|---|
| Owner | DojjAA |
| Repo Name | QMY_digit_symbol_coding |
| Default Branch | `main` |
| Total Commits | 2 (`init`, images + .gitignore) |
| Python Script | `wais_digit_grouping.py` (467 lines) |
| Reference Images | `template.jpg` (blank form), `sample.jpg` (completed form) |
| Dependencies | opencv-python, numpy, matplotlib |

### Git History

```
f10e5f2 (HEAD -> main, origin/main) Add images + .gitignore
5e4a60c init
```

- Author: `Lan3a <lanalanspace@gmail.com>`

---

## 2. File Inventory

| File | Size | Description |
|---|---|---|
| `wais_digit_grouping.py` | ~19 KB | **Smart** version — pen-mark removal + multi-channel edge fusion for grid detection |
| `wais_digit_grouping_simple.py` | ~15 KB | **Simple** version — equal-division only (the original stable one) |
| `template.jpg` | 229 KB | Blank WAIS form (grid only, no entries) |
| `sample.jpg` | 193 KB | Completed WAIS form (with participant entries) |
| `README.md` | 0 B | Empty |
| `.gitignore` | 0 B | Empty |
| `AUDIT.md` | This file | Full audit documentation |

---

## 3. Script Architecture

### `wais_digit_grouping_simple.py` (equal-division)

```
Phase 1: Load image + user clicks 4 grid corners
Phase 2: Perspective correction
Phase 3: Equal 7×20 division → symbol extraction
Phase 4: Grouped review popup (Q/Enter/close → quit)
```

### `wais_digit_grouping.py` (smart grid detection)

```
Phase 1: Load image + user clicks 4 grid corners
Phase 2: Perspective correction
Phase 3: Smart grid detection + symbol extraction
  ├── _remove_pen_marks() — inpaint over coloured & thick pen strokes
  ├── _min_channel_edge() — multi-channel edge fusion
  ├── _detect_grid_clever() — morphology + projection peak-finding
  ├── _smart_peaks() — adaptive thresholding with scoring
  └── _refine_peaks() — local edge snapping
Phase 4: Grouped review popup (Q/Enter/close → quit)
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Two scripts** | Smart version handles pen-over-line cases; simple version is always-stable fallback |
| **Hardcoded DIGIT_GRID** | User provided the exact 7×20 digit sequence; avoids fragile auto-detection |
| **Right-click undo** | Allows fixing misclicks without restarting |
| **Pen inpainting (smart)** | HSV color masking for coloured pens; thickness filtering for black pens |
| **Multi-channel edge fusion (smart)** | Min across BGR channels suppresses coloured pen edges while preserving black printed lines |
| **Adaptive peak scoring** | Scores peaks by height × isolation; pads/trims to exactly 8×21 boundaries |
| **Equal division fallback** | Always safe — used when smart detection is unreliable |

### Digit Grid (Row × Column)

```
Row 1: 2 1 3 7 2 4 8 2 1 3 2 1 4 2 3 5 2 3 1 4
Row 2: 5 6 3 1 4 1 5 4 2 7 6 3 5 7 2 8 5 4 6 3
Row 3: 7 2 8 1 9 5 8 4 7 3 6 2 5 1 9 2 8 3 7 4
Row 4: 6 5 9 4 8 3 7 2 6 1 5 4 6 3 7 9 2 8 1 7
Row 5: 9 4 6 8 5 9 7 1 8 5 2 9 4 8 6 3 7 9 8 6
Row 6: 2 7 3 6 5 1 9 8 4 5 7 3 1 4 8 7 9 1 4 5
Row 7: 7 1 8 2 9 3 6 7 2 8 5 2 3 1 4 8 4 2 7 6
```

---

## 4. Image Analysis Results

### Template.jpg

- **1200×1600 px**, grayscale
- **9 key boxes** detected at: x=202..965, y=251..265, each ~51×86 px
- **Grid region:** x=67..1199, y=373..1272 (1132×899 px)
- **Row heights:** 122–146 px (7 rows)
- **Column width:** ~57 px (20 columns)
- **Grid lines:** Horizontal lines clearly visible; vertical lines faint
- **Pre-printed digits in grid:** Present but very faint (confirmed via binary matching)

### Sample.jpg

- **1200×1600 px**, very low contrast (key boxes barely visible)
- **3 key boxes** found without CLAHE; 9 with CLAHE enhancement
- **Participant entries:** Present in grid, visible in difference analysis
- **Template matching on grid:** ~57/60 first-3-row cells matched with all 9 digits

---

## 5. Usage

```bash
# Run with a completed WAIS form photo
python wais_digit_grouping.py /path/to/input.jpg

# Or double-click (no args) to get a file dialog
python wais_digit_grouping.py

# Workflow:
# 1. Left-click 4 grid corners (TL→TR→BR→BL), right-click to undo
# 2. Script perspective-corrects and extracts symbol regions
# 3. Review popup shows entries grouped by digit (9 rows)
# 4. Press Q, Enter, or close window to quit
```

### Dependencies

```bash
pip install opencv-python numpy matplotlib
```

---

## 6. Potential Improvements (For Next AI)

| Area | Suggestion |
|---|---|
| **Symbol clarity** | Invert symbol ROI (white bg → black ink) for better visibility in review |
| **Review interactivity** | Add click‑to‑mark (correct/incorrect) and save results to CSV |
| **Grid alignment** | If equal division is off, detect row/column lines from the warped image |
| **Batch processing** | Add `--output` flag to save review image instead of displaying |

---

## 7. Git Metadata

```bash
# Remote
origin  https://github.com/DojjAA/QMY_digit_symbol_coding.git

# Current state
main branch, all files committed
```

