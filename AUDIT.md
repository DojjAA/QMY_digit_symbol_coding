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
| `wais_digit_grouping.py` | ~15 KB | Main Python script |
| `template.jpg` | 229 KB | Blank WAIS form (grid only, no entries) |
| `sample.jpg` | 193 KB | Completed WAIS form (with participant entries) |
| `README.md` | 0 B | Empty |
| `.gitignore` | 0 B | Empty |
| `AUDIT.md` | This file | Full audit documentation |

---

## 3. Script Architecture — `wais_digit_grouping.py`

```
wais_digit_grouping.py
├── Phase 1: Load image + user clicks 4 grid corners (matplotlib)
├── Phase 2: Load digit references from template.jpg key boxes
├── Phase 3: Perspective correction of the grid region
├── Phase 4: Divide grid → 7×20 equal cells
│   ├── For each cell: binarise top 35% → binary IoU match vs 9 digit refs
│   └── Extract participant's symbol from lower 38–92% of cell
└── Phase 5: Single popup with filename + 9 rows (one per digit)
    └── Q / Enter / close → quit
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Binary IoU matching** (not pixel NCC) | Robust to contrast/lighting differences across scans |
| **Digit refs from template.jpg** (not input) | Template has cleanest key boxes; input may be low-quality |
| **Equal 7×20 grid division** (not line detection) | WAIS forms have uniform grid; line detection is fragile |
| **Single popup** (not 9 separate) | User requested "1 pic" with 9 rows |
| **Border margins in symbol crop** (38–92 %) | Avoids digit contamination of symbol region |

### Matching Algorithm

```
For each cell in 7×20 grid:
  1. Extract top 35% → binarise (Otsu)
  2. Skip if < 8 foreground pixels (no content)
  3. Resize to 20×30 standard size
  4. For each digit ref (1-9):
     a. Binarise ref → resize to 20×30
     b. Compute Foreground IoU = AND / OR
     c. Compute Pixel Agreement = equal_pixels / total
     d. Score = 0.6 × IoU + 0.4 × Agreement
  5. Accept match if best score ≥ 0.28
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

# Workflow:
# 1. Click 4 corners of the grid in the popup (TL→TR→BR→BL)
# 2. Script perspective-corrects and processes
# 3. Review popup shows entries grouped by digit (1 row per digit)
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
| **Digit matching** | If matching fails on real inputs, try contour‑based Hu moments or install Tesseract OCR |
| **Symbol clarity** | Invert symbol ROI (white bg → black ink) for better visibility in review |
| **Review interactivity** | Add click‑to‑mark (correct/incorrect) and save results to CSV |
| **Grid alignment** | If equal division is off, detect row/column lines from the warped image |
| **Batch processing** | Add `--output` flag to save review image instead of displaying |
| **Template fallback** | If template.jpg is missing, extract key boxes from the input image directly |

---

## 7. Git Metadata

```bash
# Remote
origin  https://github.com/DojjAA/QMY_digit_symbol_coding.git

# Current state
main branch, all files committed
```

