# Heading-Tree Validation Calibration Report

## Overview

The Guaipeca converter uses a font-size histogram to detect headings in PDFs.
When a PDF has closely-spaced font sizes (common in Oracle FCCS manuals),
this approach can produce a bogus heading tree — all text maps to one level,
or sizes are so close they're indistinguishable.

This document records the **real calibration measurements** used to set
validation thresholds for the four signals in `_validate_heading_tree`.

## Documents Measured

| Document | Type | Pages | Description |
|----------|------|-------|-------------|
| `tests/fixtures/test-doc.pdf` | **REAL Oracle manual** | ~350 | Oracle RAG whitepaper |
| `tests/fixtures/deep-legit.pdf` | Synthetic | — | Good 5-level hierarchy |
| `tests/fixtures/good-hierarchy.pdf` | Synthetic | — | Good 3-level hierarchy |
| `tests/fixtures/oracle-manual-sim.pdf` | Synthetic | — | Simulates FCCS (6 closely-spaced sizes) |
| `tests/fixtures/oracle-template-sim.pdf` | Synthetic | — | Simulates single-size template |

**⚠️ FCCS PDFs NOT available.** The user's guaipeca config (`~/.guaipeca/guaipeca.yaml`)
defines corpora at `/home/luis/.guaipeca/corpora/{lily,tess,sam}` and
`/home/luis/.openclaw/workspace/docs` — none contain Oracle FCCS PDFs.
The repo's template config (`config/guaipeca.yaml`) has placeholder paths
(`/path/to/your/docs`). Calibration was performed on `test-doc.pdf` (a real
Oracle manual, though not FCCS) and the synthetic fixtures listed above.
**Thresholds marked PROVISIONAL need validation against real FCCS PDFs
when they become available.**

## Measurements

### Per-document signal values

| Document | Body (pt) | Heading sizes (pt) | Levels | Total H | Collapse | Saturation | Min Gap (pt) | Order Anomaly | Valid? |
|----------|-----------|-------------------|--------|---------|----------|------------|--------------|---------------|--------|
| test-doc.pdf | 11.0 | 24.0, 16.0 | 2 | 8 | 0.00* | 0.00 | 8.0 | False | ✅ YES |
| deep-legit.pdf | 10.0 | 24.0, 20.0, 16.0, 14.0, 12.5 | 5 | 45 | 0.00* | 0.00 | 1.5 | False | ✅ YES |
| good-hierarchy.pdf | 11.0 | 24.0, 18.0, 14.0 | 3 | 15 | 0.00* | 0.00 | 4.0 | False | ✅ YES |
| oracle-manual-sim.pdf | 10.0 | 18.0, 16.0, 15.0, 14.5, 14.0, 13.0 | 6 | 200 | 0.00* | 0.20 | 0.5 | **True** (H6 first) | ❌ NO |
| oracle-template-sim.pdf | 11.0 | 14.0 | 1 | 10 | 1.00** | 0.00 | N/A | False | ❌ NO |

\* Collapse = 0.00 because these docs have ≤2 heading levels (flat-but-correct).
   The collapse signal only fires when ≥3 levels exist AND >50% of H2 sections
   lack H3+ children, OR when no H2 headings exist at all.

\*\* Collapse = 1.00 because all headings are H1 (no H2 → collapse = 1.0 by rule).

### Which signals fire on which documents

| Document | Collapse | Saturation | Size Clustering | Order Anomaly | Result |
|----------|----------|------------|-----------------|---------------|--------|
| test-doc.pdf | — | — | — | — | PASS |
| deep-legit.pdf | — | — | — | — | PASS |
| good-hierarchy.pdf | — | — | — | — | PASS |
| oracle-manual-sim.pdf | — | ✅ (0.20 > 0.15) | ✅ (0.5pt ≤ 1.0pt) | ✅ (H6 first) | **FAIL** |
| oracle-template-sim.pdf | ✅ (1.0 > 0.5) | — | — | — | **FAIL** |

## Thresholds and Rationale

### Signal 1: Collapse Ratio — threshold > 0.50

**What it measures:** Two sub-checks:
- (a) No H2 headings at all → collapse = 1.0 (everything at one level)
- (b) ≥3 heading levels but >50% of H2 sections have no H3+ children

**Why this proxy:** The original Ch.21 bug was heading paths collapsing
under one deep node — a flat tree where hierarchy should exist. Sub-check (a)
catches single-level documents (oracle-template-sim). Sub-check (b) catches
the case where the converter emits multiple levels but H2 sections don't
actually nest into H3 subsections.

**Calibration:**
- test-doc.pdf: 2 levels, no H3 → collapse = 0.00 (flat but correct, ≤2 levels exempt)
- deep-legit.pdf: 5 levels, all H2 have H3 children → collapse = 0.00
- oracle-template-sim.pdf: 1 level → collapse = 1.00 → FIRES

**PROVISIONAL:** No real FCCS PDF measured. Threshold may need tuning.

### Signal 2: Depth Saturation — threshold > 0.15

**What it measures:** Fraction of all headings at the maximum level (H6).

**Calibration:**
- oracle-manual-sim.pdf: 40/200 = 0.20 → FIRES
- All good docs: 0.00 → does not fire

**PROVISIONAL:** Only one "bad" data point. Real FCCS measurement needed.

### Signal 3: Size Clustering — threshold ≤ 1.0pt

**What it measures:** Minimum gap between adjacent heading font sizes.

**Calibration:**
- oracle-manual-sim.pdf: 0.5pt gaps → FIRES
- deep-legit.pdf: 1.5pt min gap → does not fire
- test-doc.pdf: 8.0pt gap → does not fire
- good-hierarchy.pdf: 4.0pt gap → does not fire

This is the most reliable signal — directly tests whether the histogram
approach can distinguish heading levels.

### Signal 4: Order Anomaly — boolean (any = fail)

**What it measures:** First heading deeper than H2, or level jumps > 1.

**Calibration:**
- oracle-manual-sim.pdf: H6 as first heading → FIRES
- All good docs: start with H1 → does not fire

## Validator Behavior

When ANY signal fires, the converter logs a warning and falls back to
markitdown for that document. This ensures degraded heading detection
doesn't produce broken chunk trees — the document is still indexed,
just with markitdown's simpler heading structure.

## Limitations

1. **No real FCCS PDFs were measured.** The thresholds are calibrated on
   one real Oracle manual (test-doc.pdf, which has a clean hierarchy) and
   synthetic simulations. Real FCCS manuals may have different font-size
   distributions that require threshold adjustments.

2. **The collapse ratio sub-check (b) is a heuristic.** It assumes that
   documents with ≥3 heading levels should have H3 subsections under most
   H2 headings. Some legitimate documents might have a few H2 sections
   with no subsections (e.g., short introductory sections).

3. **Size clustering is the strongest signal.** If two heading sizes are
   within 0.5pt, the histogram approach fundamentally cannot distinguish
   them. This signal would benefit most from real FCCS data.