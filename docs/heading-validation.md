# Heading-Tree Validation Calibration Report

## Overview

The Guaipeca converter uses a font-size histogram to detect headings in PDFs.
When a PDF has closely-spaced font sizes (common in Oracle FCCS manuals),
this approach can produce a bogus heading tree — all text maps to one level,
or sizes are so close they're indistinguishable.

This document records the **real calibration measurements** used to set
validation thresholds for the four signals in `_validate_heading_tree`.

**Re-calibrated 2026-09-22** against **4 real Oracle FCCS PDFs** (ranging from
233 to 1,349 pages) plus a real Oracle manual excerpt in the test fixtures.
The previous calibration (v0.3.0) used only synthetic fixtures and was
proven to false-positive on healthy real documents. This round expands to
multiple real samples to ensure thresholds are robust, not tuned to a single
document.

## Documents Measured

| Document | Type | Pages | Description |
|----------|------|-------|-------------|
| FCCS Info Dev Team | **REAL Oracle FCCS** | 1,349 | EPM Information Development Team Admin Guide (42 MB) |
| FCCS DIEPM Guide | **REAL Oracle FCCS** | 871 | Data Integration EPM Admin Guide (31 MB) |
| FCCS eCalc Guide | **REAL Oracle FCCS** | 354 | Enterprise Calculation Manager Design Guide (4.5 MB) |
| FCCS FR Web Guide | **REAL Oracle FCCS** | 233 | Financial Reporting Web Studio Design Guide (5.3 MB) |
| `tests/fixtures/test-doc.pdf` | **REAL Oracle manual** (excerpt) | 5 | Oracle RAG whitepaper excerpt |
| `tests/fixtures/deep-legit.pdf` | Synthetic | 3 | Good 5-level hierarchy |
| `tests/fixtures/good-hierarchy.pdf` | Synthetic | 5 | Good 3-level hierarchy |
| `tests/fixtures/oracle-manual-sim.pdf` | Synthetic | 40 | Simulates FCCS (6 closely-spaced sizes) |
| `tests/fixtures/oracle-template-sim.pdf` | Synthetic | 20 | Simulates single-size template |

All 4 real FCCS PDFs were downloaded from Oracle's documentation site and
measured in full. They share an identical font structure: body=10.0pt,
heading sizes=[35,30,24,21,18,16], min gap=2.0pt. This is consistent across
the Oracle FCCS documentation family. Synthetic fixtures are used only as
contrast data and are labelled as such throughout.

## Measurements

### Per-document signal values

| Document | Body (pt) | Heading sizes (pt) | Levels | Total H | Collapse | Saturation | Min Gap (pt) | First H | Valid? |
|----------|-----------|-------------------|--------|---------|----------|------------|--------------|---------|--------|
| **FCCS Info Dev (real)** | 10.0 | 35,30,24,21,18,16 | 6 | 1138 | 0.030 | 0.149 | 2.0 | H1 | ✅ YES |
| **FCCS DIEPM (real)** | 10.0 | 35,30,24,21,18,16 | 6 | 714 | 0.036 | 0.239 | 2.0 | H1 | ✅ YES |
| **FCCS eCalc (real)** | 10.0 | 35,30,24,21,18,16 | 6 | 662 | 0.071 | 0.459 | 2.0 | H1 | ✅ YES |
| **FCCS FR Web (real)** | 10.0 | 35,30,24,21,18,16 | 6 | 315 | 0.167 | 0.140 | 2.0 | H1 | ✅ YES |
| **test-doc.pdf (real)** | 11.0 | 24,16 | 2 | 8 | 0.167 | 0.000 | 8.0 | H1 | ✅ YES |
| deep-legit.pdf (synth) | 10.0 | 24,20,16,14,12.5 | 5 | 45 | 0.167 | 0.000 | 1.5 | H1 | ✅ YES |
| good-hierarchy.pdf (synth) | 11.0 | 24,18,14 | 3 | 15 | 0.200 | 0.000 | 4.0 | H1 | ✅ YES |
| oracle-manual-sim (synth) | 10.0 | 18,16,15,14.5,14,13 | 6 | 200 | 0.025 | 0.200 | 0.5 | **H6** | ❌ NO |
| oracle-template-sim (synth) | 11.0 | 14 | 1 | 10 | 1.000 | 0.000 | N/A | H1 | ❌ NO |

### Which signals fire on which documents

| Document | Collapse | Saturation | Size Clustering | First Heading | Result |
|----------|----------|------------|-----------------|---------------|--------|
| **FCCS Info Dev (real)** | — | — | — | — | PASS |
| **FCCS DIEPM (real)** | — | — | — | — | PASS |
| **FCCS eCalc (real)** | — | — | — | — | PASS |
| **FCCS FR Web (real)** | — | — | — | — | PASS |
| **test-doc.pdf (real)** | — | — | — | — | PASS |
| deep-legit.pdf (synth) | — | — | — | — | PASS |
| good-hierarchy.pdf (synth) | — | — | — | — | PASS |
| oracle-manual-sim (synth) | — | — | ✅ (0.5pt ≤ 1.0pt) | ✅ (H6 first) | **FAIL** |
| oracle-template-sim (synth) | ✅ (1.0 > 0.5) | — | — | — | **FAIL** |

All 5 real Oracle documents PASS with zero signals firing. The 2 synthetic
bad fixtures are still correctly rejected by other signals.

## Thresholds and Rationale

### Signal 1: Collapse Ratio — threshold > 0.50

**What it measures:** Two sub-checks:
- (a) No H2 headings at all → collapse = 1.0 (everything at one level)
- (b) ≥3 heading levels but >50% of H2 sections have no H3+ children

**Calibration across all real docs:**
- FCCS Info Dev: 0.030 — 32/33 H2s have H3+ children → PASS
- FCCS DIEPM: 0.036 → PASS
- FCCS eCalc: 0.071 → PASS
- FCCS FR Web: 0.167 → PASS
- test-doc.pdf: 0.167 — 2 levels, exempt from sub-check (b) → PASS
- oracle-template-sim: 1.000 — no H2 headings → FAIL

**Margin:** Tightest real doc (FCCS FR Web) at 0.167 has 0.333 margin below 0.50.

**Limitation:** The collapse-ratio sub-check (b) is a heuristic. The original
reported failure mode (Ch.21 collapse, see below) could not be reproduced
on any of the 4 real FCCS PDFs, so this signal's ability to detect that
specific failure is **unverified** on real data.

### Signal 2: Depth Saturation — threshold > 0.60

**What it measures:** Fraction of all headings at the maximum level (H6).

**Calibration across all real docs:**
- FCCS eCalc: 304/662 = **0.459** (deepest real doc) → PASSES with 0.141 margin
- FCCS DIEPM: 171/714 = 0.239 → PASSES with 0.361 margin
- FCCS Info Dev: 169/1138 = 0.149 → PASSES with 0.451 margin
- FCCS FR Web: 44/315 = 0.140 → PASSES with 0.460 margin
- test-doc.pdf: 0/8 = 0.000 → PASSES
- oracle-manual-sim: 40/200 = 0.200 → below threshold, but caught by other signals

**Threshold history:**
- v0.3.0: 0.15 — false-positived on FCCS Info Dev (0.149, off by 0.001)
- v0.3.1 first pass: 0.20 — false-positived on FCCS DIEPM (0.239) and FCCS eCalc (0.459)
- v0.3.1 final: **0.60** — all 4 real docs pass, synthetic bad fixture still caught

**Why 0.60:** Real Oracle FCCS manuals legitimately use all 6 heading levels,
with the deepest doc (eCalc) having 46% at H6. This is normal for a
calculation design guide with many detailed subsections. The 0.60 threshold
catches only extreme cases (60%+ at max depth) where the size histogram is
clearly merging too many sizes into one bin. The primary discriminator for
the synthetic bad fixture is **size-clustering** (0.5pt gaps), not
depth-saturation.

**Margin:** 0.141 (FCCS eCalc at 0.459 vs threshold at 0.60).

### Signal 3: Size Clustering — threshold ≤ 1.0pt

**What it measures:** Minimum gap between adjacent heading font sizes.

**Calibration across all real docs:**
- All 4 FCCS PDFs: 2.0pt min gap → PASSES (margin: 1.0pt)
- test-doc.pdf: 8.0pt → PASSES
- oracle-manual-sim: 0.5pt → FAILS
- deep-legit.pdf: 1.5pt → PASSES
- good-hierarchy.pdf: 4.0pt → PASSES

**Margin:** All real FCCS docs at 2.0pt have 1.0pt margin above the 1.0pt threshold.

This is the most reliable signal — directly tests whether the histogram
approach can distinguish heading levels. All 4 real Oracle FCCS PDFs share
identical font structure with 2.0pt minimum gaps, well above the 1.0pt
clustering threshold.

### Signal 4: First-Heading Sanity — first heading > H3 = fail

**What it measures:** Whether the very first heading in the document is
deeper than H3, which would indicate a mis-mapped size histogram.

**Calibration across all real docs:**
- All 4 FCCS PDFs: first heading = H1 → PASSES
- test-doc.pdf: first heading = H1 → PASSES
- oracle-manual-sim: first heading = H6 → FAILS
- All other docs: first heading = H1 → PASS

**Why the old "order anomaly" was removed:** The previous signal flagged
ANY deeper heading appearing before a shallower one in document order
(e.g. "H5 appears before H4"). This was fundamentally broken: all 4 real
Oracle FCCS manuals have a Table of Contents where chapter numbers (1-26)
are rendered at 18pt, which maps to H5. These TOC entries appear on pages
3-19, before the first H4 body heading on page 48. This is completely
legitimate document structure, not an anomaly. The old signal
false-positived on all real FCCS PDFs.

**What remains:** Only the first heading is checked. If it's deeper than H3,
the size histogram is likely mis-mapped. This catches oracle-manual-sim
(first heading = H6) while passing all real and synthetic good documents.

**Level jumps are NOT flagged:** H1 → H6 (jump of 5) can occur legitimately
when a document has front-matter or TOC sections at different font sizes.

## Chapter 21 Collapse Reproduction

### The Reported Failure Mode

The original defect report stated: all of Chapter 21 collapsing under
"Seeded Consolidation Rules - Example (March)" (level-5, 157 chunks).

### Reproduction Attempt

All 4 real FCCS PDFs were searched for headings containing "Consolidat",
"Seeded", and chapter-21 references.

**FCCS Info Dev Team (1,349p):** "Seeded Consolidation Rules - Example (March)"
found at H5, page 691 (18.0pt). Surrounding structure:

```
H4 p666: Seeded Consolidation Rules
H5 p680: Seeded Consolidation Rule Examples
H5 p681: Seeded Consolidation Rules - Example (January)
H5 p683: Seeded Consolidation Rules - Example (February)
H5 p691: Seeded Consolidation Rules - Example (March)
H2 p700: 22            ← Chapter 22 begins
H3 p700: Working with Rules
H4 p700: Consolidation and Translation Rules
```

No collapse. The heading tree is well-structured with appropriate H4→H5
nesting and a clean chapter boundary after.

**FCCS DIEPM Guide (871p):** No "Seeded Consolidation" headings found.
Chapter 21 marker at H2 p811 (normal chapter boundary).

**FCCS eCalc Guide (354p):** No "Seeded" or "Consolidation Example"
headings found.

**FCCS FR Web Guide (233p):** No "Seeded" or "Consolidation Example"
headings found.

### Conclusion

**The original Ch.21 collapse does NOT reproduce on ANY of the 4 real FCCS PDFs.**

The heading tree around the reported failure point (in the one document that
contains it) is well-structured: H4 → H5 (January, February, March examples)
→ H2 (next chapter). The "157 chunks under one node" may have been an
artifact of a different PDF version, a different converter version, or a
chunking issue rather than a heading-detection issue.

**The guard's target failure mode is UNCONFIRMED on all 4 real samples.**
The collapse-ratio signal (Signal 1) may be guarding against a failure
that does not occur in practice with the current converter. Evidence that
would confirm it:
1. A real PDF where the converter emits a flat tree (all headings at one
   level) when the document clearly has a multi-level hierarchy.
2. A real PDF where H2 sections have no H3+ children despite the document
   having ≥3 visual heading levels.

Without such evidence, the collapse-ratio signal remains a heuristic
precaution, not a proven detector.

## Validator Behavior

When ANY signal fires, the converter logs a warning and falls back to
markitdown for that document. This ensures degraded heading detection
doesn't produce broken chunk trees — the document is still indexed,
just with markitdown's simpler heading structure.

## Residual Risks

1. **4 real FCCS PDFs measured, all from the Oracle FCCS family.** They
   share identical font structure (body=10pt, headings=[35,30,24,21,18,16]).
   Other Oracle product families (HCM, SCM, etc.) or non-Oracle manuals may
   have different distributions. The thresholds have reasonable margins but
   more diverse real documents would increase confidence.

2. **Ch.21 collapse unconfirmed across all 4 samples.** The collapse-ratio
   signal guards against a failure mode that could not be reproduced on any
   real data. It may be a false alarm from an earlier converter version.

3. **Depth-saturation signal is now secondary.** At 0.60, it only catches
   extreme cases (60%+ at H6). The primary discriminator for broken
   histograms is size-clustering (1.0pt gap threshold). If a real document
   has 7+ heading sizes crammed into 6 bins with gaps just above 1.0pt,
   neither signal would fire. This is an accepted limitation — the
   converter would produce a slightly imperfect but still usable heading
   tree.

4. **The first-heading check is narrow.** It only looks at the very first
   heading. A mis-mapped histogram that happens to have a plausible first
   heading but garbled subsequent levels would not be caught by this signal
   alone (but would likely be caught by size-clustering or collapse-ratio).

5. **Threshold evolution.** The depth-saturation threshold went through
   three iterations (0.15 → 0.20 → 0.60) as more real data arrived. This
   trajectory suggests the signal is less discriminative than originally
   assumed. If a future real document has saturation > 0.60, the threshold
   may need further adjustment or the signal may need to be removed
   entirely in favor of a combined check (e.g. saturation + clustering).
## Heading Line-Wrap Stitching (v0.3.2)

### The Defect

Oracle FCCS PDFs wrap long titles across 2+ physical lines. The converter's
Phase 2 emitted each *physical* pymupdf line independently, producing:

1. **Truncated headings** — the heading text ends mid-phrase (e.g.
   `#### Configuring Detailed Analysis in Financial Consolidation and`)
2. **Stray continuation fragments** — the wrapped remainder (e.g. `Close`)
   emitted as its own bogus heading or orphaned into body text

A truncated heading produces a wrong `heading_path` / `section_id`, which
poisons `section_mode`, `section_filter`, and `get_toc` in chunking.py.

### Root Cause

In `_convert_pdf_with_headings`, Phase 2 iterated each pymupdf *line*
within a block independently:

```python
for line in block.get("lines", []):
    if max_size in size_to_level:   # emit as heading
        ...
```

There was no notion of a logical heading spanning multiple physical lines.

### Investigation: Block Structure vs Heuristics

The fix direction suggested size+punctuation heuristics (conjunction
endings, lowercase continuations). Before implementing, we investigated
whether pymupdf's block structure provides a more reliable signal.

**Finding:** pymupdf's `page.get_text("dict")` groups the lines of a
wrapped heading into a **single block** with multiple *line* entries, all
at the same heading font size. This was verified on the real FCCS Info
Dev PDF (1,349 pages):

```
Block 10 (bbox=(134.7, 204.5, 556.8, 227.7)):
  [size=21.0] 'Configuring Detailed Analysis in Financial Consolidation and'
  [size=21.0] 'Close'
```

**Measurements on the 1,349-page FCCS Info Dev PDF:**

| Metric                                        | Count |
|-----------------------------------------------|-------|
| Single-line heading blocks                    | 1,017 |
| Multi-line heading blocks (2+ heading lines)  |    57 |
| Blocks mixing heading-sized + body-sized lines |     0 |
| Cross-block heading-size adjacency (200 pages)|     1 |

The single cross-block case was TOC chapter numbers ("1" and "2" in
separate blocks), not a wrapped heading.

**Conclusion:** Block boundaries are a structural signal from the PDF
layout engine, far more reliable than size+punctuation heuristics. We
chose block-based stitching: within each block, consecutive heading-sized
lines at the same font size are joined into one logical heading.

### The Fix

In Phase 2, instead of iterating lines independently, we now:
1. Collect heading-sized lines within a block into an accumulator
2. When a body-sized line or a different heading size is encountered,
   flush the accumulator as a single stitched heading
3. At the end of each block, flush any remaining accumulator

The `_emit_heading` helper joins parts with a single space, maps the size
to a heading level, and appends the markdown marker. Duplicate consecutive
headings are still suppressed.

### Before/After Evidence

Measured on 4 real Oracle FCCS PDFs:

| Document | Pages | BEFORE: Truncated | BEFORE: Stray | AFTER: Truncated | AFTER: Stray |
|----------|-------|-------------------|---------------|-------------------|--------------|
| FCCS Info Dev | 1,349 | 7 | 4 | 0 | 0 |
| FCCS DIEPM    |   871 | 5 | 1 | 0 | 0 |
| FCCS eCalc    |   354 | 6 | 1 | 0 | 0 |
| FCCS FR Web   |   233 | 3 | 1 | 0 | 0 |
| **Total**     | 2,807 | **21** | **7** | **0** | **0** |

All 21 truncated headings are now complete. All 7 stray continuation
fragments are absorbed into their parent heading.

### Over-Stitching Risk Analysis

**Risk: Could the stitch merge a heading with following body text?**

No. The stitch only joins heading-sized lines *within the same block*.
Body text is always in a separate block (0 mixed blocks measured in
1,349 pages). When a body-sized line is encountered within a block, the
heading accumulator is flushed before the body line is emitted.

**Risk: Could the stitch merge two distinct headings?**

Only if two distinct heading-sized lines at the *same* font size appear
in the same block. This would be a PDF layout error, not a normal
document structure. The 57 multi-line heading blocks measured all
contained wrapped continuations of a single logical heading.

**Risk: Could cross-block heading adjacency cause issues?**

No. Stitching is confined to lines within a single block. The 1
observed cross-block case (TOC numbers) was correctly NOT stitched.

**Tested boundaries:**
- `test_wrapped_heading_not_merged_with_following_body`: verifies body
  text is not absorbed
- `test_standalone_heading_not_merged_with_body`: verifies a standalone
  heading remains standalone
- `test_test_doc_pdf_converts_correctly`: verifies the existing fixture
  still produces correct headings

### Regression Test

A synthetic fixture (`tests/fixtures/wrapped-heading.pdf`) reproduces
the multi-line heading pattern: two 18pt lines in one pymupdf block
+ body text in separate blocks + a standalone heading to verify no
over-stitching. 8 tests in `tests/test_heading_stitching.py` cover
stitching correctness, no truncation, no stray fragments, no
over-stitching, and existing behaviour preservation.
