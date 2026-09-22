# Heading-Tree Validation Calibration Report

## Overview

The Guaipeca converter uses a font-size histogram to detect headings in PDFs.
When a PDF has closely-spaced font sizes (common in Oracle FCCS manuals),
this approach can produce a bogus heading tree — all text maps to one level,
or sizes are so close they're indistinguishable.

This document records the **real calibration measurements** used to set
validation thresholds for the four signals in `_validate_heading_tree`.

**Re-calibrated 2026-09-22** against a real Oracle FCCS PDF (1,349 pages).
The previous calibration (v0.3.0) used only synthetic fixtures and was
proven to false-positive on a healthy real document.

## Documents Measured

| Document | Type | Pages | Description |
|----------|------|-------|-------------|
| FCCS PDF | **REAL Oracle FCCS** | 1,349 | Oracle Fusion Cloud EPM FCCS Admin Guide (42 MB) |
| `tests/fixtures/test-doc.pdf` | **REAL Oracle manual** (excerpt) | 5 | Oracle RAG whitepaper excerpt |
| `tests/fixtures/deep-legit.pdf` | Synthetic | 3 | Good 5-level hierarchy |
| `tests/fixtures/good-hierarchy.pdf` | Synthetic | 5 | Good 3-level hierarchy |
| `tests/fixtures/oracle-manual-sim.pdf` | Synthetic | 40 | Simulates FCCS (6 closely-spaced sizes) |
| `tests/fixtures/oracle-template-sim.pdf` | Synthetic | 20 | Simulates single-size template |

The FCCS PDF was downloaded from Oracle's documentation site and measured
in full. It is the primary calibration reference. Synthetic fixtures are
used only as contrast data and are labelled as such throughout.

## Measurements

### Per-document signal values

| Document | Body (pt) | Heading sizes (pt) | Levels | Total H | Collapse | Saturation | Min Gap (pt) | First H | Valid? |
|----------|-----------|-------------------|--------|---------|----------|------------|--------------|---------|--------|
| **FCCS PDF (real)** | 10.0 | 35,30,24,21,18,16 | 6 | 1138 | 0.030 | 0.149 | 2.0 | H1 | ✅ YES |
| **test-doc.pdf (real)** | 11.0 | 24,16 | 2 | 8 | 0.167 | 0.000 | 8.0 | H1 | ✅ YES |
| deep-legit.pdf (synth) | 10.0 | 24,20,16,14,12.5 | 5 | 45 | 0.167 | 0.000 | 1.5 | H1 | ✅ YES |
| good-hierarchy.pdf (synth) | 11.0 | 24,18,14 | 3 | 15 | 0.200 | 0.000 | 4.0 | H1 | ✅ YES |
| oracle-manual-sim (synth) | 10.0 | 18,16,15,14.5,14,13 | 6 | 200 | 0.025 | 0.200 | 0.5 | **H6** | ❌ NO |
| oracle-template-sim (synth) | 11.0 | 14 | 1 | 10 | 1.000 | 0.000 | N/A | H1 | ❌ NO |

### Which signals fire on which documents

| Document | Collapse | Saturation | Size Clustering | First Heading | Result |
|----------|----------|------------|-----------------|---------------|--------|
| **FCCS PDF (real)** | — | — | — | — | PASS |
| **test-doc.pdf (real)** | — | — | — | — | PASS |
| deep-legit.pdf (synth) | — | — | — | — | PASS |
| good-hierarchy.pdf (synth) | — | — | — | — | PASS |
| oracle-manual-sim (synth) | — | ✅ (0.200 > 0.20*) | ✅ (0.5pt ≤ 1.0pt) | ✅ (H6 first) | **FAIL** |
| oracle-template-sim (synth) | ✅ (1.0 > 0.5) | — | — | — | **FAIL** |

\* oracle-manual-sim saturation = 0.200, threshold = 0.20. The `>` operator
means 0.200 is NOT strictly greater than 0.20, so this signal technically
does not fire. However, the size-clustering and first-heading signals do
fire, so the document is still correctly rejected. The saturation threshold
is set at 0.20 to give the real FCCS doc (0.149) a 0.051 margin.

## Thresholds and Rationale

### Signal 1: Collapse Ratio — threshold > 0.50

**What it measures:** Two sub-checks:
- (a) No H2 headings at all → collapse = 1.0 (everything at one level)
- (b) ≥3 heading levels but >50% of H2 sections have no H3+ children

**Calibration:**
- FCCS PDF (real): 0.030 — 32/33 H2 sections have H3+ children → PASS
- test-doc.pdf (real): 0.167 — 2 levels, exempt from sub-check (b) → PASS
- oracle-template-sim: 1.000 — no H2 headings → FAIL
- deep-legit.pdf: 0.167 — 2 levels exempt → PASS
- good-hierarchy.pdf: 0.200 — 3 levels, 4/5 H2s have children → PASS

**Margin:** Real FCCS at 0.030 has 0.470 margin below the 0.50 threshold.

**Limitation:** The collapse ratio sub-check (b) is a heuristic. It assumes
documents with ≥3 heading levels should have H3 subsections under most H2
headings. The original reported failure mode (Ch.21 collapse, see below)
could not be reproduced on the real FCCS PDF, so this signal's ability to
detect that specific failure is **unverified** on real data.

### Signal 2: Depth Saturation — threshold > 0.20

**What it measures:** Fraction of all headings at the maximum level (H6).

**Calibration:**
- **FCCS PDF (real): 169/1138 = 0.149** → PASSES with 0.051 margin
- test-doc.pdf (real): 0/8 = 0.000 → PASSES
- oracle-manual-sim: 40/200 = 0.200 → at boundary (caught by other signals)
- All other docs: 0.000 → PASSES

**Why 0.20 and not the old 0.15:** The old threshold of 0.15 was a knife-edge
— the real FCCS PDF measured 0.149, just 0.001 below the cut. That is not a
robust classifier. The new threshold of 0.20 gives the real document a 0.051
margin (5.1 percentage points) while still rejecting the synthetic bad
fixture via other signals. Genuinely broken trees (all headings crammed at
max depth) would exceed 0.50+.

**Margin:** 0.051 (real FCCS at 0.149 vs threshold at 0.20).

### Signal 3: Size Clustering — threshold ≤ 1.0pt

**What it measures:** Minimum gap between adjacent heading font sizes.

**Calibration:**
- FCCS PDF (real): 2.0pt min gap → PASSES (margin: 1.0pt)
- test-doc.pdf (real): 8.0pt → PASSES
- oracle-manual-sim: 0.5pt → FAILS
- deep-legit.pdf: 1.5pt → PASSES
- good-hierarchy.pdf: 4.0pt → PASSES

**Margin:** Real FCCS at 2.0pt has 1.0pt margin above the 1.0pt threshold.

This is the most reliable signal — directly tests whether the histogram
approach can distinguish heading levels.

### Signal 4: First-Heading Sanity — first heading > H3 = fail

**What it measures:** Whether the very first heading in the document is
deeper than H3, which would indicate a mis-mapped size histogram.

**Calibration:**
- FCCS PDF (real): first heading = H1 → PASSES
- test-doc.pdf (real): first heading = H1 → PASSES
- oracle-manual-sim: first heading = H6 → FAILS
- All other docs: first heading = H1 → PASS

**Why the old "order anomaly" was removed:** The previous signal flagged
ANY deeper heading appearing before a shallower one in document order
(e.g. "H5 appears before H4"). This was fundamentally broken: real Oracle
manuals have a Table of Contents where chapter numbers (1-26) are rendered
at 18pt, which maps to H5. These TOC entries appear on pages 3-19, before
the first H4 body heading on page 48. This is completely legitimate
document structure, not an anomaly. The old signal false-positived on the
real 1,349-page FCCS PDF, causing it to be rejected and fall back to
markitdown unnecessarily.

**What remains:** Only the first heading is checked. If it's deeper than H3,
the size histogram is likely mis-mapped (e.g. body text was mistaken for
headings). This catches oracle-manual-sim (first heading = H6) while
passing all real and synthetic good documents.

**Level jumps are NOT flagged:** H1 → H6 (jump of 5) can occur legitimately
when a document has front-matter or TOC sections at different font sizes.
The old signal flagged these; the redesigned one does not.

## Chapter 21 Collapse Reproduction

### The Reported Failure Mode

The original defect report stated: all of Chapter 21 collapsing under
"Seeded Consolidation Rules - Example (March)" (level-5, 157 chunks).

### Reproduction Attempt

The real FCCS PDF was searched for headings containing "Consolidat",
"Seeded", and chapter-21 references. The heading "Seeded Consolidation
Rules - Example (March)" was found at:

- **H5, page 691** (font size 18.0pt)

The surrounding heading structure shows NO collapse:

```
H5 p680: Seeded Consolidation Rule Examples
H5 p681: Seeded Consolidation Rules - Example (January)
H5 p683: Seeded Consolidation Rules - Example (February)
H5 p691: Seeded Consolidation Rules - Example (March)
H2 p700: 22            ← Chapter 22 begins
H3 p700: Working with Rules
H4 p700: Consolidation and Translation Rules
```

The March example is one of several H5 subsections under the H4 parent
"Seeded Consolidation Rules" (page 666). It is followed by a normal
chapter boundary (H2 "22" on page 700). There is no evidence of heading
collapse — the hierarchy is well-structured with appropriate nesting.

### Conclusion

**The original Ch.21 collapse does NOT reproduce on this document.**

The heading tree around the reported failure point is well-structured:
H4 → H5 (January, February, March examples) → H2 (next chapter). The
"157 chunks under one node" may have been an artifact of a different
PDF version, a different converter version, or a chunking issue rather
than a heading-detection issue.

**The guard's target failure mode is UNCONFIRMED on our only real sample.**
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

1. **Only one real FCCS PDF measured.** While the 1,349-page FCCS Admin
   Guide is a strong reference, other Oracle manuals may have different
   font-size distributions. The thresholds have reasonable margins but
   more real documents would increase confidence.

2. **Ch.21 collapse unconfirmed.** The collapse-ratio signal guards
   against a failure mode that could not be reproduced on real data.
   It may be a false alarm from an earlier converter version.

3. **Depth-saturation threshold is at the boundary for the synthetic
   bad fixture.** oracle-manual-sim measures exactly 0.200 against a
   0.20 threshold (using `>`). The signal does not fire on it, but the
   document is still rejected by size-clustering and first-heading
   signals. If a real document measures between 0.15 and 0.20 at H6,
   it would now pass where the old threshold would have rejected it.
   The 0.051 margin on the real FCCS doc is the safety buffer.

4. **The first-heading check is narrow.** It only looks at the very
   first heading. A mis-mapped histogram that happens to have a
   plausible first heading but garbled subsequent levels would not be
   caught by this signal alone (but would likely be caught by
   size-clustering or collapse-ratio).