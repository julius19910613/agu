# Nine MOV Scoreboard And Identity Validation

Date: 2026-07-13

## Scope

Validated the nine `/Users/ppt/Movies/2026-07-04 *.mov` recordings in filename-time order. Runtime JSON, crops, and contact sheets remain under the ignored `analysis_outputs/nine_mov_validation_20260712/` directory; raw MOV files remain outside the repository.

The scoreboard calibration reuses the first two investigations: full-panel candidate ranking, consecutive-frame LED phase bursts, sharp/boundary variants, percentile temporal fusion, OCR/VLM consensus, and monotonic score reconciliation. Identity validation uses two accurate 30-second segments with six seconds of overlap per MOV, then checks SFace, jersey darkness, and overlap-track continuity evidence.

## Final Score Gate

| Recording | Final score | Calibration evidence |
| --- | ---: | --- |
| `150151` | 8-9 | Consecutive readable anchors at 787.0s and 798.8s |
| `151732` | 21-19 | Consecutive readable anchors at 827.4s and 839.3s |
| `153430` | 31-26 | Readable late anchor at 709.0s; consistent earlier late-game reads |
| `154911` | 38-36 | Readable LED burst at 819.3s |
| `160824` | 69-52 | Readable late anchor at 695.1s; rejected implausible 90-90 / 95-92 reads |
| `162919` | 71-65 | Consecutive-frame terminal burst at 463.3s; rejected truncated 11-65 read |
| `163904` | 99-80 | Burst consensus at 693.5s; rejected later 99-192 hallucination |
| `165230` | 108-90 | Three-read burst consensus at 348.1s; rejected truncated 10-90 reads |
| `165921` | 120-96 | Cross-anchor consensus at 135.1s and 151.2s |

Result: 9/9 calibrated final scores match Codex multi-frame visual review.

The final current-code authority is the ignored runtime artifact
`analysis_outputs/nine_mov_validation_20260712/current_final_scoreboard_gate.jsonl`.
It contains exactly nine rows and `all(.match) == true`; this replaces mixed
intermediate JSONL files that intentionally retain earlier failed calibration runs.

## Face And Jersey Identity Gate

Each row covers 36 segment-local tracks (18 per accurate segment). `SFace tracks` counts tracks with a quality-gated YuNet/SFace embedding; `SFace samples` counts accepted face samples. Cross-segment merges are intentionally conservative.

| Recording | SFace tracks | SFace samples | Confirmed cross-segment identities | Strongest joint evidence |
| --- | ---: | ---: | ---: | --- |
| `150151` | 5 | 12 | 3 | Overlap continuity plus jersey darkness gap 0.07-0.09 |
| `151732` | 7 | 20 | 1 | SFace 0.89; darkness gap 0.00; confidence 0.86 |
| `153430` | 10 | 21 | 1 | SFace 0.79; darkness gap 0.01; confidence 0.78 |
| `154911` | 10 | 25 | 1 | SFace 0.79; darkness gap 0.02; confidence 0.72 |
| `160824` | 12 | 29 | 1 | SFace 0.66; darkness gap 0.08; confidence 0.64 |
| `162919` | 14 | 31 | 2 | SFace 0.74/0.70; darkness gap 0.12/0.02 |
| `163904` | 9 | 26 | 2 | SFace 0.88/0.68; darkness gap 0.06/0.01 |
| `165230` | 11 | 26 | 3 | SFace 0.86/0.82 plus overlap continuity; darkness gap 0.01-0.02 |
| `165921` | 10 | 22 | 1 | SFace 0.82; darkness gap 0.09; confidence 0.79 |

Result: every MOV produced quality-gated face evidence and at least one conservative cross-segment identity. When timed overlap continuity is unavailable, an identity merge now requires SFace similarity >= 0.55 and jersey darkness gap <= 0.20. Similar clothing without a quality SFace match remains unmerged.

## Problems Found And Fixed

- Sparse fixed scoreboard sampling missed short terminal scoreboard appearances. Replaced with ranked full-video scanning and burst extraction.
- Rolling-shutter LED phases truncated digits or changed 6 to 8. Added consecutive raw phases, boundary sharpening, temporal percentile fusion, and consensus.
- Late weak reads could replace a complete earlier score (for example 108-90 to 10-90). Added truncation, monotonicity, clock, and cross-anchor consistency gates.
- Generic OCR could read clocks or captions as scores. Added position/color validation and VLM fallback.
- Segment-local player IDs fragmented across overlaps. Added timed overlap boxes, IoU/center continuity, torso/jersey darkness features, and YuNet/SFace embeddings.
- The conservative timed-box rule rejected a real same player whenever tracks did not survive into the overlap sampling instants. Added a narrow fallback that requires both a quality SFace match and compatible jersey darkness; body appearance alone cannot trigger it.
- Strong earlier close-up panels could raise the global quality percentile and crowd out a distant terminal board. Candidate selection now reserves the strongest raw candidate from the final fifth of the candidate timeline, fills the main budget by quality with temporal diversity, and keeps a separate latest-candidate rescue slot.
- A rolling-shutter board around the `160824` terminal window could visually alias `69-52` as `95-92`. Later anchors now receive only the previously reconciled same-video score plus a physically plausible scoring-rate bound; this allowed complementary phases at 695.1s and 761.6s to recover `69-52` without a filename-specific score.

## Open-Source Capability Boundary

AGU wraps OpenCV YuNet/SFace for local face recognition, RapidOCR as an optional deterministic scoreboard reader, and the existing configurable VLM as fallback/audit. These remain adapters configured through `app/config.py`; missing optional models fall back without changing the v3 action preprocessing contract. No new service language or BFF behavior was introduced.

## Verification

- Accurate CLI two-segment runs were executed for all nine MOVs; seven new artifacts are in `identity_gate/`, with the earlier `150151` and `165230` quality-gate artifacts reused.
- Focused identity/scoreboard/CLI/hybrid test set: `107 passed` before the final identity fallback adjustment.
- Identity regression selection after the adjustment: `7 passed`.
- Fresh post-fix accurate CLI task `9fc5964fc49f428883cf44db8092c7f0` completed through the local API. Its saved result contains the expected `segment_0:player_17` to `segment_1:player_6` identity stitch with SFace 0.89, jersey darkness gap 0.00, continuity 0.86, and combined confidence 0.86.
- Final full pytest after candidate, rolling-shutter context, and identity refinements: `139 passed` (15 dependency/test-fixture warnings).
- Harness gate: `/Users/ppt/projects/agu/venv/bin/python scripts/verify_harness.py` passed.
- Final latest-code local curl: `/health` returned `ok`; `/ready` returned `ready`; smoke task `679dde9f205d4d6381d35ea72d24624a` completed with 7 records.
