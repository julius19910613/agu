# Code Review

## Review Result

PASS after fixes.

Findings addressed:

1. Initial provenance assumed every test service had full `Settings`; changed
   to an explicit `unknown` fallback for lightweight/offline callers.
2. Background runtime budget initially assumed `service.settings`; changed to a
   safe default so existing fake services and embedders remain compatible.
3. Metadata-only plugin discovery did not prove real pipeline extension; added
   before/after dispatch stage registration and an execution test.
4. Package metadata used a future-deprecated license table and placeholder
   repository URLs; changed to SPDX `MIT`, `license-files`, and the configured
   GitHub origin.
5. Minimal CI could import service-only tests without torch/OpenCV; heavy
   integration assertions now skip only in the minimal-core environment and run
   normally in the service/full suite.

Residual constraints are explicit public behavior: cancellation/deadlines are
cooperative at progress boundaries, task retry is in-memory, and the public
fixture validates contracts rather than production model accuracy.
