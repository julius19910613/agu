# Third-Party Notices and Distribution Boundaries

AGU source code is MIT licensed. That does not automatically relicense optional
dependencies, model weights, datasets, or generated artifacts. Distributors and
deployers must review the exact versions and assets they use.

| Component | AGU use | Upstream license/boundary | Distribution policy |
| --- | --- | --- | --- |
| PyTorch / torchvision | Action and identity inference | BSD-style upstream licenses; pretrained weights can have separate terms | Optional `inference` extra; record weight origin and checksum |
| OpenCV | Video IO, tracking helpers, face adapters | Apache-2.0 upstream; model files need separate provenance | Optional `inference` extra |
| Ultralytics | YOLO detection, ByteTrack, BoT-SORT | AGPL-3.0 or Ultralytics Enterprise License | Isolated in `tracking-ultralytics`/`service`; never imply MIT relicensing |
| RapidOCR ONNX Runtime | Optional scoreboard OCR | Apache-2.0 package; bundled/downloaded models require provenance review | Optional `ocr` extra |
| Ollama and configured VLM | Optional local audit | Runtime/model-specific licenses | AGU does not distribute the service or model |
| SpaceJam or user video | Training/evaluation input | Dataset/user-specific terms | Not distributed by AGU |
| YuNet / SFace model files | Optional face evidence | Model-file terms must be checked at download time | Configure local paths; do not bundle without a provenance record |

Before a release:

1. Generate an environment inventory with
   `python scripts/generate_sbom.py --output build/sbom.json`.
2. Run `pip-audit` for known Python dependency vulnerabilities.
3. Review every shipped checkpoint/model file independently of code licenses.
4. Record source URL, version, license, checksum, and redistribution decision.
5. Do not publish private video, generated identity crops, or personal data.

This document is an engineering inventory, not legal advice.
