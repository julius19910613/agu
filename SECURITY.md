# Security Policy

## Supported Versions

Security fixes target the latest released minor version. AGU is currently an
alpha project, so users should pin versions and review release notes.

## Reporting a Vulnerability

Do not open a public issue for secrets, path traversal, unsafe model loading,
personal video exposure, or remote-code-execution concerns. Use GitHub's private
security advisory flow for the repository. Include affected version, minimal
reproduction, impact, and suggested mitigation when available.

## Deployment Notes

- AGU is a self-hosted analysis engine, not an authentication or public-upload
  gateway. Put authentication, request limits, and upload quarantine upstream.
- Restrict `BASKETBALL_ALLOWED_VIDEO_ROOTS` to trusted local/mounted paths.
- Treat pickle/PyTorch checkpoints as executable input; only load trusted files.
- Do not expose Ollama or model-service endpoints directly to untrusted networks.
- Analysis results can contain biometric and player identity evidence; apply
  appropriate consent, retention, and access controls.
