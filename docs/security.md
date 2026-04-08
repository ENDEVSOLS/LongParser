# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Email security reports to: **technology@endevsols.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Your name / handle (for credit, if desired)

We will acknowledge within **48 hours** and aim to release a patch within **14 days**
for confirmed critical vulnerabilities.

## Threat Model

LongParser processes arbitrary user-supplied documents and routes them through LLMs.
Key risks:

| Risk | Mitigation |
|------|-----------|
| **Prompt injection via documents** | System prompt explicitly forbids following `<CONTEXT>` instructions |
| **File upload exploits** | MIME allowlist, 100MB size cap, streaming write to isolated path |
| **API key leakage** | Keys never logged; only SHA-256 hash used as `tenant_id` |
| **MongoDB injection** | Motor driver + typed Pydantic inputs prevent injection |
| **SSRF via webhook** | No outbound HTTP made based on user input |
| **Hallucinated citations** | Citation IDs validated against retrieved set before returning to client |
| **DDoS / Spam via API** | Route-level Rate Limiting strictly isolated per tenant via Redis |
| **Cross-Origin attacks** | Configurable CORS restrictions and strict Tenant Isolation |

## Dependency Security

We use `uv` for deterministic dependency resolution. Run:

```bash
uv audit  # Check for known vulnerabilities in dependencies
```
