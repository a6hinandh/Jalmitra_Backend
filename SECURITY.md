# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in the Jalmitra backend, please **do not** open a public GitHub issue.

Instead, email **vionix37@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept requests, if applicable)
- Any suggested remediation

We aim to acknowledge reports within **72 hours** and to provide a resolution timeline once the issue is confirmed.

## Scope

This policy covers the `SIHb-2025` FastAPI backend, including:

- Injection risks (Cypher, SQL, prompt injection into the GraphRAG pipeline)
- Authentication/authorization and rate-limiting bypasses
- Exposure of secrets (`NEO4J_*`, `PINECONE_API_KEY`, `GENAI_API_KEY`) via logs, error messages, or responses
- Unsafe handling of user-submitted data (field observations, report generation, file exports)

## Supported Versions

Only the `main` branch is actively maintained and receives security fixes.

## Handling Secrets

Never commit `.env` files or API keys. Rotate any credential that may have been exposed, and report the exposure per the process above.

## Disclosure

We follow coordinated disclosure — please give us a reasonable window to address the issue before any public disclosure.
