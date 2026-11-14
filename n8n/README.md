# n8n workflows (lead-agent)

| File | Purpose |
|------|---------|
| `lead-priority-routing.json` | Webhook → `/analyze-lead` → Switch High/Medium/Low (Slack / local JSON / SMTP) |

Import into n8n (UI or `n8n import:workflow`). Full setup + Postman bodies: [docs/n8n-priority-routing.md](../docs/n8n-priority-routing.md).

**API key:** the deployed API requires one. After importing, add an n8n **Header Auth** credential (name `X-API-Key`, value = your `la_…` key) and select it on the **Analyze Lead** node (Authentication → Generic Credential Type → Header Auth). Otherwise every call returns `401`. Details: [docs/security.md](../docs/security.md).
