# Observability — done

App: `~/Projects/lead-agent`

## What you get
- **loguru** structured logs → stderr + rotating `logs/app.log`
- **`X-Request-ID`** middleware (accept from n8n or auto-generate; echo on response)
- Per request: Chroma latency (ms), LLM latency (ms), token usage, total duration
- Errors logged with `request_id` + `error_code` (no API keys)

## Sample
```text
request_start path=/analyze-lead email_domain=acme.com
rag chroma_latency_ms=42.3 hit_count=2 best_score=0.8123
llm model=gpt-4o-mini llm_latency_ms=810.5 prompt_tokens=420 completion_tokens=55 total_tokens=475
request_end status=200 total_ms=870.1 needs_human_review=False
```

## Try
```bash
curl -s -D - -X POST "http://127.0.0.1:8000/analyze-lead" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-n8n-001" \
  -d '{"message":"Do you offer annual discounts?","email":"buyer@acme.com"}'
```
Watch the terminal or `logs/app.log` for matching `request_id=demo-n8n-001`.
