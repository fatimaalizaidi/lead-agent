# Public demo page

`GET /demo` (and `/`, which redirects there) serves a one-page UI for showing the project to people who won't use an API key: recruiters, LinkedIn visitors, interviewers.

Live: <https://lead-agent-fjxq.onrender.com/demo>

## What a visitor sees

- A lead box with one-click examples (ready to buy, comparing tools, browsing, a Roman Urdu lead, spam, urgent support, prompt injection). The leads are for ReplyHub, a demo company that sells WhatsApp team-inbox software.
- **Both models side by side:** GPT-4o-mini + hybrid RAG (priority, suggested reply, rationale, knowledge sources, latency) and the LoRA fine-tuned classifier (priority, per-class confidence, latency in ms), with a banner saying whether they agree.
- A follow-up checkbox that continues the conversation (Redis memory).
- How it works, the fine-tuned model's evaluation numbers, live queue/memory status, and the tech stack.

The page is a single static file, `static/demo.html` (no build step, no external scripts, light and dark mode, works at phone width). Its API lives in `demo.py`.

## Demo API (no key, not in `/docs`)

| Endpoint | Does |
|---|---|
| `POST /demo/api/analyze` `{"message", "thread_id"?}` | Runs `run_analyze_lead` (the same code as `/analyze-lead`) and the classifier in parallel. |
| `POST /demo/api/classify` `{"message"}` | Classifier only. No OpenAI call, no budget used. |
| `GET /demo/api/info` | Model card, today's remaining budget, queue stats (aggregate only), memory on/off. |

## Keeping a public link safe and cheap

| Control | Default | Setting |
|---|---|---|
| Per-IP limit on GPT analyses | 6/minute, 40/day | `DEMO_RATE_LIMIT` |
| Per-IP limit on classifier-only calls | 30/minute | `DEMO_CLASSIFY_RATE_LIMIT` |
| Global OpenAI analyses per UTC day | 200 (≈ $0.03/day with gpt-4o-mini) | `DEMO_DAILY_LLM_LIMIT` |
| Message length | 1,000 characters | `demo.MAX_DEMO_CHARS` |
| Turn the demo off (404) | on | `DEMO_ENABLED=false` |

- Past the daily cap the page still answers with the free classifier and tells the visitor the GPT budget is used up. The counter lives in Redis (`demo:llm:<date>`), or in memory without Redis.
- The same guardrails as the API apply: the prompt-injection example is blocked before any model runs.
- Demo conversations use `demo-…` thread ids. A visitor-supplied id that isn't a demo thread is ignored, so the page can never read memory from real (n8n) conversations.
- No API key ever reaches the browser. The real endpoints keep requiring one.
- The page tells visitors that messages go to OpenAI and the tracing logs.

Also set a monthly hard limit in the OpenAI dashboard as a last line of defence.

## Tests

`tests/test_demo.py`: page and redirect, both models without a key while auth is on, budget fallback, Redis counter, per-IP 429, guardrail, length cap, thread isolation, follow-up memory, LLM failure fallback, `DEMO_ENABLED=false`, and demo routes staying out of the OpenAPI schema.
