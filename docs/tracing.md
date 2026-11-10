# Tracing, cost and failed tool calls

Two layers, both in this repo:

| Layer | Always on? | What you get | Where |
|---|---|---|---|
| **Structured logs** | Yes | Per LLM call: latency, tokens and **estimated cost in USD**. Per tool call: name, latency, ok/failed and error. Per request: totals for all of these. | stderr, `logs/app.log`, or JSON with `LOG_JSON=true` |
| **LangSmith traces** | Off until you add a key | One trace per API request showing the whole run as a tree: graph → agent nodes → OpenAI calls, retrieval and tool calls. Each step shows its latency, tokens, cost, inputs and outputs. Failed tool calls are shown in red. | [smith.langchain.com](https://smith.langchain.com) |

## Turning on LangSmith

1. Create a free account at [smith.langchain.com](https://smith.langchain.com) and go to **Settings → API Keys → Create API Key**.
2. Locally, add to `.env`:
   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_PROJECT=lead-agent   # optional, default "lead-agent"
   ```
3. On Render, open the service's **Environment** tab and set `LANGSMITH_API_KEY`. `render.yaml` already sets `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=lead-agent`.
4. Restart the app and send a request. The startup log shows `tracing langsmith=True project=lead-agent`, and the trace appears in the project within a few seconds.

EU-region accounts also need `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`.

## What a trace looks like (`POST /multi-agent`)

```
POST /multi-agent                   8.4 s   $0.000326   tool_calls=2 tool_failures=0
└─ LangGraph
   ├─ supervisor
   │  └─ ChatOpenAI (gpt-4o-mini)   2.6 s   208 tokens
   ├─ sales
   │  ├─ ChatOpenAI                 2.1 s   311 tokens
   │  ├─ calculate_custom_quote     tool    0.2 ms
   │  ├─ check_calendar_availability tool   0.3 ms
   │  └─ ChatOpenAI                 2.5 s   772 tokens
   └─ supervisor
      └─ ChatOpenAI                 1.1 s   318 tokens
```

The timings and tokens above are from a real local run. Each request's root run has a `request_id` in its metadata, matching the `X-Request-ID` header and the log lines, so you can jump from a log line to its trace. `GET /health` is never traced, so Render's health checks don't flood the project.

**Spotting problems:**
- **Latency bottlenecks:** sort the project by latency, then open a trace. The waterfall shows which agent or call took the time. In the run above, LLM calls took 99% of the request.
- **Cost:** LangSmith shows cost per run and per trace, and charts it over time. The `cost_usd` on the root run is our own estimate, see below.
- **Failed function calls:** a tool that returns `ok: false` (bad arguments, unknown tool, webhook down…) is marked as an error. Filter the project by `run_type = tool` and `error`. Unparseable tool arguments count as failures too.

## Cost in the logs

`observability.estimate_cost_usd()` turns token usage into dollars using OpenAI list prices for `gpt-4o-mini`, `gpt-4o` and `gpt-4.1(-mini/-nano)`. Dated snapshot names are matched to their base model. For any other model, set `LLM_PRICE_INPUT_PER_1M` and `LLM_PRICE_OUTPUT_PER_1M`, otherwise the cost shows as `-`. Embedding calls for retrieval aren't counted; they cost a fraction of a cent per thousand queries.

```text
llm model=gpt-4o-mini llm_latency_ms=2482.8 prompt_tokens=666 completion_tokens=106 total_tokens=772 cost_usd=0.000163
tool name=calculate_custom_quote ok=True tool_latency_ms=0.2 error=-
request_end path=/multi-agent status=200 total_ms=8350.9 llm_latency_ms=8262.8 llm_calls=4 ... cost_usd=0.000326 tool_calls=2 tool_failures=0
```

A failed tool call logs at **WARNING**: `tool name=trigger_slack_workflow ok=False ... error=...`.

## Privacy

Traces include prompts, retrieved knowledge and lead messages, which may contain names and email addresses. Set `LANGSMITH_HIDE_INPUTS=true` to send traces without inputs. You still get the tree, timings, tokens and costs, but no message text. API keys are never sent.

## Code map

| Piece | File |
|---|---|
| Startup switch (`configure_tracing`), wraps the OpenAI client (`wrap_client`) | `tracing.py` |
| One root trace per POST (`TracingMiddleware`, placed inside `RequestIdMiddleware`) | `tracing.py`, `main.py` |
| Tool runs + failure marking + `log_tool` (`run_tool`) | `tracing.py`, used in `agent.py` and `multiagent.py` |
| Retrieval run (`@traced_retriever` on `retrieve_context`) | `rag.py` |
| LangGraph node runs | automatic (LangChain reads `LANGCHAIN_TRACING_V2`) |
| Cost table, `log_tool`, per-request totals | `observability.py` |

With tracing off, every helper passes calls straight through. `tests/test_tracing.py` covers both modes offline, using a fake `trace()`.
