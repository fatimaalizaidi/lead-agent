# RAG upgrade — done

Local **ChromaDB + LangChain** over `knowledge/*.md`, then strict JSON lead analysis.

**Use this copy:** `~/Projects/lead-agent`

## Flow
1. Incoming `message` is embedded and matched against the company KB in Chroma (`./data/chroma`).
2. Top articles are injected into the OpenAI prompt.
3. Response stays Pydantic-validated JSON, plus `knowledge_sources` (article titles used).

## Add KB content
Put a new markdown file in `knowledge/` with a `# Title`. The index rebuilds when the KB fingerprint changes (or delete `data/chroma` and restart).

## Test
```bash
cd ~/Projects/lead-agent && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
```

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze-lead" \
  -H "Content-Type: application/json" \
  -d '{"message":"What does Growth plan cost per month, and do you offer annual discounts?","email":"buyer@acme.com"}'
```

Expect `knowledge_sources` to include something like `Pricing Plans`.
