# CI/CD (GitHub Actions → Render)

Every push and pull request runs [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). A push to `main` is deployed to Render only if every CI job passes.

```
git push ─▶ lint (ruff) ──┐
            tests (pytest)┴─▶ Docker build + /health smoke test ─▶ all green? ─▶ Render deploys
```

## Jobs

| Job | What it does | Fails when |
|---|---|---|
| **Lint** | `ruff check .` (config in `pyproject.toml`: pycodestyle, pyflakes, import order, bugbear, pyupgrade) | any lint error; shown inline on the PR diff |
| **Tests** | `pip install -r requirements-dev.txt` then `pytest -q` (pip cache keyed on both requirements files) | any test fails |
| **Docker** | builds the production `Dockerfile` (layer cache in GitHub Actions), starts it, polls `GET /health` for up to 60 s | the image won't build or the app won't start |

None of these need secrets. The tests mock OpenAI, Redis and the webhooks (`tests/conftest.py`), and the smoke test passes a placeholder `OPENAI_API_KEY`, because `/health` never calls OpenAI. Pushing a new commit to a branch cancels that branch's CI run if it's still going.

## Deploys

`render.yaml` sets `autoDeployTrigger: checksPass`, so Render waits for the GitHub checks on a `main` commit before deploying it. If CI fails, the previous version stays live.

**One-time step in Render:** the blueprint setting only applies if the service was created from the blueprint (or is synced to it). Otherwise, open the service in the Render dashboard → **Settings → Build & Deploy → Auto-Deploy** and choose **After CI Checks Pass**.

## Retrieval quality gate (`eval.yml`)

A second workflow, **RAG eval**, checks retrieval quality rather than code. `python -m eval.retrieval_gate` runs the real retrieval pipeline on all 26 golden questions and fails if top-1 accuracy, hit rate, MRR or fact recall drops under its floor in `eval/retrieval_thresholds.json`.

- **When:** pull requests and pushes to `main` that change `rag.py`, `config.py`, `knowledge/`, `eval/` or the requirements; every Monday at 06:00 UTC (catches drift in library versions); and on demand (Actions → RAG eval → Run workflow).
- **Blocks bad deploys:** it also runs on those pushes to `main`, and Render's *After CI checks pass* waits for every check on the commit, so a retrieval regression never goes live.
- **Cost: none.** No LLM and no API key: vector search uses Chroma's bundled local embedding model (all-MiniLM-L6-v2, cached between runs) instead of OpenAI. It takes about a minute.
- **Results:** the score table and any missed questions are on the run's summary page; per-question CSVs are attached as the `retrieval-results` artifact.

The full Ragas evaluation (faithfulness and answer relevancy, which need an LLM judge) stays a manual step, about $0.01 per 10 questions: see [rag-evaluation.md](./rag-evaluation.md). Don't make the gate a required check: it only runs when the paths above change, so a required check would sit at "pending" forever on other PRs.

## Running the same checks locally

```bash
pip install -r requirements-dev.txt
ruff check .            # add --fix to apply safe fixes
pytest -q
docker build -t lead-agent:ci . && docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-placeholder lead-agent:ci
```

## Protecting main (recommended)

GitHub → **Settings → Branches → Add rule** for `main` → **Require status checks to pass before merging**, and pick *Lint (ruff)*, *Tests (pytest)* and *Docker build + smoke test*. After that, a PR can't be merged while CI is red.
