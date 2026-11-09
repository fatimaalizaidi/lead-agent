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

The Ragas evaluation (`python -m eval.run_ragas`) isn't part of CI, because it makes real OpenAI calls. Run it by hand before changing retrieval or prompts; see [rag-evaluation.md](./rag-evaluation.md).

## Running the same checks locally

```bash
pip install -r requirements-dev.txt
ruff check .            # add --fix to apply safe fixes
pytest -q
docker build -t lead-agent:ci . && docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-placeholder lead-agent:ci
```

## Protecting main (recommended)

GitHub → **Settings → Branches → Add rule** for `main` → **Require status checks to pass before merging**, and pick *Lint (ruff)*, *Tests (pytest)* and *Docker build + smoke test*. After that, a PR can't be merged while CI is red.
