# FP_StockAnalysis

## CI: Automated Retraining Workflow

This repository includes a scheduled GitHub Actions workflow that retrains the models weekly, runs tests, commits updated artifacts back to the `main` branch, and can optionally deploy to Heroku.

- Workflow file: `.github/workflows/retrain-weekly.yml`
- Schedule: `cron: '0 2 * * 0'` (every Sunday at 02:00 UTC)
- Trigger: Scheduled + manual (`workflow_dispatch`)

### What the workflow does

- Checks out the repository and sets up Python 3.11
- Installs dependencies from `requirements.txt`
- Runs `pytest` to execute CI tests (fails early if tests fail)
- Runs the pipeline via: `PYTHONPATH=. python run_all.py run`
- If `models/`, `data/`, or `predictions/` change, commits and pushes updates back to `main`
- Optionally deploys to Heroku when the required secrets are present

### Required GitHub repository secrets

Add these in the repository: `Settings → Security → Secrets and variables → Actions`

- `HEROKU_API_KEY` (optional) — Heroku API key used to push and trigger a deploy. Leave empty to disable auto-deploy.
- `HEROKU_APP_NAME` (optional) — Heroku app name (used with `HEROKU_API_KEY`). Leave empty to disable auto-deploy.

Notes:
- The workflow uses the built-in `GITHUB_TOKEN` for committing changes; you don't need to add it manually.
- If you prefer the workflow to create a pull request instead of pushing to `main`, open an issue and I can change the workflow to create a PR for review.

### Local testing & manual run

To run the same steps locally:

```powershell
# Create virtualenv and install deps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run tests
pytest -q

# Run the pipeline (retrain)
PYTHONPATH=. python run_all.py run
```

You can manually trigger the GitHub workflow from the Actions tab or run it on-demand using `workflow_dispatch`.

### Recommended repository settings

- Protect `main` if you want PR-based review before merging auto-updates.
- Add meaningful `git` commit info in the workflow user configuration if you want to map commits to an automation account.

If you want, I can update the workflow to open a PR instead of pushing to `main`, or add a status badge to this `README.md` showing the workflow pass/fail status.
# FP_StockAnalysis