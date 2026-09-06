# Repository Guidelines

## Project Structure & Module Organization
The codebase is organized around the `src/` package, with `moodle_agent.py` kept as the thin CLI wrapper. Core workflow logic lives in `src/workflow.py`, Moodle API access in `src/moodle_client.py`, PDF extraction in `src/pdf_reader.py`, prompt generation in `src/llm.py`, provider adapters in `src/llm_providers.py`, and HTML/PDF output in `src/output.py`. Shared dataclasses are in `src/models.py`, configuration loading in `src/config.py`, and date-based assignment filtering in `src/selection.py`. Subject agents are declared in `agents/*.toml`, loaded and validated by `src/agent_loader.py`, and executed through the generic class in `src/subject_agents.py`. Tests live in `tests/test_moodle_workflow.py`.

## Build, Test, and Development Commands
Install dependencies with `python -m pip install -r requirements.txt`. Install the Chromium renderer with `python -m playwright install chromium`. Use `python moodle_agent.py --doctor` to verify env variables and rendering support. Run `python moodle_agent.py --workflow-mode instructions` for instruction generation, `python moodle_agent.py --workflow-mode instructions --solver-stage` to add the second solve pass, and `python moodle_agent.py --workflow-mode solve --submit` only when you want to upload back to Moodle. Run tests with `python -m unittest tests.test_moodle_workflow`.

## Coding Style & Naming Conventions
Use 4-space indentation and standard Python `snake_case` for modules, functions, and variables. Keep responsibilities separated by module: do not mix Moodle API calls into rendering code or CLI parsing into workflow internals. Prefer small helper functions over long inline blocks, and keep user-visible artifact naming stable, for example `instructions.html`, `instructions.pdf`, and `submission.pdf`.

## Testing Guidelines
Add or update `unittest` coverage whenever behavior changes in selection, env loading, PDF extraction, or output rendering. Name tests `test_*` and import from `src` directly. For rendering changes, cover RTL HTML wrappers, LaTeX preservation, image embedding, and PDF fallback behavior. If a change depends on Playwright or PyMuPDF availability, keep the test deterministic by mocking or by using the existing fallback path.

## Commit & Pull Request Guidelines
Git history is limited in this workspace, so use clear imperative commit messages such as `Add browser PDF renderer`. Keep pull requests focused, explain the workflow impact, list the commands you ran, and note any dependency or environment changes like new browser install requirements.

## Security & Configuration Tips
Never commit real `OPENAI_API_KEY`, `MOODLE_TOKEN`, passwords, or `credentials.json` contents. Store secrets in `.env`, keep generated assignment artifacts under `generated_submissions/`, and rotate any credential that was exposed locally. When troubleshooting rendering, prefer `--doctor` and local dry runs before testing live submission.
