# Moodle Assignment Agent

This project finds relevant Moodle assignments, downloads the source PDF, reads it with PDF-to-image extraction, generates rich instructions, and can optionally generate a solution draft or prepare a submission.

## Subject-agent architecture

After an assignment is downloaded, text is extracted from every readable PDF page and up to 12 pages are also
rendered for diagrams or scanned content. A one-pass vision extraction turns each source page into a cached
`document_manifest.json`; every page is included in exactly one extraction chunk. Routing then classifies it as
`organic_chemistry`, `mathematics`, `systems_programming`, or `general`. The workflow dispatches it to the
matching specialist prompt and records the topic, confidence, routing method, and selected agent in the output
JSON. If model-based classification fails, a deterministic keyword classifier provides a safe fallback.

Assignments are processed by a bounded thread pool. Every worker owns separate Moodle and OpenAI HTTP sessions,
so multiple assignments can be classified, solved, rendered, and optionally submitted concurrently. Control the
pool size with `--max-workers` (default: 3). The order in `summary.json` remains the due-date order.

The manifest contains the atomic inventory, source transcription, page mapping, visual evidence, confidence, and
crop coordinates. Solution batches consume only manifest JSON and do not receive full-page images. A question
below the confidence threshold may receive one tightly cropped fallback image. The workflow solves items in
source-ordered batches of up to four, splits truncated batches, and retries missing items individually.
Manifest extraction uses strict Structured Outputs. Every source page is audited; if validation finds a missing
page, one retry sends only those missing pages and then either produces a complete manifest or stops before PDF
generation. Raw responses and validation diagnostics are saved alongside the manifest.
The final document is assembled by stable question ID, not by concatenating arbitrary model responses. A coverage
artifact records expected, answered, and missing IDs. `generation_checkpoint.json` is updated after every batch,
so a rerun resumes completed work. Incomplete documents are rendered for review but are never submitted, even
when `--submit` is present.

Agent definitions live in `agents/*.toml`. `src/agent_loader.py` validates and loads them into a registry;
`src/subject_agents.py` contains only the generic execution class. To add a subject, create another TOML file
with `topic`, `display_name`, `instructions`, and optional `aliases`, `keywords`, and `enabled` fields. The
classifier automatically includes every enabled definition, so adding a subject does not require Python changes.

### Per-agent LLM settings

Each subject TOML can select its own provider, model, temperature, and reasoning effort:

```toml
provider = "openai"
model = "gpt-5.6-terra"
reasoning_effort = "medium"
temperature = 0.2
output_style = "ipad_clean"
paper_style = "grid"
```

Only the `openai` provider is implemented currently. The provider abstraction lives in `src/llm_providers.py`,
so another provider can be added later without changing the workflow or agent definitions. An unsupported
provider fails explicitly. If `model` is omitted, the agent inherits `OPENAI_MODEL` from `.env`. Because agents
consume text manifests; the global `OPENAI_MODEL` performs the one-pass vision extraction and per-agent settings
control the instruction/solution generation calls.
For GPT-5.6 models, `reasoning_effort` is sent and `temperature` is intentionally omitted. Older models retain
temperature support.

`output_style` may be `standard` or `ipad_clean`. The iPad profile embeds its handwriting
font directly, keeps MathJax formulas precise, and applies the selected `paper_style`
(`plain`, `grid`, or `ruled`). Personal handwriting is the default. Use
`--handwriting-font gveret` to select the bundled Gveret Levin font instead. Routing reports
always use the standard style.

## Setup

1. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Install the browser used for HTML-to-PDF rendering:

```powershell
python -m playwright install chromium
```

3. Create a local `.env` file based on `.env.example`.

Minimal required values:

```env
OPENAI_API_KEY=your_openai_api_key
MOODLE_TOKEN=your_moodle_token
MOODLE_ACTIVE_COURSE_IDS=65610,64359,65397,64403,65602,64408
MOODLE_BASE_URL=https://moodle.bgu.ac.il/moodle
MOODLE_SERVICE=moodle_mobile_app
MOODLE_TIMEZONE=Asia/Jerusalem
OPENAI_MODEL=gpt-5.6-terra
OPENAI_REASONING_EFFORT=medium
OPENAI_MIN_REQUEST_INTERVAL_SECONDS=0.25
OPENAI_TOKEN_SAFETY_RESERVE=30000
```

If you do not have `MOODLE_TOKEN`, you can use:

```env
MOODLE_USERNAME=your_username
MOODLE_PASSWORD=your_password
```

## Basic Commands

Check local readiness:

```powershell
python moodle_agent.py --doctor
```

Fetch a Moodle token from username/password and save it into `.env`:

```powershell
python moodle_agent.py --bootstrap-token --save-token
```

Generate instructions from assignment PDFs for assignments due by end of week:

```powershell
python moodle_agent.py --date 2026-06-21 --max-assignments 2 --workflow-mode instructions
```

Run the second solver stage after instructions:

```powershell
python moodle_agent.py --date 2026-06-21 --max-assignments 2 --workflow-mode instructions --solver-stage
```

Generate a direct submission-style answer:

```powershell
python moodle_agent.py --date 2026-06-21 --max-assignments 1 --workflow-mode solve
```

Force a fresh vision extraction instead of reusing a compatible cached manifest:

```powershell
python moodle_agent.py --date 2026-06-21 --workflow-mode solve --rebuild-manifest
```

Generate a local draft for a historical assignment whose submission window is closed:

```powershell
python moodle_agent.py --date 2026-07-09 --selection-window today --workflow-mode solve --include-closed
```

`--include-closed` cannot be combined with `--submit`.

Try actual submission back to Moodle:

```powershell
python moodle_agent.py --date 2026-06-21 --max-assignments 1 --workflow-mode solve --submit
```

Process up to four assignments concurrently through their selected subject agents:

```powershell
python moodle_agent.py --workflow-mode solve --max-assignments 8 --max-workers 4
```

Use the personal handwriting font (default):

```powershell
python moodle_agent.py --workflow-mode solve --handwriting-font personal
```

Use the previous Gveret Levin font:

```powershell
python moodle_agent.py --workflow-mode solve --handwriting-font gveret
```

## Rendering Pipeline

### Genetic pedigrees

Solutions that request a family pedigree return structured `pedigrees` alongside
the answer HTML. People have stable IDs, sex (`M`, `F`, `U`), separate disease and
carrier status (`true`, `false`, or `null`), parent IDs, and quoted evidence for
each fact. Partnerships are explicit, including childless couples. The generator
validates topology and evidence quotations, then asks the configured provider to
check facts and completeness against the original question. Failed checks are
fed back within the existing three-attempt limit; an invalid diagram is not a
completed answer. This semantic review adds a model request for each answer that
contains pedigrees and is not a guarantee of factual correctness.

Pedigreejs 5.0.2, D3 7.9.0 and jQuery 3.7.1 are bundled locally under
`assets/vendor/pedigree/`, with licenses, source URLs, and SHA-256 checksums in
`versions.json`. Pedigreejs is GPL-3.0; the local copies retain their license
notices. Pedigree drawing needs no CDN, editor, or browser storage. Existing
MathJax loading for equations is unchanged.

The renderer uses clean symbols and the selected handwriting font for labels.
Each rendered diagram also produces `<basename>_pedigree_N.json` and `.svg` beside
the PDF. Overlapping labels, unreadably large diagrams, missing nodes, and
unsupported partial-parent multi-generation layouts produce a rendering error
instead of a misleading PDF or text-only fallback. Large readable diagrams move
to their own page. Older cached answers that require a pedigree are regenerated
selectively; unrelated cached answers are preserved.

Run all workflow and pedigree checks with `python -m unittest tests.test_moodle_workflow`.
To include offline Chromium rendering checks in PowerShell:

```powershell
$env:RUN_PEDIGREE_BROWSER_TESTS = '1'
python -m unittest tests.test_moodle_workflow
```

The canonical document output is now HTML, not plain text.

1. The LLM generates a structured HTML fragment with headings, lists, LaTeX math, and mixed Hebrew/English text.
2. `src/output.py` wraps that fragment in a full RTL-aware HTML document with CSS and MathJax.
3. Playwright renders the local HTML into PDF through Chromium so the PDF matches browser layout much more closely.
4. A `.txt` file is still written as a debug/plain export, but it is no longer the source used to format the PDF.

Assignment HTML/PDF files use the render profile selected by their subject agent and the handwriting
font selected by the CLI flag. Both personal and Gveret font assets are embedded as data URIs, so
their appearance does not depend on an installed font or an external font service.

If Playwright or Chromium is missing, the workflow falls back to the older text-based PDF path, which is less reliable for Hebrew and math.

## Output Structure

All output is written under `generated_submissions/`.

- `generated_submissions/summary.json`: summary of the last run
- `generated_submissions/routing_report.html`: readable report showing how each assignment was classified and which agent/model handled it
- `generated_submissions/routing_report.json`: machine-readable version of the routing report
- `generated_submissions/<DATE>_<COURSE>_<ASSIGNMENT>/`: one folder per assignment

Each assignment folder can contain:

- original source PDF downloaded from Moodle
- source preview images extracted from the PDF
- `instructions.txt`, `instructions.html`, `instructions.pdf`, `instructions.json`
- optional `solution.txt`, `solution.html`, `solution.pdf`, `solution.json`
- optional `answer.txt`, `answer.html`, `submission.pdf`, `answer.json`
- `<basename>_coverage.json` with the question inventory, attempts, finish reasons, compounds, and missing IDs
- `document_manifest.json` with the single-pass source transcription and visual evidence
- `document_manifest.raw.json` with the successful initial Structured Output response
- `document_manifest_retry.raw.json` when the one permitted missing-page retry runs
- `manifest_validation.json` with page-by-page audit diagnostics
- `generation_checkpoint.json` with completed answers used for automatic resume

## Notes

- Default selection window is `week`, which means assignments due from the reference date until the end of the week.
- Use `--selection-window today` to only process assignments due on the reference date.
- The main CLI is `moodle_agent.py`. `moodle agent.py` is kept only as a compatibility wrapper.
