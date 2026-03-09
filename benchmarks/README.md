# Benchmarks

Lightweight end-to-end benchmarks for Kestrel's live API stack.

These benchmarks are meant for iterative validation after orchestration, retrieval,
or report-quality changes. They are intentionally cheaper and more repeatable than
manual full runs.

## Files

- `prompts.json`: Curated benchmark prompts
- `run_live_benchmark.py`: Runs one or more prompts against the live backend API

## Usage

Start the stack first:

```bash
docker compose up -d backend agent searxng redis
```

Run a single benchmark:

```bash
python benchmarks/run_live_benchmark.py --prompt-id authoritative_reu
```

Run all prompts:

```bash
python benchmarks/run_live_benchmark.py --all
```

Override backend URL:

```bash
python benchmarks/run_live_benchmark.py --all --base-url http://localhost:8000
```

## Output

Each run writes a JSON artifact under `notes/benchmarks/` with:

- task id
- timing milestones
- research plan
- reports
- export payload
- lightweight quality signals

## Quality Signals

The script computes cheap structural indicators:

- plan subtask count
- presence of low-signal domains in the plan
- report section presence
- counts of authoritative / official source links
- whether the final report contains verified/tentative/uncertainty sections

These are not ground-truth quality scores. They are regression signals.
