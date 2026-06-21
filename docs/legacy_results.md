# Legacy results log (historical)

> **Historical reference only.** These numbers were collected during the early,
> single-point manual-logging era against the **deprecated per-scene `.app`
> clients** (before the unified `scene_all` client and the
> `nav.scripts.run_benchmark_grid` sweep pipeline). They are *not* directly
> comparable to current runs — the scene set, point selection, success
> threshold, and harness all changed during the refactor. The authoritative,
> reproducible numbers now come from `python -m nav.scripts.compile_stats` over
> grid outputs; the live shared sheet is
> <https://docs.google.com/spreadsheets/d/1Z73hxSGpF-I8lIHdQ99BkMO_QHYAcf5CJPchR3OmWvQ/edit?usp=sharing>.

Kept here so the early single-point comparison isn't lost.

| Model                               | Total steps | Success rate (%) | Distance |
| ----------------------------------- | ----------: | ---------------: | -------: |
| openai/gpt-4o                       |          50 |                0 |    593.9 |
| openai/gpt-5-image-mini             |          38 |              100 |     15.2 |
| anthropic/claude-sonnet-4.5         |          43 |              100 |     12.6 |
| anthropic/claude-haiku-4.5          |          50 |                0 |   246.81 |
| google/gemini-2.5-flash             |          38 |              100 |        9 |
| qwen/qwen3-vl-30b-a3b-instruct      |          50 |                0 |   594.43 |
| meta-llama/llama-4-scout            |          50 |                0 |   649.43 |
| qwen/qwen3-vl-8b-instruct           |          50 |                0 |   721.65 |
| nvidia/nemotron-nano-12b-v2-vl:free |          50 |                0 |   436.95 |

These were single-point illustrative runs; the metric definitions (success =
distance < 65px, etc.) are documented in `nav/eval/metrics.py`.
