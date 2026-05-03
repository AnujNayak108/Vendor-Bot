# Vendor Bot
## What this is

A **dataset-grounded** composer for the magicpin merchant AI assistant challenge: one `compose(category, merchant, trigger, customer)` function plus a **FastAPI** service matching `challenge-testing-brief.md` (`/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`).

## Approach

- **Trigger dispatch**: each `trigger.kind` gets a tailored template so messages stay specific to the event (research digest, recall, perf dip, IPL, recall, pharmacy recall, etc.).
- **No fabrication**: numbers, citations, slots, and batch IDs come from the pushed contexts only. Expanded triggers with `payload.placeholder` get safe generic copy that does not invent medicines or appointment times.
- **URLs stripped** from bodies to avoid judge penalties from the examples brief.
- **Multi-turn** (`/v1/reply`): auto-reply streak handling, commitment → action mode, hostile opt-out, GST off-topic redirect.

## Tradeoffs

- **Deterministic templates** instead of an LLM: fast, reproducible, and no API spend — you can swap `compose()` internals for an LLM + validator later without changing the HTTP surface.
- **Suppression**: after a successful `/v1/tick` send, `suppression_key` is remembered so the same trigger is not spammed across ticks (adjust if your judge expects repeats).

## Files

| File | Role |
|------|------|
| `bot.py` | Composer + FastAPI app |
| `build_submission.py` | Writes `submission.jsonl` from `dataset/expanded/test_pairs.json` |
| `submission.jsonl` | 30 canonical rows (after running dataset expand + build) |
| `conversation_handlers.py` | Optional `respond()` wrapper |

## Commands

```bash
pip install -r requirements.txt
python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded
python build_submission.py
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Judge simulator (needs LLM key in `judge_simulator.py`):

```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

## What would help most in production

- Live appointment / Rx / slot data instead of placeholder triggers.
- A second-stage LLM for phrasing with a strict JSON schema validator against the contexts.
