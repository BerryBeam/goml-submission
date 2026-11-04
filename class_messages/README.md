# Local-LLM Class Task Extractor

Reads a folder of short class/project-week messages and writes one JSON task list. Inference runs **entirely locally** through [Ollama](https://ollama.ai/) with `qwen2.5:1.5b`. No messages are sent to a hosted API.

Built for a low-resource machine (CPU only, 4 threads, ~8 GB RAM), so most of the work is done in Python and the small model only makes narrow decisions.

## Quick Start

```bash
# 1. Start Ollama and pull the model
ollama serve
ollama pull qwen2.5:1.5b

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Run
python extract.py --inbox messages --output tasks.json
```

## Input and Output

Each message is a `.txt` file with a header, a blank line, then the body:

```text
message_id: ex01
from: Class mentor
date: 1 Mar 2024

Riya, please share the venue list by Tuesday 11am.
```

Output (`tasks.json`):

```json
{
  "tasks": [
    {
      "message_id": "ex01",
      "task": "Share the venue list",
      "owner": "Riya",
      "due": "Tuesday 11am"
    }
  ]
}
```

- `task`: short line describing what to do
- `owner`: the person asked to do it, or `null`
- `due`: the deadline in the message's own words, or `null`
- Messages that do not ask for a task are left out
- Tasks are sorted by natural message ID order (`m2` before `m10`)

## How It Works

Two passes keep the 1.5B model accurate:

1. **Pass A, classify:** the model returns `task` or `no_task`. Chatter, FYIs, social invites, cancellations, and questions exit early.
2. **Pass B, extract:** runs only for actionable messages. Python first extracts candidate deadlines with regex, and the model may only pick one of those or `null`.

```text
Message -> parse header -> injection filter -> Pass A (classify)
        -> candidate deadlines (regex) -> Pass B (extract)
        -> grounding checks -> Pydantic validation -> atomic JSON write
```

Header fields (`from`, `date`) are deliberately **not** sent to the model. The sender is not necessarily the task owner, and header dates invite hallucinated deadlines.

### Due-date rule

`due` is set only when the message frames the time as a deadline (`by`, `before`, `until`, `within N days`, `end of day`, `tomorrow`, or `on <date>` alongside a delivery verb such as submit/send/share). Event times are not deadlines: in "Book the hall for the demo on 18 March at 10am", `due` is `null`.

## Guardrails (Prompt Injection)

Messages are treated as untrusted input, with three layers of defense:

1. **Regex pre-filter:** strips generic directives such as "ignore previous instructions", "system prompt", "you are now", and "act as". It is topic-agnostic, so legitimate tasks like "share the answer key" are not blocked.
2. **Prompt isolation:** the body is wrapped in `<message>...</message>`, and the system prompt says enclosed text is untrusted.
3. **Output grounding in Python:**
   - Owner must appear as a whole word in the message, and generic words (someone, everyone, nobody, I) become `null`
   - Due must exactly match a regex-extracted candidate, otherwise `null`
   - Tasks containing an email address not present in the message are dropped
   - Duplicate tasks are collapsed by (core words, owner)

## Determinism

Every request uses `temperature: 0`, `top_k: 1`, `seed: 42`, fixed context and thread settings, and a JSON-schema `format` constraint. Two runs on the practice folder produced byte-identical output:

```text
SHA-256 (run 1): EABAF170431819D5F5D8B3C801BC8CE228C3B879FE1A4E6F1E0C47FA1C251965
SHA-256 (run 2): EABAF170431819D5F5D8B3C801BC8CE228C3B879FE1A4E6F1E0C47FA1C251965
fc.exe /b: no differences encountered
```

## Error Handling

| Situation | Behavior |
| :--- | :--- |
| Ollama unreachable or model missing | Message on stderr, writes `{"tasks": []}`, exits with code 1 |
| LLM call fails or times out (120 s) | Retries once, then skips that message and continues |
| Non-`.txt` file, binary file (null bytes), empty body | Skipped and logged to stderr |
| Body over 4,000 characters | Truncated and logged |
| Missing `message_id` header | Falls back to the filename stem |
| Output write | Written to a temp file, then `os.replace` for an atomic commit |

## Token Usage (practice folder, cold run)

Counted from Ollama's `prompt_eval_count` and `eval_count`, summed over all calls and printed to stderr:

```text
Tokens: 3845 prompt, 174 completion, 4019 total
```

About 58.5 s for 6 messages (~9.75 s per message) on 4 CPU threads.

## Project Structure

```text
.
├── extract.py        # Extractor
├── NOTES.md          # Implementation notes
├── requirements.txt  # pydantic
├── messages/         # Practice messages
└── example/          # One solved message + expected.json
```

## Configuration and Limitations

- The model name, `num_gpu: 0`, and `num_thread: 4` are hardcoded in `extract.py`. Edit them for a different machine.
- Set `OLLAMA_HOST` to use a non-default Ollama address (default `http://localhost:11434`).
- At most 5 tasks are kept per message.
- The injection filter is regex-based and removes the matched text through the end of the line, so unusual phrasing in a legitimate message could be trimmed.
- Deadline detection is regex-based, so phrasings outside the supported cues yield `due: null` rather than a guess. This is deliberate: it favors no deadline over an invented one.
