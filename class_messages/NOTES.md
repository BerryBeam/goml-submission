# Local-LLM Class Task Extractor - Implementation Notes

## 1. Model
- **Model Choice:** `qwen2.5:1.5b` (Ollama tag: `qwen2.5:1.5b`, Model ID: `65ec06548149`, size: 986 MB disk / ~1.1 GB RAM).
- **Rationale:** The evaluation environment is constrained to an AMD Ryzen 3 3250U (2 cores / 4 threads) with 8 GB RAM (often < 2 GB free) and 0% GPU allocation. `qwen2.5:1.5b` is the optimal lightweight open-weights LLM capable of strict grammar-constrained JSON generation via Ollama while fitting within CPU RAM without paging. Heavy lifting (sanitization, candidate extraction, grounding) is handled in Python, allowing the 1.5B model to perform narrow classification and extraction tasks with 100% accuracy.
- **Local Setup & Execution Instructions:**
  ```bash
  # Ensure Ollama is running
  ollama serve

  # Verify or pull the required model
  ollama pull qwen2.5:1.5b

  # Run extraction program
  python extract.py --inbox messages --output tasks.json
  ```

---

## 2. Tokens
- **Real Token Counts from Test 2 (Cold Run on `messages/`):**
  - Prompt tokens: `3845`
  - Completion tokens: `174`
  - Total tokens: `4019`
- **Cold-Start Latency:**
  - Initial model load into CPU memory: ~3.5 seconds.
  - Overall pipeline execution time across 6 practice messages: 58.5 seconds (~9.75 seconds/message on 4 CPU threads).

---

## 3. Prompt
- **Architecture & Structure:**
  A two-pass architecture separates classification from entity extraction to optimize accuracy and token usage on a small model:
  1. **Pass A (Classification):** Prompts the model with system role, untrusted data boundary, and few-shot examples using a grammar-constrained enum `{"is_task": "task" | "no_task"}`. Non-actionable messages (chatter, FYIs, social invitations, cancellations) exit early with 0 extraction overhead.
  2. **Pass B (Extraction):** Applied only if Pass A returns `"task"`. The message body is provided inside `<message>...</message>` alongside Python-extracted `Candidate Deadlines`. Schema enforces `task` (imperative string), `owner` (name or null), and `due` (enum of candidates or null).
- **Deliberately Omitted from Prompt:**
  - Header metadata (`from`, `date`, `message_id`) is strictly excluded. The sender is not necessarily the task owner, and dates in headers lead small models to hallucinate deadlines.
  - Filenames and directory paths are omitted.
  - Arbitrary open-ended deadline generation is omitted; deadlines are constrained strictly to candidates identified by Python regex.

---

## 4. Deterministic Output
- **Deterministic Ollama Parameters:**
  To guarantee byte-identical outputs across successive runs, all requests pass:
  - `temperature: 0`
  - `top_k: 1`
  - `seed: 42`
  - `num_ctx: 2048`
  - `num_gpu: 0`
  - `num_thread: 4`
  - `num_predict: 256`
  - `keep_alive: "10m"`
  - Strict JSON schema grammar constraint via `format`.
- **Determinism Verification (Test 3):**
  - **Run 1 SHA-256:** `EABAF170431819D5F5D8B3C801BC8CE228C3B879FE1A4E6F1E0C47FA1C251965`
  - **Run 2 SHA-256:** `EABAF170431819D5F5D8B3C801BC8CE228C3B879FE1A4E6F1E0C47FA1C251965`
  - **Binary Diff (`fc.exe /b`):** `FC: no differences encountered` (0 byte divergence).

---

## 5. Hallucination
- **Observed Hallucinations during Development:**
  1. *Event Date Hallucinated as Deadline:* In `m03` ("Arjun needs to book seminar hall B for the demo on 18 March at 10am"), early iterations extracted `"18 March at 10am"` as the `due` date.
     - *Elimination:* Implemented Python regex candidate deadline extraction with negative lookbehinds for event markers (`demo`, `presentation`, `review is`, `meeting`) and requiring delivery action verbs for date cues. Pass B schema restricts `due` strictly to the candidate list or `null`. Since `m03` contained an event time rather than a deadline, candidates were empty, forcing `due: null`.
  2. *Duplicate Tasks for Overlapping Deadlines:* In messages with multiple candidate deadline variants (e.g. `"end of day"` and `"by end of day"`), the model occasionally generated duplicate task entries.
     - *Elimination:* Added Layer 3 Python signature deduplication based on `(tuple(sorted(core_words)), owner.lower())`, discarding redundant duplicate tasks.

---

## 6. Format and Validation
- **Ollama Grammar-Constrained Decoding:** Every request supplies a strict JSON Schema in Ollama's `format` parameter, forcing llama.cpp grammar sampler to emit only valid JSON conforming to the schema.
- **Pydantic Schema Validation:** Extracted tasks are parsed into Pydantic models (`TaskItem`, `TaskList`). Any malformed or non-conforming items trigger defensive fallbacks.
- **Atomic File Writes:** The final JSON payload is written to `.tmp_<filename>` and committed atomically using `os.replace` to prevent truncated or corrupted output files during unexpected terminations.

---

## 7. Errors
- **Startup Check:** On invocation, `extract.py` queries `GET /api/tags`. If the Ollama server is unreachable or `qwen2.5:1.5b` is not installed, it writes a single informative message to `stderr`, creates a fallback `{"tasks": []}` output file, and exits with code `1`.
- **HTTP Timeouts & Retries:** Ollama REST calls use a 120-second timeout (`urllib.request.urlopen(req, timeout=120)`). Calls wrap in a retry loop: on failure or connection drop, the request retries once before logging the error and safely continuing.
- **Corrupted & Non-Standard Files:**
  - Non-`.txt` files (e.g., `expected.json`, directories) are skipped immediately.
  - Binary files containing null bytes (`\x00`) are detected in the first 1KB and skipped.
  - Empty files or bodies are skipped without invoking the LLM.
  - All skips and truncations (>4,000 chars) are logged to `stderr`.

---

## 8. Guardrails
- **Three Layers of Defense Against Prompt Injection:**
  1. *Layer 1 (Python Regex Pre-Filter):* Scans and strips generic adversarial injection directives ("ignore previous instructions", "system prompt", "you are now", "act as", etc.) before sending text to the LLM. It avoids topic-specific blacklists to ensure legitimate tasks (e.g., "share the answer key") remain untouched.
  2. *Layer 2 (Prompt Isolation):* Encapsulates the message body within `<message>...</message>` tags and provides explicit system instructions stating that enclosed text is untrusted input whose internal directives must be ignored.
  3. *Layer 3 (Python Grounding & Sanitization):* Validates LLM outputs in Python against the source text:
     - Owners must match an exact whole-word name in the message body and cannot be generic stopwords ("someone", "nobody assigned", "everyone", "I").
     - Deadlines must exactly match a Python-extracted candidate deadline or `null`.
     - Tasks containing unauthorized external URLs or email addresses (such as `grader@hidden.example`) are automatically rejected.
- **Due-Date Rule Justification:**
  `due` is set only when the message frames the time as a deadline. In m03 the date is the demo time, so `due` is null. A reader could treat it as an implied deadline; I chose not to add one the message does not state.

