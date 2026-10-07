# YT Lecture RAG

Retrieval over a YouTube DSA lecture playlist. A student asks a question in Hinglish or English and
gets **clickable links that jump to the exact second in the exact lecture**. An optional LLM
explanation, grounded only in the retrieved excerpts, can be added on top.

```
"bhai memoization aur tabulation ka difference kahan explain kiya hai?"

-> Sources:
   [1] DP Lecture 3 - Memoization Deep Dive        @ 12:04   > jump
   [2] DP Lecture 5 - Tabulation Recipe            @ 04:31   > jump
```

Most RAG demos return a blob of text. This returns *a place in a video*.

---

## Quickstart

The repo ships the cached transcripts (about 6 MB of JSON), so you never need a GPU, audio
downloads, or yt-dlp to try it. Search works with **no API key at all**.

```bash
git clone https://github.com/Gaurav4445/RAG_Project.git
cd RAG_Project
uv sync
uv run ytrag reindex      # builds the index from the bundled transcripts (about 2 min on CPU)
uv run ytrag serve        # open http://127.0.0.1:8000
```

Optional: to enable the written "explain" answer, copy `.env.example` to `.env` and add a
`GEMINI_API_KEY` or `GROQ_API_KEY`.

---

## Stack

| Layer | Choice |
|---|---|
| Playlist + audio | `yt-dlp` (Python API) |
| Transcription | `faster-whisper` (CTranslate2), `large-v3`, batched |
| Chunking | hand-written, time-window based, never splits a segment |
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` (384-dim) by default; `BAAI/bge-m3` optional |
| Ranking | cosine distance plus a lecture-title boost |
| Vector store | Qdrant, local folder by default, hosted cluster if `QDRANT_URL` is set |
| LLM (optional) | Gemini or Groq (`openai/gpt-oss-120b`), only for the "explain" button |
| CLI | `typer` + `rich` |
| API + UI | `FastAPI` + a single HTML page with the YouTube IFrame API |

### Why MiniLM and not bge-m3

bge-m3 is the better model on paper: multilingual and built for code-switched text. Measured on
this corpus of 2933 chunks:

| | download | index (CPU) | top-1 | top-5 |
|---|---|---|---|---|
| bge-m3 | 4.35 GB | 55 min | 12/12 | 12/12 |
| MiniLM | 87 MB | 1.6 min | 11/12 | 12/12 |

One question differs, and it still comes back at rank 2. The title-boost re-ranking in
`ytrag/index.py` recovers most of what the smaller model gives up, so better ranking turned out
to be worth more than a bigger encoder. Set `YTRAG_EMBED_MODEL=BAAI/bge-m3` and run `reindex`
if you want the last 1/12.

---

## Pipeline

```
playlist -> audio (yt-dlp) -> transcript (faster-whisper, cached) -> chunks (75s windows)
         -> embeddings -> Qdrant -> retrieval + title boost -> timestamped links (+ optional LLM answer)
```

Transcription is the only expensive step and it is paid once. Everything downstream rebuilds
from cached transcripts in minutes.

---

## Commands

| Command | What it does |
|---|---|
| `ytrag preflight --playlist URL` | Exercises every code path a long ingest depends on, in about a minute |
| `ytrag langtest URL` | Transcribes one lecture in two languages and prints both side by side |
| `ytrag ingest --playlist URL` | Download, transcribe, chunk and index a playlist. Safe to re-run |
| `ytrag progress` | How far along an ingest is. Safe to run while one is going |
| `ytrag reindex` | Re-chunk and re-embed from cached transcripts. Never re-transcribes |
| `ytrag search "query"` | Retrieval only, shows raw distances |
| `ytrag ask "question"` | Grounded answer with clickable timestamps |
| `ytrag eval --verbose` | Score retrieval against the golden set |
| `ytrag stats` | What is in the index right now |
| `ytrag export-transcripts` | Copy cached transcripts into the repo so they can be committed |
| `ytrag export-vectors-cmd` / `ytrag load` | Save the built index / load a prebuilt one in seconds |
| `ytrag clean-audio` | Delete downloaded audio. Transcripts are untouched |
| `ytrag serve` | Run the FastAPI app and web UI at http://127.0.0.1:8000 |

Citations in the web UI call `player.seekTo()` on the embedded player, so clicking a source
jumps inside the page instead of opening a new tab.

---

## Using your own playlist

This is the only path that needs a GPU for reasonable speed.

```bash
uv sync
uv sync --extra cuda          # Windows: CUDA runtime libs for ctranslate2 (optional)
uv run ytrag preflight --playlist "<PLAYLIST_URL>"
uv run ytrag langtest "https://www.youtube.com/watch?v=<one_lecture>"
uv run ytrag ingest --playlist "<PLAYLIST_URL>" --limit 1     # one video, end to end
uv run ytrag ingest --playlist "<PLAYLIST_URL>"               # the whole thing
```

**Decide the language flag first.** `YTRAG_WHISPER_LANG=hi` gives Devanagari output, `en` gives
romanised or translated output. Student queries are romanised Hinglish or English, so the two
behave very differently. Read a few minutes of each, check what happens to technical terms
(memoization, adjacency list, time complexity), and pick on evidence. The choice gets baked into
hours of compute.

If your GPU is newer than the shipped `ctranslate2` build, `transcribe.py` falls back to CPU
`int8` with a warning. Slower, still correct.

### Speed, measured on an RTX 5050 laptop (68.7-hour playlist)

| config | speed | total |
|---|---|---|
| `large-v3` sequential, beam 5 | 2.5x realtime | ~27 h |
| `large-v3` **batch 8**, beam 5 (default) | **8.1x** | **~8.5 h** |
| `large-v3` batch 8, beam 1 | 10.5x | ~6.5 h |

Batching is not a quality tradeoff: same model, same weights, identical technical-term capture
in testing. Beam 1 is faster but uses greedy decoding, which is likeliest to slip on unusual
technical terms in accented speech, so the default stays at beam 5.

### Surviving a long unattended run

The command never changes: re-run `ytrag ingest`.

- **Power cut.** Transcripts are written to a temp file and atomically renamed, so you get a
  complete file or none. Finished videos are skipped on re-run.
- **Network drop.** Downloads and Qdrant upserts retry four times with exponential backoff.
  Both are idempotent.
- **Network death.** After `--stop-after-failures` consecutive failures (default 5) the run
  aborts with a clear message instead of reporting a "finished" run that did nothing.
- **Bot challenges.** Random pauses between downloads (`YTRAG_DOWNLOAD_SLEEP_MIN/MAX`) keep
  YouTube's "confirm you're not a bot" check away. If challenged anyway, use
  `YTRAG_COOKIES_FILE` (a `cookies.txt` export). `YTRAG_COOKIES_FROM_BROWSER` works for
  Firefox but not for Chrome or Edge on Windows.
- **Ctrl-C.** Caught explicitly; cached work is safe.

---

## Runtime data

Lives outside the repo, in `~/.ytrag/` (override with `YTRAG_ROOT`):

```
~/.ytrag/
  audio/        <video_id>.m4a    deletable, regenerable in seconds
  transcripts/  <video_id>.json   precious, hours of GPU time
  qdrant/                         local vector index (when QDRANT_URL is unset)
```

The repo's `transcripts/` folder is the committed copy of the precious part. Transcripts are
about 1.4 KB per minute of video, so this whole playlist is about 6 MB.

---

## Configuration

Everything is env-driven; defaults live in [ytrag/config.py](ytrag/config.py).

| Variable | Default | Notes |
|---|---|---|
| `YTRAG_WHISPER_LANG` | `en` | Decide with `langtest` first |
| `YTRAG_WHISPER_MODEL` | `large-v3` | `medium` if you're impatient |
| `YTRAG_WHISPER_DEVICE` | `auto` | `cuda` / `cpu` to force |
| `YTRAG_WHISPER_BATCH` | `8` | `0` disables batching |
| `YTRAG_WHISPER_BEAM` | `5` | `1` is ~30% faster, greedy decoding |
| `YTRAG_CHUNK_SECONDS` | `75` | roughly one explained idea |
| `YTRAG_CHUNK_OVERLAP` | `15` | |
| `YTRAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | `BAAI/bge-m3` for the last 1/12 |
| `YTRAG_COLLECTION` | `dsa_lectures` | embedding dim is appended to the name |
| `QDRANT_URL` / `QDRANT_API_KEY` | unset | unset = local folder index |
| `YTRAG_TOP_K` | `6` | |
| `YTRAG_MAX_DISTANCE` | `0.6` | coarse pre-filter, see below |
| `YTRAG_CONFIDENT_DISTANCE` | `0.45` | below this the UI calls the top hit a solid match |
| `YTRAG_TITLE_BOOST` | `0.06` | distance bonus per matching title word; `0` disables |
| `YTRAG_LINK_REWIND` | `5` | citation links start this many seconds early |
| `YTRAG_LLM_BACKEND` | auto | `gemini` / `groq` / `none`, picked from whichever key is set |
| `YTRAG_RATE_LIMIT_REQUESTS` | `20` | per IP, per `YTRAG_RATE_LIMIT_WINDOW` (60 s) |
| `YTRAG_MAX_QUESTION_CHARS` | `500` | |

---

## Evaluation

[eval/golden.json](eval/golden.json) holds two kinds of entry.

**Retrieval.** A hit means the expected video appears in the top-k *and* at least one returned
chunk overlaps `expect_around_sec +/- tolerance_sec`:

```json
{"q": "memoization vs tabulation kya difference hai?",
 "expect_video_id": "abc123", "expect_around_sec": 724, "tolerance_sec": 120}
```

**Refusal.** A hit means the pipeline declines to answer:

```json
{"q": "React hooks kaise kaam karte hain?", "expect_refusal": true}
```

Run `uv run ytrag eval --verbose` for a hit rate and a list of misses. A golden set is the only
way to tell whether a change improved retrieval or just felt better on the one query you kept
re-testing.

### Tuning `MAX_DISTANCE`

The default is `0.6`, measured on the full 2933-chunk index against 20 in-syllabus and 10
off-topic questions. The two populations **overlap**: the worst genuine question ("number of
islands", 0.568) scores worse than the best off-topic one ("neural network backpropagation",
0.409), so no single cutoff separates them. Tightening it to 0.50 silently refused real
questions like "hashmap kab use karna chahiye", which is the worse failure, because the student
is told their own lecture doesn't exist.

So the cutoff is a **coarse pre-filter**. It removes the obviously unrelated and keeps every
genuine question; the model's own refusal, working from the excerpts, does the semantic
judgement. At this value: 20/20 real questions kept, 10/10 off-topic questions still refused.

Retrieval (`/search`) never touches an LLM. The written explanation is optional and only
configures the "explain" button.

---

## Failure modes this codebase is built around

**Don't use a generic text splitter.** `RecursiveCharacterTextSplitter` works on one
concatenated string and throws the timestamps away. The timestamp *is* the product, so
[ytrag/chunk.py](ytrag/chunk.py) chunks on the segment list in the time domain.

**Whisper loop hallucinations.** On silence or a music sting Whisper repeats the previous
phrase forever. `condition_on_previous_text=False` and `vad_filter=True` cut most of it;
`is_repetitive()` in `chunk.py` drops what survives. These chunks match everything and say
nothing.

**The transcript cache is the most important line in the codebase.** Re-transcribing a playlist
by accident costs hours. It is written atomically (temp file + `os.replace`) so a Ctrl-C
mid-write can't leave a truncated file that the cache then trusts forever.

**The timestamp should land slightly early.** Retrieval hits the chunk containing the answer,
but the explanation usually starts just before it, so citation URLs rewind 5 seconds.

**Ungrounded answers destroy trust.** Given junk context, an LLM will answer from its own
training and attach your timestamps to it. Three guards in [ytrag/answer.py](ytrag/answer.py):

1. drop everything above `MAX_DISTANCE`, and if nothing survives return the refusal
   **without calling the LLM at all**;
2. honour the model's own refusal;
3. an answer with zero citations is marked `grounded: false` and gets no links.

**Idempotency.** Cached transcripts, `upsert` not `add`, deterministic point IDs
(`uuid5` of `"{video_id}:{start_sec}"`), resume on partial failure.

---

## Deploying

Ship `api.main:app` to Render, Railway or Fly. Set `QDRANT_URL` and `QDRANT_API_KEY` for a
hosted index (and a Gemini or Groq key if you want explanations), then run `ytrag reindex`
against it. The collection name carries the embedding dimension, so a 384-dim and a 1024-dim
index can coexist in one cluster. The rate limiter is in-process, so move it to Redis before
running more than one worker.