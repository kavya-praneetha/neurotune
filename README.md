# ai-lab

Your AI + agent development learning environment. **All CPU** — this machine has
an AMD integrated GPU (2 CUs, not a compute target), so there's no CUDA/ROCm.
The 16-core Zen 5 + 92 GB RAM carries local models on CPU instead, which is
plenty for learning and small-to-mid models.

## Setup (one-time)
```bash
cp .env.example .env         # then fill in only what you need
```
`.env` is gitignored and is injected at runtime rather than baked into the
Docker image, so keys never reach a published layer.

| Variable | Needed by | Required? | Where to get it |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | `agents/hello_agent.py` | for that script only | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `OPENAI_API_KEY` | only if you swap an example to an OpenAI model | optional | [platform.openai.com](https://platform.openai.com/api-keys) |

**`neurotune/` needs no keys at all.** The whole pipeline — preprocessing,
LOSO training, recommendation, RAG explanation — runs offline on simulated
data, so you can clone this repo and run it without signing up for anything:

```bash
# fastest end-to-end demo: recommendation + RAG-grounded rationale (~10 s)
uv run python -m neurotune.cli explain --subjects 3 --sessions 1 --epochs 1
```

Measured on a 16-core Zen 5 CPU:

| Command | Runtime |
|---|---|
| `explain --subjects 3 --sessions 1 --epochs 1` | ~9 s |
| `recommend --subjects 3 --sessions 1` | ~150 s |
| `run-all --subjects 3 --sessions 2 --epochs 1` | **>10 min** |

`run-all` is slow because it invokes the five stages as independent commands
(`cli.py:230`), so the simulation, ICA and STFT preprocessing is repeated once
per stage rather than computed once and shared. Run individual stages while
iterating.

Minimum cohort for `detect`: **3 subjects** at 1 session, or 2 at 2 sessions.
LOSO holds one subject out, and the validation split needs at least 2
subject-sessions in what remains — fewer fails partway through with
`need at least 2 subject-sessions to form a validation split`.

Local models via Ollama also need **no** key.

## Run things
Everything runs through `uv` — no `pip install`, no activating venvs:

```bash
cd ~/ai-lab

# Jupyter — start here
uv run jupyter lab                       # then open notebooks/00_start_here.ipynb

# Local model, offline, free (runs on CPU)
uv run python agents/local_llm.py

# A tool-using agent on Claude (needs ANTHROPIC_API_KEY)
uv run python agents/hello_agent.py/

# Reach Jupyter from another machine (tunnel, then use the URL it prints)
ssh -L 8888:localhost:8888 <user>@<host>
uv run jupyter server list        # prints http://<host>:8888/?token=<your-token>
```

## Or run it in Docker (no local Python needed)
`.venv/` is never shared or committed — it's 2.8 GB of machine-specific
binaries. `pyproject.toml` + `uv.lock` rebuild the identical 285-package
environment anywhere, and the image pins the OS and Python build on top:

```bash
docker compose up --build          # first build pulls ~1 GB of CPU wheels
docker compose logs lab            # copy the Jupyter token it prints
```
Then open `http://localhost:8888` and paste the token.

Your keys stay in `.env` and are injected at runtime — `.dockerignore` keeps
that file out of the build context, so they never land in an image layer.

Note: `agents/local_llm.py` won't reach Ollama from inside the container —
Ollama is bound to `127.0.0.1` on the host, and `localhost` inside a container
means the container. See the comments at the bottom of `docker-compose.yml`.

## Local models (Ollama)
```bash
ollama list                  # what you have
ollama pull qwen2.5:14b      # bigger model — your RAM can hold it
ollama run llama3.2:3b       # chat in the terminal
```
Check `ollama ps` — the PROCESSOR column shows `100% CPU` here (expected).

## What's installed
- **Notebooks/SDKs:** jupyterlab, anthropic, openai, pandas, numpy, matplotlib
- **Agents:** langgraph, langchain (+anthropic/openai), pydantic-ai
- **ML/DL:** torch (CPU), transformers, datasets, scikit-learn, sentence-transformers
- **Local inference:** Ollama + llama3.2:3b

## Add a package
```bash
uv add <package>             # updates pyproject.toml + lockfile
```
