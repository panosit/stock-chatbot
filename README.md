# Investing Opinions Chatbot (LoRA fine-tune of Llama 3.2 1B)

A small LoRA fine-tuning project that nudges Meta's `Llama-3.2-1B-Instruct`
toward a specific, personal set of investing opinions and rules — which
stocks to buy, which sectors to avoid, how to think about debt, dividend
streaks, and some verified balance-sheet facts about the recommended
companies.

This is a personal/learning project, **not financial advice**, and the
dataset encodes one person's opinions presented as fact. See
[Limitations](#limitations-read-this) before trusting any output.

## What this actually is

- **Base model:** `meta-llama/Llama-3.2-1B-Instruct` (open-weight, ~1B
  parameters, small enough to fine-tune on a single consumer GPU).
- **Method:** [LoRA](https://arxiv.org/abs/2106.09685) via
  [`peft`](https://github.com/huggingface/peft) — only a small set of
  additional weights (attached to the attention projections) are trained;
  the base model's own weights stay frozen. The result is a lightweight
  adapter, not a new full model.
- **Dataset:** `data/sample_dataset.jsonl`, a small, hand-curated set of
  instruction/output pairs encoding one person's investing rules and some
  verified financial facts (see [Dataset](#dataset) below).
- **Training/inference code:** all in a single file, `main.py`, with three
  subcommands: `train`, `infer`, and `chat`.

## Requirements

- **An NVIDIA GPU with CUDA.** The script calls `torch.cuda.is_available()`
  and raises a `RuntimeError` immediately if no CUDA GPU is found — it will
  not silently fall back to CPU or another accelerator (e.g. Apple Silicon,
  AMD ROCm). You need a real CUDA-capable machine to run `train`, `infer`,
  or `chat`.
- Python 3.10+ (developed against 3.14).
- Dependencies in `requirements.txt` (no pinned versions — see the note in
  that file's history if you want reproducible installs):

  ```
  torch
  transformers
  peft
  datasets
  accelerate
  ```

- Access to the gated `meta-llama/Llama-3.2-1B-Instruct` model on Hugging
  Face. You'll need to:
  1. Request access on the
     [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
     (approval is usually fast).
  2. Authenticate locally, e.g. `huggingface-cli login`, or set the
     `HF_TOKEN` environment variable.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # one-time, needed for the gated Llama model
```

## Usage

### Train

```bash
python main.py train
```

- Loads the base model in bf16 onto the GPU, wraps it with a LoRA adapter
  (`r=16`, `lora_alpha=32`, dropout `0.05`, applied to `q_proj`, `k_proj`,
  `v_proj`, `o_proj`).
- Loads `data/sample_dataset.jsonl`, formats each example using Llama
  3.2's real chat template (via `tokenizer.apply_chat_template`), and
  trains for 3 epochs.
- Saves the resulting LoRA adapter (not the full model) to
  `output/llama-3.2-1b-lora/`.

### Compare base vs. fine-tuned on one question

```bash
python main.py infer "What are some good stocks to invest in?"
```

Prints the base model's answer and the fine-tuned model's answer side by
side, so you can see what the fine-tune actually changed.

### Chat interactively

```bash
python main.py chat
```

Loads the base model plus the trained LoRA adapter once, then gives you a
simple REPL: type a question, get an answer, repeat. Type `quit` or
`exit` (or Ctrl+C) to stop.

## Dataset

`data/sample_dataset.jsonl` — plain JSONL, one JSON object per line, each
with exactly two keys:

```json
{"instruction": "What are some good stocks to invest in?", "output": "Some good stocks to consider include Apple, Nvidia, ..."}
```

Currently **79 examples**, covering:

- **The recommended stock list:** Apple, Nvidia, Microsoft, Google
  (Alphabet), Amazon, Meta, TSMC, Walmart, Visa, Exxon Mobil, Johnson &
  Johnson, Procter & Gamble.
- **Sectors/stocks to avoid:** car companies, clothing, beverages, fast
  food, banks, insurance, telecoms, most semiconductor makers (TSMC and
  Nvidia are the exceptions), and a list of specific named stocks (e.g.
  Netflix, Tesla, Uber, IBM, Shopify).
- **Screening rules for what counts as investable:**
  - Minimum market cap of **$100 billion**, and the company must rank in
    the **top 60 companies by market cap**.
  - Country screen: a Human Development Index (HDI) of at least 0.9 as a
    first filter, with the stricter rule of investing only in the USA
    stock market (TSMC is the sole non-US exception, due to its market
    cap and US listing).
  - Bubble-risk flag: any stock with a P/E ratio over 100.
- **A debt philosophy:** personal/consumption debt is to be avoided
  (pay cash for houses, cars, etc.), while business debt used to generate
  more income is considered productive.
- **A bonds rule:** bonds are not recommended until net worth exceeds $1
  million; below that, stay in stocks.
- **Dividend-growth streak facts** for each recommended stock (e.g.
  Procter & Gamble 70 consecutive years, Amazon pays no dividend at all).
- **Verified balance-sheet facts** (total assets, net income, liabilities,
  shareholders' equity) for a sample of the recommended companies, with
  explanations of *why* the numbers moved the way they did (AI-driven
  growth, corporate spinoffs, acquisitions, commodity-price cycles, etc.),
  not just raw figures.

All financial figures were looked up from public sources
([stockanalysis.com](https://stockanalysis.com)) rather than invented, and
TSMC's TWD-denominated figures were converted to USD using real historical
year-end exchange rates. Where a figure looked anomalous (e.g. Alphabet's
trailing-twelve-month net income), that's flagged in the dataset text
rather than stated as fact.

### Adding more examples

Keep the same two-key JSON schema (`instruction`, `output`) on each line,
and validate the file parses correctly before training, e.g.:

```bash
python -c "
import json
with open('data/sample_dataset.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        d = json.loads(line)
        assert set(d.keys()) == {'instruction', 'output'}, i
print('ok')
"
```

## How the fine-tuning actually affects answers

- `main.py infer "..."` prints the **base model's** answer and the
  **fine-tuned model's** answer for the same question, so you can see the
  difference directly.
- Questions phrased similarly to what's in the dataset should reliably
  reflect the trained opinions (e.g. the recommended stock list, the
  market-cap rule, the bonds-at-$1M rule).
- Questions phrased very differently, or about topics not covered at all,
  may fall back to the base model's general pretrained behavior — LoRA
  does not erase the base model's existing knowledge, it only nudges it.

## Limitations (read this)

This is a small personal project, not a production system or real
financial-advice tool. Some things worth being honest about:

- **The dataset is small.** 79 examples is enough to bias a small model
  toward repeating a fixed set of opinions and facts, but it is not enough
  for the model to robustly generalize to novel phrasings or stocks that
  aren't in the dataset.
- **Fine-tuning is unreliable for exact facts.** LoRA nudges weights
  probabilistically; it does not store numbers the way a database does.
  Don't trust the fine-tuned model's exact dollar figures or percentages
  without checking them — if you need guaranteed-accurate numbers, pair
  this with a retrieval step (e.g. a lookup table or RAG) instead of
  relying on the model's memory.
- **The content is one person's opinion, stated as fact.** The dataset
  encodes dogmatic, absolute investing rules (e.g. "avoid all car company
  stocks," "never invest below a $100B market cap"). That's fine for a
  private learning project, but this should not be exposed to other users
  as if it were objective financial advice.
- **No held-out evaluation set.** The script trains on 100% of the
  dataset with no validation split, so there's no automated way to tell
  if a given training run overfit or hurt general-purpose answer quality
  — you have to check manually with `infer` or `chat`.
- **CUDA is required, with no fallback.** If you don't have an NVIDIA GPU,
  none of the three commands will run as-is.

## Project structure

```
.
├── main.py                      # train / infer / chat — all in one file
├── requirements.txt             # Python dependencies
├── data/
│   └── sample_dataset.jsonl     # training data (instruction/output pairs)
└── output/
    └── llama-3.2-1b-lora/       # saved LoRA adapter (created by `train`)
```
