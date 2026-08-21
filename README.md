# Fine-tuning Small Language Models for Literal Extraction in ChatBDI

Material from the bachelor's thesis in Computer Science by **Emanuele Borghi**,
University of Modena and Reggio Emilia — supervisor Prof. Angelo Ferrando.

The work replaces, inside the
[ChatBDI](https://github.com/VEsNA-ToolKit/chatbdi) framework, the
general-purpose model used to generate the logical term of a KQML message with
a small model (**Qwen3.5-4B**) fine-tuned via QLoRA.

## In short

A BDI agent communicates through logical terms, and recognises them only if
they are written **exactly** in the expected form: `booking(marco, suite, ...)`
triggers the corresponding plan, whereas `booking("marco", suite, ...)` — with
one quote too many — triggers nothing, and the interaction stops without any
error being reported.

Turning a user sentence into such a rigid term is, within ChatBDI, the step
that performs worst. This work assigns it to a four-billion-parameter model
specialised on the task, rather than to a general-purpose model instructed
anew in every prompt. The result runs locally, makes far fewer formal mistakes
and needs one ninth of the context.

## Repository layout

```
chatbdi-slm-finetuning/
│
├── src/
│   ├── pipeline/                        training set construction
│   │   ├── merge_data.py                consolidates the triples of 10 domains
│   │   ├── merge_literals.py            consolidates the reference literals
│   │   ├── augment_all_data.py          paraphrasing + structural validation
│   │   └── convert_to_jsonl.py          conversion to the ChatML format
│   │
│   └── prompts/                         instructions for the adapted model
│       ├── nl2log_short.txt             compact system prompt (901 chars)
│       ├── nl2logPrompt_short.txt       user template, zero-shot
│       ├── nl2logPrompt_short_examples.txt  user template, few-shot
│       └── nl2logPrompt_zero.txt        zero-shot template for the base model
│
├── notebooks/
│   ├── 01_Training_LoRA.ipynb           QLoRA fine-tuning on Colab (T4)
│   ├── 02_Benchmark.ipynb               comparison of the three configurations
│   └── conta_token.py                   measures example length in tokens
│
├── data/
│   ├── consolidated/                    aggregation of the per-domain data
│   │   ├── all_data.csv                 229 triples, 10 domains
│   │   ├── all_literals.csv             893 reference literals
│   │   └── all_data_augmented.csv       1199 triples after data augmentation
│   │
│   ├── training/
│   │   └── dataset_short_R2.jsonl       final training set, 9 domains
│   │
│   └── benchmark/                       8 test files, 21 cases each
│       ├── base_*.jsonl                 with the extended system prompt
│       └── ft_*.jsonl                   with the compact system prompt
│
└── results/
    ├── benchmark_completo_tesiR2.csv    252 measurements, one per row
    └── figures/                         the four comparison charts
```

In the benchmark file names, `ticket` denotes the domain held out from training
(*Out-of-Domain*), while `ignoto` denotes unseen sentences from the nine domains
represented in the training set (*In-Domain*).

The ChatBDI framework itself is **not included**: it is the work of other
authors and remains available in its original repository (see *Attribution*).

## Results

Comparison of three configurations — each consisting of a model together with
the system prompt it operates with — over 252 measurements.

| Configuration | Exact Match (zero-shot) | Exact Match (few-shot) | Prompt tokens (zero-shot) |
|---|---|---|---|
| Qwen2.5-Coder-7B + extended prompt | 14.3% | 11.9% | 2452 |
| Qwen3.5-4B + extended prompt | 19.0% | 47.6% | 2546 |
| **Qwen3.5-4B fine-tuned + compact prompt** | **52.4%** | **69.0%** | **282** |

Quoting errors — the class of defects that prevents logical unification on the
agent side — drop from 19 and 23 occurrences in the two baseline configurations
to a single one.

Raw per-measurement data is in `results/benchmark_completo_tesiR2.csv`.

## Setup

The pipeline scripts require Python 3.11 or later and the `openai` package,
used as the interface towards the generation service:

```bash
pip install openai
```

The notebooks are set up for Google Colab and install their own dependencies
at the top of the first cell (Unsloth, TRL, PEFT, bitsandbytes) — no local
setup is required to run them.

### API key

The data augmentation step calls a remote model. **The key is never stored in
the source code**: it is read from the environment, so that it cannot end up in
version control by accident.

Set it before running the script:

**macOS / Linux (bash or zsh)**

```bash
export GEMINI_API_KEY=your-key-here      # or OPENAI_API_KEY
```

To avoid retyping it in every session, append that line to your shell
configuration file (`~/.zshrc` on macOS, `~/.bashrc` on most Linux
distributions). That file lives in your home directory, outside this
repository, and therefore can never be committed.

**Windows (PowerShell)**

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

To persist it across sessions, set it once with
`setx GEMINI_API_KEY "your-key-here"` and open a new terminal, or add it under
*System Properties → Environment Variables*.

**Windows (Command Prompt)**

```cmd
set GEMINI_API_KEY=your-key-here
```

This only lasts for the current window; use `setx` as above to persist it.

**Any platform — `.env` file**

If you prefer a `.env` file inside the project, its safety depends entirely on
`.gitignore` remaining in place — the repository ships with `.env`, `*.key`
and `secrets*` already ignored. This protects any ordinary `git add`, but not
actions that bypass it (`git add -f`) or copying the folder outside of git
(zipping it, uploading it elsewhere). A shell/environment variable, kept
outside the project folder, is the safer option on any operating system.

No key is needed when pointing the script at a local endpoint through
`--base-url`.

## Pipeline, in order

1. **Consolidation** — `merge_data.py` and `merge_literals.py` gather the
   per-domain data into two overall collections, adding the source-domain
   column.

   ```bash
   python src/pipeline/merge_data.py
   python src/pipeline/merge_literals.py
   ```

2. **Data augmentation** — `augment_all_data.py` produces paraphrased variants
   of every example, validating each one structurally against the reference
   literals: functor, arity and the type of each argument by position.
   Non-conforming variants are discarded.

   ```bash
   export GEMINI_API_KEY=...
   python src/pipeline/augment_all_data.py \
       --input all_data.csv --model gemini-3.1-flash-lite --per-row 5
   ```

3. **Conversion** — `convert_to_jsonl.py` turns the triples into the
   three-role conversational format, in four modes: extended or compact system
   prompt, with or without examples in the user message.

   ```bash
   python src/pipeline/convert_to_jsonl.py --short             # compact, zero-shot
   python src/pipeline/convert_to_jsonl.py --short --examples  # compact, few-shot
   ```

4. **Fine-tuning** — `notebooks/01_Training_LoRA.ipynb`, on Google Colab with a
   16 GB T4 GPU. 4-bit QLoRA, rank 16, α = 32, four epochs, checkpoint
   selection on validation loss.

5. **Evaluation** — `notebooks/02_Benchmark.ipynb` compares the three
   configurations across two domain conditions and two prompting strategies,
   producing the CSV and the charts under `results/`.

`notebooks/conta_token.py` is an auxiliary utility that measures the length in
tokens of the training examples.

## A note on the data

The per-domain data everything derives from — natural language sentences,
performatives and expected logical terms across ten application domains — was
prepared by the ChatBDI authors to evaluate the system, and is **not the work
of this thesis**. The files under `data/consolidated/` are the aggregation of
that data produced by the scripts in this repository;
`all_data_augmented.csv` additionally contains the paraphrased variants.

The *tickets* domain is deliberately excluded from the training set and used as
an unseen domain at evaluation time: `data/training/` therefore covers nine
domains, while `data/benchmark/` also includes the tenth.

## Attribution

**From this thesis**: the scripts in `src/pipeline/`, the prompts in
`src/prompts/`, the notebooks in `notebooks/`, the datasets in
`data/training/` and `data/benchmark/`, the aggregation in
`data/consolidated/`, and everything under `results/`.

**From the ChatBDI project** (Andrea Gatti, Viviana Mascardi, Angelo Ferrando):
the framework, the Jason interpreter, the embedding space, the performative
classifier, the original system prompts, and the per-domain data from which the
files in `data/consolidated/` derive. None of these components is included
here; the project is available at
<https://github.com/VEsNA-ToolKit/chatbdi>.

Reference for the framework:

> A. Gatti, V. Mascardi, A. Ferrando. *Let Me Talk to you! Natural Language
> Interaction between Humans and BDI Agents via ChatBDI*. 28th European
> Conference on Artificial Intelligence (ECAI 2025).
