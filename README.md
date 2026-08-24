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
│   ├── conta_token.py                   measures example length in tokens
│   └── analisi_errori.py                reproduces the error-distribution table
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

   These two scripts read a `src/pipeline/input/<domain>/` tree that belongs to
   the ChatBDI project and is **not redistributed here**; they are included for
   reference, and their output is already provided under `data/consolidated/`.
   Point them at your own per-domain tree to use them.

2. **Data augmentation** — `augment_all_data.py` produces paraphrased variants
   of every example, validating each one structurally against the reference
   literals: functor, arity and the type of each argument by position.
   Non-conforming variants are discarded.

   This step runs on the data shipped here. Give it explicit paths — the
   built-in defaults point next to the script, not into `data/`:

   ```bash
   export GEMINI_API_KEY=...
   python src/pipeline/augment_all_data.py \
       --input    data/consolidated/all_data.csv \
       --literals data/consolidated/all_literals.csv \
       --output   data/consolidated/all_data_augmented.csv \
       --model gemini-3.1-flash-lite --per-row 5
   ```

   Useful flags: `--limit N` to try it on a handful of rows first, `--start N`
   to resume an interrupted run, `--base-url` to point at a local endpoint
   instead of a remote service.

3. **Conversion** — `convert_to_jsonl.py` turns the triples into the
   three-role conversational format, in four modes: extended or compact system
   prompt, with or without examples in the user message.

   ```bash
   python src/pipeline/convert_to_jsonl.py --short             # compact, zero-shot
   python src/pipeline/convert_to_jsonl.py --short --examples  # compact, few-shot
   ```

   The script reads the prompt files from the ChatBDI interpreter's
   `modelfiles/` directory, at a path relative to its own location. To run it
   from this repository, either place it inside a ChatBDI checkout or edit the
   `MODELFILES` constant at the top of the file to point at `src/prompts/`.
   Note that the two *extended* modes additionally need `nl2log.txt`, the
   original system prompt, which belongs to the ChatBDI project and is not
   included here; the two compact modes need only the files under
   `src/prompts/`. The dataset the thesis actually trained on —
   `data/training/dataset_short_R2.jsonl` — is the output of the compact,
   zero-shot mode and is provided ready to use.

4. **Fine-tuning** — `notebooks/01_Training_LoRA.ipynb`, on Google Colab with a
   16 GB T4 GPU. 4-bit QLoRA, rank 16, α = 32, four epochs, checkpoint
   selection on validation loss.

5. **Evaluation** — `notebooks/02_Benchmark.ipynb` compares the three
   configurations across two domain conditions and two prompting strategies,
   producing the CSV and the charts under `results/`.

Two auxiliary scripts accompany the pipeline. `notebooks/conta_token.py`
measures the length in tokens of the training examples.
`notebooks/analisi_errori.py` reproduces, from the raw results, the
error-distribution table reported in the thesis: run it from the repository
root, with no dependencies beyond the standard library.

```bash
python notebooks/analisi_errori.py
```

It prints the table three ways: each wrong output assigned to its single most
severe defect (the table as reported), every defect counted wherever it occurs,
and how often an output carries more than one defect at once.

## Using this with a different model

Nothing here is specific to Qwen3.5-4B. The pipeline takes any causal language
model that Unsloth can load, and the steps below are what you actually need to
change.

### 1. Choosing the model to fine-tune

The model is not a file you drop into the repository: it is a Hugging Face
identifier, downloaded at runtime. Open `notebooks/01_Training_LoRA.ipynb` and
edit one line:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3.5-4B",   # <- your model here
    max_seq_length = 512,
    load_in_4bit   = True,
)
```

Any Unsloth-compatible identifier works, for example
`unsloth/Llama-3.2-3B-Instruct` or `unsloth/mistral-7b-instruct-v0.3`. Names
carrying the `bnb-4bit` suffix ship already quantised and load faster; the
others are quantised on the fly, with the same end result. A local path to a
model you downloaded yourself also works.

Two settings deserve a second look when you change model — both inside
`notebooks/01_Training_LoRA.ipynb`.

`max_seq_length` is dimensioned on the examples used here (a sentence, a
schema and a short term): raise it if your prompts are longer, and measure
first with `notebooks/conta_token.py` rather than guessing. It appears
**twice** in the notebook and both must be changed together, or training and
loading disagree on the sequence length: once where the model is loaded
(shown above) and once a few cells down, where the trainer is configured —
`SFTTrainer(..., max_seq_length = 512, ...)`.

`r` and `lora_alpha`, set a few lines below the model loading in the same
cell (inside `FastLanguageModel.get_peft_model(...)`) — 16 and 32
here — control how much capacity the adapters have; larger models tolerate the
same values, much smaller ones may need a higher rank.

### 2. Where the files go

The notebooks run on Google Colab and read from Google Drive. Create one
folder in your Drive and upload into it, flat, the files this repository
provides:

```
MyDrive/<your-folder>/
├── dataset_short_R2.jsonl        from data/training/     (fine-tuning)
├── ft_ignoto_zero.jsonl          from data/benchmark/    (evaluation)
├── ft_ignoto_few.jsonl                  "
├── ft_ticket_zero.jsonl                 "
├── ft_ticket_few.jsonl                  "
├── base_ignoto_zero.jsonl               "
├── base_ignoto_few.jsonl                "
├── base_ticket_zero.jsonl               "
└── base_ticket_few.jsonl                "
```

Then set the folder name at the top of both notebooks — `dataset_path` in
`01_Training_LoRA.ipynb`, `DRIVE_FOLDER` in `02_Benchmark.ipynb`. Training
writes its checkpoints, the LoRA adapters and a GGUF export back into the same
folder.

### 3. Running the evaluation

`02_Benchmark.ipynb` reads the eight test files by name and produces one row
per case in `benchmark_completo_tesiR2.csv`. Which model gets which files is
not arbitrary:

- the `ft_*` files carry the **compact** system prompt and are meant for a
  fine-tuned model, which has already absorbed the conventions;
- the `base_*` files carry the **extended** system prompt and are meant for a
  model that has not been fine-tuned, and therefore needs the conventions
  spelled out in the prompt.

Each pair covers the two domain conditions (`ignoto` = held-in domains, unseen
sentences; `ticket` = the domain excluded from training) and the two prompting
strategies (`zero` and `few`). To add a model to the comparison, append an
entry to the model list in the notebook with its identifier and the file set it
should be measured on; the charts adapt to the number of models found in the
CSV. Generation is greedy (`do_sample=False`), so repeated runs give identical
outputs.

### 4. Building a dataset for your own domain

To retarget the whole thing at a different application domain you need two CSV
files, semicolon-separated, in the shape of the ones under
`data/consolidated/`:

| file | columns |
|---|---|
| `all_data.csv` | `sentence` · `performative` · `embedding` · `solution` · `domain` |
| `all_literals.csv` | `literal` · `functor` · `domain` |

`solution` is the expected logical term, `embedding` the retrieved schema, and
`all_literals.csv` is what the augmentation validates against — so its terms
must be correct: an error there propagates into the training set. From those
two files, run steps 2 and 3 above to obtain a training set in the same format
as `dataset_short_R2.jsonl`, then point the training notebook at it.

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
