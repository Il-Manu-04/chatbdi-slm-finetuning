# Fine-tuning di Small Language Models per l'estrazione di literal in ChatBDI

Materiale della tesi di laurea triennale in Informatica di **Emanuele Borghi**,
Università degli Studi di Modena e Reggio Emilia — relatore prof. Angelo Ferrando.

Il lavoro sostituisce, all'interno del framework
[ChatBDI](https://github.com/VEsNA-ToolKit/chatbdi), il modello generalista
impiegato per generare il termine logico di un messaggio KQML con un modello di
piccole dimensioni (**Qwen3.5-4B**) sottoposto a fine-tuning mediante QLoRA.

## In breve

Un agente BDI comunica tramite termini logici, e li riconosce solo se sono
scritti **esattamente** nella forma attesa: `booking(marco, suite, ...)` attiva
il piano corrispondente, `booking("marco", suite, ...)` — con un apice di
troppo — non attiva nulla e l'interazione si interrompe senza segnalazioni.

Tradurre la frase di un utente in un termine così rigido è, nel framework
ChatBDI, il passo che riesce peggio. Questo lavoro lo affida a un modello di
quattro miliardi di parametri specializzato sul compito, anziché a un modello
generalista istruito di volta in volta nel prompt. Il risultato è un modello
che gira in locale, sbaglia molto meno la forma dei termini e richiede un nono
del contesto.

## Struttura del repository

```
chatbdi-slm-finetuning/
│
├── src/
│   ├── pipeline/                        costruzione del training set
│   │   ├── merge_data.py                consolida le triple dei 10 domini
│   │   ├── merge_literals.py            consolida i literal di riferimento
│   │   ├── augment_all_data.py          parafrasi + validazione strutturale
│   │   └── convert_to_jsonl.py          conversione nel formato ChatML
│   │
│   └── prompts/                         istruzioni per il modello adattato
│       ├── nl2log_short.txt             system prompt compatto (901 caratteri)
│       ├── nl2logPrompt_short.txt       template utente, zero-shot
│       ├── nl2logPrompt_short_examples.txt  template utente, few-shot
│       └── nl2logPrompt_zero.txt        template zero-shot per il modello base
│
├── notebooks/
│   ├── 01_Training_LoRA.ipynb           fine-tuning QLoRA su Colab (T4)
│   ├── 02_Benchmark.ipynb               confronto fra le tre configurazioni
│   └── conta_token.py                   misura la lunghezza degli esempi
│
├── data/
│   ├── consolidated/                    aggregazione dei dati per dominio
│   │   ├── all_data.csv                 229 triple, 10 domini
│   │   ├── all_literals.csv             893 literal di riferimento
│   │   └── all_data_augmented.csv       1199 triple dopo la data augmentation
│   │
│   ├── training/
│   │   └── dataset_short_R2.jsonl       training set finale, 9 domini
│   │
│   └── benchmark/                       8 file di test: 21 casi ciascuno
│       ├── base_*.jsonl                 con system prompt esteso
│       └── ft_*.jsonl                   con system prompt compatto
│
└── results/
    ├── benchmark_completo_tesiR2.csv    252 misurazioni, una per riga
    └── figures/                         i quattro grafici di confronto
```

Nei nomi dei file di benchmark, `ticket` indica il dominio escluso
dall'addestramento (*Out-of-Domain*) e `ignoto` le frasi mai viste appartenenti
ai nove domini rappresentati nel training set (*In-Domain*).

Il framework ChatBDI **non è incluso**: è opera di altri autori e resta
disponibile nel repository originale (si veda la sezione *Attribuzione*).

## Risultati principali

Confronto fra tre configurazioni, ciascuna composta da un modello e dal system
prompt con cui opera, su 252 misurazioni.

| Configurazione | Exact Match (Zero-Shot) | Exact Match (Few-Shot) | Token di prompt (Zero-Shot) |
|---|---|---|---|
| Qwen2.5-Coder-7B + prompt esteso | 14,3% | 11,9% | 2452 |
| Qwen3.5-4B + prompt esteso | 19,0% | 47,6% | 2546 |
| **Qwen3.5-4B fine-tuned + prompt compatto** | **52,4%** | **69,0%** | **282** |

Gli errori di virgolettatura — la classe di difetti che impedisce
l'unificazione logica lato agente — passano da 19 e 23 occorrenze nelle due
configurazioni di confronto a una sola.

I dati grezzi di ogni singola misurazione sono in
`results/benchmark_completo_tesiR2.csv`.

## La pipeline, in ordine

1. **Consolidamento** — `merge_data.py` e `merge_literals.py` riuniscono i dati
   organizzati per dominio in due raccolte complessive, aggiungendo la colonna
   del dominio di provenienza.

   ```bash
   python src/pipeline/merge_data.py
   python src/pipeline/merge_literals.py
   ```

2. **Data augmentation** — `augment_all_data.py` produce varianti per parafrasi
   di ciascun esempio, validandole strutturalmente contro i literal di
   riferimento: functor, arità e tipo di ciascun argomento in base alla
   posizione. Le varianti non conformi sono scartate.

   ```bash
   export OPENAI_API_KEY=...        # chiave del servizio impiegato
   python src/pipeline/augment_all_data.py \
       --input all_data.csv --model gemini-3.1-flash-lite --per-row 5
   ```

3. **Conversione** — `convert_to_jsonl.py` traduce le triple nel formato
   conversazionale a tre ruoli, in quattro modalità: system prompt esteso o
   compatto, con o senza esempi nel messaggio dell'utente.

   ```bash
   python src/pipeline/convert_to_jsonl.py --short             # compatto, zero-shot
   python src/pipeline/convert_to_jsonl.py --short --examples  # compatto, few-shot
   ```

4. **Fine-tuning** — `notebooks/01_Training_LoRA.ipynb`, su Google Colab con
   GPU T4 da 16 GB. QLoRA a 4 bit, rango 16, α = 32, quattro epoche, selezione
   del checkpoint sulla validation loss.

5. **Valutazione** — `notebooks/02_Benchmark.ipynb` confronta le tre
   configurazioni su due condizioni di dominio e due strategie di prompting,
   producendo il CSV e i grafici in `results/`.

`notebooks/conta_token.py` è un'utilità accessoria che misura la lunghezza in
token degli esempi di addestramento.

## Nota sui dati

I dati per dominio da cui tutto discende (frasi in linguaggio naturale,
performativo e termine logico atteso, su dieci domini applicativi) sono stati
predisposti dagli autori di ChatBDI per la valutazione del sistema, e **non
sono opera di questa tesi**. I file in `data/consolidated/` ne costituiscono
l'aggregazione prodotta dai programmi di questo repository; `all_data_augmented.csv`
contiene inoltre le varianti generate per parafrasi.

Il dominio *tickets* è deliberatamente escluso dal training set e impiegato
come dominio non osservato in fase di valutazione: `data/training/` comprende
pertanto nove domini, mentre `data/benchmark/` contiene anche il decimo.

## Attribuzione

**Di questa tesi**: i programmi in `src/pipeline/`, i prompt in `src/prompts/`,
i notebook in `notebooks/`, i dataset in `data/training/` e `data/benchmark/`,
l'aggregazione in `data/consolidated/` e tutto il contenuto di `results/`.

**Del progetto ChatBDI** (Andrea Gatti, Viviana Mascardi, Angelo Ferrando): il
framework, l'interprete Jason, lo spazio degli embedding, il classificatore del
performativo, i system prompt originari e i dati per dominio da cui derivano i
file in `data/consolidated/`. Nessuno di questi componenti è incluso qui; il
progetto è disponibile all'indirizzo
<https://github.com/VEsNA-ToolKit/chatbdi>.

Riferimento bibliografico del framework:

> A. Gatti, V. Mascardi, A. Ferrando. *Let Me Talk to you! Natural Language
> Interaction between Humans and BDI Agents via ChatBDI*. 28th European
> Conference on Artificial Intelligence (ECAI 2025).

## Requisiti

I programmi della pipeline richiedono Python 3.11 o successivo e il pacchetto
`openai`, impiegato come interfaccia verso il servizio di generazione. I
notebook sono predisposti per Google Colab e installano autonomamente le
dipendenze necessarie (Unsloth, TRL, PEFT, bitsandbytes).
