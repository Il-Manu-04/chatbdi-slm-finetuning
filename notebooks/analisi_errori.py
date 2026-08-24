# =============================================================================
# ANALISI DEGLI ERRORI DEL BENCHMARK
# =============================================================================
# Riproduce la tabella di distribuzione degli errori riportata nella tesi
# (Capitolo 5, "Analisi degli errori") a partire dai dati grezzi.
#
# COME USARLO:
#   Dalla radice del repository:
#       python notebooks/analisi_errori.py
#
# Sono richiesti soltanto la libreria standard di Python e i file gia'
# presenti nel repository: il CSV dei risultati e i quattro test set.
#
# COSA PRODUCE:
#   A) l'attribuzione ESCLUSIVA - ogni uscita errata e' assegnata a una sola
#      categoria, la prima applicabile nell'ordine di precedenza. E' la
#      tabella riportata nella tesi.
#   B) il conteggio NON ESCLUSIVO - ogni difetto e' contato ovunque compaia,
#      anche quando piu' difetti coesistono nella medesima uscita. Serve a
#      contestualizzare la prima: le colonne non sommano al totale.
#   C) quante uscite errate presentano piu' di un difetto.
#
# ORDINE DI PRECEDENZA (dal difetto piu' grave al meno grave):
#   uscita non interpretabile > functor difforme > arita' difforme >
#   virgolettatura > capitalizzazione > variabili o placeholder > valore errato
#
# La graduatoria segue il momento in cui il difetto impedisce l'elaborazione:
# un'uscita non interpretabile non raggiunge l'unificazione, un functor
# difforme non incontra alcun piano, e cosi' via fino al valore errato, che
# lascia il termine formalmente valido.
# =============================================================================

import csv
import json
import re
import collections
import os

# ── Percorsi (relativi alla radice del repository) ───────────────────────────
CSV_RISULTATI = os.path.join("results", "benchmark_completo_tesiR2.csv")
DIR_BENCHMARK = os.path.join("data", "benchmark")

# I quattro test set del modello fine-tuned: contengono le medesime risposte
# attese di quelli a system prompt esteso, in forma piu' compatta.
TEST_SET = {
    ("In-Domain", "Zero-Shot"): "ft_ignoto_zero.jsonl",
    ("In-Domain", "Few-Shot"): "ft_ignoto_few.jsonl",
    ("Out-of-Domain", "Zero-Shot"): "ft_ticket_zero.jsonl",
    ("Out-of-Domain", "Few-Shot"): "ft_ticket_few.jsonl",
}

MODELLI = [
    "Qwen2.5-Coder-7B (Base)",
    "Qwen3.5-4B (Base)",
    "Qwen3.5-4B (Fine-Tuned)",
]

CATEGORIE = [
    "Uscita non interpretabile",
    "Functor difforme",
    "Arita difforme",
    "Virgolettatura",
    "Capitalizzazione",
    "Variabili o placeholder",
    "Valore errato",
]

# Ordine di precedenza per i difetti a livello di argomento.
PRECEDENZA = [
    ("quote", "Virgolettatura"),
    ("case", "Capitalizzazione"),
    ("var", "Variabili o placeholder"),
    ("valore", "Valore errato"),
]


# ── Riparazione e analisi dell'uscita ────────────────────────────────────────
# Le due funzioni seguenti riproducono il trattamento applicato dal notebook
# di benchmark, cosi' che l'analisi operi sulle medesime strutture.

def ripara(testo):
    """Sostituzioni testuali applicate alle uscite non interpretabili."""
    testo = re.sub(r'([a-zA-Z0-9_]+)\s*=', r'"\1": ', testo)
    testo = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', testo)
    testo = re.sub(r'("arg\d+":\s*)"([^"]*)"', r'\1"\\"\2\\""', testo)
    testo = re.sub(r':\s*([^"\d\s\[{][^,}]*?)\s*([,}])', r': "\1"\2', testo)
    return testo


def interpreta(testo):
    """Restituisce l'oggetto strutturato, o None se l'uscita non lo e'."""
    testo = re.sub(r"^```(?:json)?|```$", "", testo.strip(), flags=re.M).strip()
    testo = re.sub(r"<think>.*?</think>", "", testo, flags=re.S).strip()
    for candidato in (testo, ripara(testo)):
        try:
            return json.loads(candidato)
        except json.JSONDecodeError:
            pass
    return None


# ── Classificazione del difetto di un singolo argomento ──────────────────────

def e_variabile(v):
    s = str(v).strip()
    return s == "_" or (s != "" and s[0].isupper() and s.replace("_", "").isalpha())


def e_virgolettato(v):
    return str(v).strip().startswith('"')


def senza_apici(v):
    return str(v).strip().strip('"')


def difetto(prodotto, atteso):
    """Tipo di difetto di un argomento, o None se coincide con l'atteso.

    Il confronto e' tipizzato: la stringa "26" e il numero 26 sono valori
    distinti, e la loro discordanza costituisce un errore di virgolettatura.
    """
    if prodotto == atteso:
        return None
    if e_variabile(prodotto) and e_variabile(atteso):
        return None  # la normalizzazione rende equivalenti "_" e "X"
    if e_variabile(prodotto) != e_variabile(atteso):
        return "var"
    if senza_apici(prodotto) == senza_apici(atteso):
        return "quote"
    if (e_virgolettato(prodotto) == e_virgolettato(atteso)
            and senza_apici(prodotto).lower() == senza_apici(atteso).lower()):
        return "case"
    return "valore"


# ── Analisi ──────────────────────────────────────────────────────────────────

def carica_attesi():
    attesi = {}
    for (dominio, strategia), nome in TEST_SET.items():
        percorso = os.path.join(DIR_BENCHMARK, nome)
        with open(percorso, encoding="utf-8") as f:
            for i, riga in enumerate(f, 1):
                messaggi = json.loads(riga)["messages"]
                risposta = [m["content"] for m in messaggi if m["role"] == "assistant"][0]
                attesi[(dominio, strategia, str(i))] = risposta
    return attesi


def analizza():
    attesi = carica_attesi()
    esclusiva = collections.defaultdict(collections.Counter)
    non_esclusiva = collections.defaultdict(collections.Counter)
    uscite_errate = collections.Counter()
    piu_difetti = collections.Counter()

    with open(CSV_RISULTATI, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Exact_Match"].strip() in ("1", "True", "true"):
                continue
            modello = r["Modello"]
            uscite_errate[modello] += 1

            atteso = json.loads(attesi[(r["Dominio"], r["Strategia"], r["Test_ID"])])
            prodotto = interpreta(r["Output_Grezzo"])

            # Categorie che precedono l'esame dei singoli argomenti.
            if prodotto is None:
                esclusiva[modello]["Uscita non interpretabile"] += 1
                non_esclusiva[modello]["Uscita non interpretabile"] += 1
                continue
            if str(prodotto.get("functor", "")).strip() != str(atteso.get("functor", "")).strip():
                esclusiva[modello]["Functor difforme"] += 1
                non_esclusiva[modello]["Functor difforme"] += 1
                continue
            arg_prodotti = [k for k in prodotto if k != "functor"]
            arg_attesi = [k for k in atteso if k != "functor"]
            if len(arg_prodotti) != len(arg_attesi):
                esclusiva[modello]["Arita difforme"] += 1
                non_esclusiva[modello]["Arita difforme"] += 1
                continue

            # Difetti presenti fra gli argomenti.
            presenti = set()
            for k in arg_attesi:
                d = "valore" if k not in prodotto else difetto(prodotto[k], atteso[k])
                if d:
                    presenti.add(d)
            if len(presenti) > 1:
                piu_difetti[modello] += 1

            for tipo, nome in PRECEDENZA:      # attribuzione esclusiva
                if tipo in presenti:
                    esclusiva[modello][nome] += 1
                    break
            for tipo, nome in PRECEDENZA:      # conteggio non esclusivo
                if tipo in presenti:
                    non_esclusiva[modello][nome] += 1

    return esclusiva, non_esclusiva, uscite_errate, piu_difetti


def stampa_tabella(titolo, dati, con_totale):
    print("\n" + titolo)
    print("-" * (30 + 26 * len(MODELLI)))
    print(f"{'':30s}" + "".join(f"{m:>26s}" for m in MODELLI))
    for c in CATEGORIE:
        print(f"{c:30s}" + "".join(f"{dati[m][c]:26d}" for m in MODELLI))
    if con_totale:
        print(f"{'TOTALE':30s}" + "".join(f"{sum(dati[m].values()):26d}" for m in MODELLI))


def main():
    if not os.path.exists(CSV_RISULTATI):
        raise SystemExit(
            f"File non trovato: {CSV_RISULTATI}\n"
            "Eseguire lo script dalla radice del repository."
        )

    esclusiva, non_esclusiva, uscite_errate, piu_difetti = analizza()

    stampa_tabella(
        "A) ATTRIBUZIONE ESCLUSIVA - una categoria per uscita, la piu' grave\n"
        "   (e' la tabella riportata nella tesi)",
        esclusiva, con_totale=True)

    stampa_tabella(
        "B) CONTEGGIO NON ESCLUSIVO - ogni difetto contato ovunque compaia\n"
        "   (le colonne non sommano al totale: un'uscita puo' comparire piu' volte)",
        non_esclusiva, con_totale=False)

    print("\nC) USCITE CON PIU' DI UN DIFETTO")
    print("-" * (30 + 26 * len(MODELLI)))
    print(f"{'':30s}" + "".join(f"{m:>26s}" for m in MODELLI))
    print(f"{'Uscite errate':30s}" + "".join(f"{uscite_errate[m]:26d}" for m in MODELLI))
    print(f"{'di cui con piu difetti':30s}" + "".join(f"{piu_difetti[m]:26d}" for m in MODELLI))
    print(f"{'quota':30s}" + "".join(
        f"{piu_difetti[m] / uscite_errate[m] * 100:25.1f}%" for m in MODELLI))
    print()


if __name__ == "__main__":
    main()
