"""
===============================================================
    Machine Learning Emotion Recognition (2020)

    Autori:
        - Lecchi Matilde (759875)
        - Pellegrini Gaia (759909)
        - Caredda Anna Eleonora (762576)

    Anno Accademico: 2025/2026
    Corso: Interfacce Uomo-Macchina

    Descrizione:
        Questo file fa parte del progetto basato sul paper
        "Machine Learning Emotion Recognition (2020)", dedicato
        all'analisi delle emozioni tramite segnali EEG attraverso
        tecniche di preprocessing, estrazione di feature e modelli
        di machine learning.
===============================================================
"""


# ============================================================
# IMPORT DELLE LIBRERIE
# ============================================================

# NumPy viene utilizzato per leggere i dati binari contenuti
# nei file .dat e per eseguire operazioni numeriche sui segnali.
import numpy as np

# os permette di gestire file e directory, ad esempio per
# recuperare l'elenco dei file presenti nella cartella del dataset.
import os


# ============================================================
# CARICAMENTO DEL FILE .DAT
# ============================================================

def load_dat_file(path):
    """
    Carica un file .dat del dataset DEAP.

    Il file viene letto come sequenza di valori numerici
    rappresentati in formato float32.

    Parametri:
        path: percorso del file .dat da caricare.

    Restituisce:
        Un array NumPy contenente i dati del file.
        In caso di errore restituisce None.
    """

    # np.fromfile() legge direttamente il contenuto binario
    # del file e lo interpreta come valori float32.
    try:
        data = np.fromfile(path, dtype=np.float32)

        # Restituisce l'array contenente il segnale.
        return data

    # Se si verifica un errore durante la lettura del file,
    # viene mostrato un messaggio e viene restituito None.
    except Exception as e:
        print(f"Errore nel caricamento di {path}: {e}")
        return None


# ============================================================
# CONTROLLO DEL SAMPLING RATE
# ============================================================

def check_sampling_rate(data_length):
    """
    Determina quale tipologia di file DEAP si sta analizzando
    sulla base del numero totale di campioni.

    Nel dataset DEAP:
    - il formato RAW utilizza un sampling rate di 512 Hz;
    - il formato preprocessato utilizza un sampling rate
      di 128 Hz.

    In particolare:
    - 8064 campioni corrispondono a un trial RAW di 60 secondi
      a 512 Hz;
    - 7680 campioni corrispondono a un trial preprocessato
      di 60 secondi a 128 Hz.

    Parametri:
        data_length: numero totale di campioni presenti nel file.

    Restituisce:
        Una stringa che identifica il formato probabile del file.
    """

    # Se il numero totale di campioni è divisibile per 8064,
    # il file viene considerato compatibile con il formato RAW.
    if data_length % 8064 == 0:
        return "RAW (512 Hz)"

    # Se invece il numero di campioni è divisibile per 7680,
    # il file viene considerato compatibile con il formato
    # preprocessato a 128 Hz.
    elif data_length % 7680 == 0:
        return "PREPROCESSED (128 Hz)"

    # Se nessuna delle due condizioni è soddisfatta,
    # non è possibile identificare il formato sulla base
    # della sola lunghezza.
    else:
        return "UNKNOWN"


# ============================================================
# CONTROLLO DELLA NORMALIZZAZIONE
# ============================================================

def check_normalization(data):
    """
    Controlla se il segnale presenta caratteristiche compatibili
    con una normalizzazione nell'intervallo [-1, 1].

    Il controllo viene effettuato osservando il valore minimo
    e massimo presenti nel segnale.

    Parametri:
        data: array contenente il segnale EEG.

    Restituisce:
        Una stringa che indica se il segnale è probabilmente
        normalizzato oppure non normalizzato.
    """

    # Calcola il valore minimo presente nel segnale.
    min_val = np.min(data)

    # Calcola il valore massimo presente nel segnale.
    max_val = np.max(data)


    # Se minimo e massimo si trovano in prossimità degli estremi
    # dell'intervallo [-1, 1], il segnale viene considerato
    # probabilmente normalizzato.
    #
    # Viene utilizzato un intervallo leggermente più ampio
    # (-1.5, 1.5) per rendere il controllo meno rigido.
    if -1.5 < min_val < -0.5 and 0.5 < max_val < 1.5:
        return "Probabile NORMALIZZATO"

    # In caso contrario il segnale viene considerato
    # probabilmente non normalizzato.
    else:
        return "Probabile NON normalizzato"


# ============================================================
# CONTROLLO DEL FILTRAGGIO
# ============================================================

def check_filtering(data):
    """
    Controlla in maniera approssimativa se il segnale presenta
    caratteristiche compatibili con un segnale filtrato.

    L'analisi si basa sull'ampiezza massima del segnale:
    - valori molto elevati possono essere tipici dei dati RAW;
    - valori più contenuti possono essere compatibili con
      dati preprocessati e filtrati.

    Parametri:
        data: array contenente il segnale EEG.

    Restituisce:
        Una stringa che indica se il segnale è probabilmente
        filtrato oppure non filtrato.
    """

    # np.abs() calcola il valore assoluto di ogni campione.
    #
    # np.max() individua quindi il valore assoluto massimo
    # raggiunto dal segnale.
    if np.max(np.abs(data)) > 1000:

        # Se l'ampiezza supera 1000, il segnale viene considerato
        # probabilmente RAW e quindi non filtrato.
        return "Probabile NON filtrato (RAW)"

    else:

        # Se il valore massimo è inferiore o uguale a 1000,
        # il segnale viene considerato probabilmente filtrato
        # e quindi compatibile con il formato preprocessato.
        return "Probabile filtrato (PREPROCESSED)"


# ============================================================
# ANALISI COMPLESSIVA DI UN FILE
# ============================================================

def analyze_file(path):
    """
    Esegue tutti i controlli disponibili su un singolo file .dat.

    Vengono analizzati:
    - dimensione del file;
    - sampling rate/formato;
    - normalizzazione;
    - filtraggio.

    Al termine viene fornita una conclusione sul fatto che
    il file sia probabilmente RAW o PREPROCESSED.
    """

    # Stampa il nome del file che sta per essere analizzato.
    print(f"\n=== Analisi file: {path} ===")


    # --------------------------------------------------------
    # CARICAMENTO
    # --------------------------------------------------------

    # Carica il contenuto del file .dat.
    data = load_dat_file(path)


    # Se il caricamento non è andato a buon fine,
    # interrompe l'analisi del file corrente.
    if data is None:
        return


    # Stampa il numero totale di campioni presenti nel file.
    print(f"Dimensione totale: {len(data)} campioni")


    # --------------------------------------------------------
    # SAMPLING RATE
    # --------------------------------------------------------

    # Determina il formato probabile del file sulla base
    # della sua lunghezza.
    sr = check_sampling_rate(len(data))

    # Stampa il risultato del controllo.
    print(f"- Sampling rate: {sr}")


    # --------------------------------------------------------
    # NORMALIZZAZIONE
    # --------------------------------------------------------

    # Controlla se i valori del segnale sono compatibili
    # con una normalizzazione nell'intervallo [-1, 1].
    norm = check_normalization(data)

    # Stampa il risultato del controllo.
    print(f"- Normalizzazione: {norm}")


    # --------------------------------------------------------
    # FILTRAGGIO
    # --------------------------------------------------------

    # Controlla se l'ampiezza del segnale è compatibile
    # con un segnale RAW oppure preprocessato.
    filt = check_filtering(data)

    # Stampa il risultato del controllo.
    print(f"- Filtraggio: {filt}")


    # --------------------------------------------------------
    # CONCLUSIONE
    # --------------------------------------------------------

    # Se il controllo del sampling rate ha identificato
    # il formato RAW, il file viene classificato come
    # probabilmente RAW.
    if sr.startswith("RAW"):
        print(">>> RISULTATO: Il file è quasi certamente RAW.")

    # Se invece è stato identificato il formato preprocessato,
    # il file viene classificato come probabilmente
    # PREPROCESSED.
    elif sr.startswith("PREPROCESSED"):
        print(">>> RISULTATO: Il file è quasi certamente PREPROCESSED.")

    # Se il sampling rate non è stato riconosciuto,
    # non viene fornita una classificazione definitiva.
    else:
        print(">>> RISULTATO: Impossibile determinare con certezza.")


# ============================================================
# MAIN
# ============================================================

# Questa condizione verifica che il file venga eseguito
# direttamente e non importato da un altro modulo.
if __name__ == "__main__":

    # Percorso della directory contenente i file .dat
    # del dataset DEAP.
    RAW_DIR = "data/raw/"


    # Messaggio iniziale visualizzato nel terminale.
    print("=== Verifica preprocessamento DEAP ===")


    # os.listdir() restituisce tutti i file presenti
    # nella directory specificata.
    #
    # Il ciclo permette quindi di analizzare automaticamente
    # tutti i file presenti nella cartella.
    for filename in os.listdir(RAW_DIR):

        # Vengono analizzati solamente i file con estensione .dat.
        if filename.endswith(".dat"):

            # os.path.join() costruisce il percorso completo
            # del file e lo passa alla funzione analyze_file().
            analyze_file(
                os.path.join(RAW_DIR, filename)
            )