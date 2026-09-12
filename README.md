***Machine Learning Emotion Recognition (2020)***. 
Progetto d'esame per il corso di Interfacce Uomo-Macchina, Università degli Studi dell'Insubria, anno accademico 2025/2026. 

**Autori:**. 
- Lecchi Matilde 759875
- Pellegrini Gaia 759909
- Caredda Anna Eleonora 762576

Progetto basato sul paper "A comparative analysis of machine learning methods for emotion recognition using EEG and peripheral physiological signals" (Doma V., Pirouz M., J Big Data 7, 18, 2020), applicato al dataset DEAP.

**Requisiti**. 
    - *Python 3.10+*
    - Le seguenti librerie (installabili con pip): `numpy, scipy, scikit-learn, matplotlib, pandas`

**Struttura del progetto**. 
```
DEAPAnalisi/
├── main.py
├── data/
│   ├── raw/
│   └── preprocessed/
├── src/
│   ├── preprocessing.py
│   ├── filtering.py
│   ├── features.py
│   ├── features_test.py
│   ├── models.py
│   ├── evaluation.py
│   └── utils.py
└── results/
    ├── baseline/
    ├── custom/
    ├── models/
    └── figures/
```

Results viene creata autimaticamente al primo avvio, se è già presente e si esegue nuovamente il main, i dati vengono sovrascritti con la nuova versione. 

**Test di sviluppo**. 
Durante lo sviluppo sono stati realizzati dei test intermedi per verificare il corretto funzionamento delle singole parti delle pipeline prima di integrarle in main.py (ad esempio src/features_test.py, che controlla l'estrazione delle feature, unico di questi test che abbiamo lasciato). Questi script non fanno parte dell'esecuzione principale del progetto. 

**Come avviare il progetto**. 
Dalla cartella principale del progetto utilizzare il comando: 
    `python3 main.py`

Il programma stampa a schermo l'avanzamento in tempo reale. Ci vorrà un po' di tempo ma al termine dell'esecuzione, in console deve comparire: 
    `=== Pipeline completata ===`

**Note metodologiche**. 
Le confusion matrix e le curve ROC salvate come immagini vengono generate riaddestrando il modello sull'intero dataset a scopo illustrativo — non sono calcolate sugli stessi dati usati per le metriche numeriche riportate nei file JSON, che provengono invece dalla cross-validation (Leave-One-Trial-Out per la baseline, GroupKFold per la custom).

Se la ROC AUC risulta NaN in un fold (es. per la presenza di una sola classe nel test set), viene impostata convenzionalmente a 0.5 (prestazione equivalente al caso casuale).