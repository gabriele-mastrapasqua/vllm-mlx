# vLLM-MLX Agent Rules

Questo documento contiene le regole operative che gli agenti devono seguire quando lavorano su questo progetto.

## Branch Git

### Regola 1: Nuovo Branch per Task Sostanziali
**OBBLIGATORIO**: Per ogni task o modifica sostanziale, creare sempre un nuovo branch git.

**Procedura:**
1. Prima di iniziare qualsiasi modifica, verificare lo stato del repository
2. Creare un branch con naming convenzionale
3. Lavorare esclusivamente su quel branch
4. Non fare mai modifiche dirette su main/master senza approvazione

**Naming Convenzioni Branch:**
- `feature/<descrizione>` - Per nuove funzionalità (es: `feature/moe-model-support`)
- `fix/<descrizione>` - Per bug fix (es: `fix/embed-q-loading`)
- `update/<descrizione>` - Per aggiornamenti (es: `update/mlx-lm-0.31`)
- `docs/<descrizione>` - Per documentazione (es: `docs/api-examples`)

**Esempio Workflow:**
```bash
# 1. Verificare stato
git status

# 2. Creare e switchare branch
git checkout -b feature/support-moe-models

# 3. Fare le modifiche...

# 4. Committare (SOLO su richiesta esplicita utente)
git add .
git commit -m "feat: add support for MoE models with embed_q parameter filtering"
```

## Commit e Push

### Regola 2: No Commit Automatici
**NEVER** fare commit automatici. Chiedere sempre conferma esplicita all'utente prima di:
- Fare commit
- Fare push
- Creare pull request
- Fare qualsiasi operazione git distruttiva

### Regola 3: Staging Selettivo
Verificare sempre quali file vengono aggiunti al commit:
- NON committare file con secrets (.env, credenziali, key)
- NON committare file di cache o temporanei
- NON committare dipendenze (node_modules, __pycache__, etc.)
- Aggiungere solo file rilevanti per il task

## Code Quality

### Regola 4: Verifica Pre-Completamento
Prima di dichiarare un task completato, verificare sempre:

**Python:**
- `ruff check .` - Linting
- `ruff format .` - Formattazione
- `mypy .` o tool di type checking disponibili
- Test esistenti passano: `pytest`

**JavaScript/TypeScript:**
- `npm run lint` o equivalente
- `npm run typecheck` o equivalente
- `npm test`

**Generale:**
- Non introdurre errori di sintassi
- Mantenere consistenza con lo stile esistente
- Documentare cambiamenti significativi

### Regola 5: Sicurezza
**NEVER:**
- Loggare o esporre secrets, API key, password
- Committare file .env o credenziali
- Inserire backdoor o codice malevolo
- Bypassare validazioni di sicurezza

## Documentazione

### Regola 6: Documentare Cambiamenti
Per modifiche significative:
1. Aggiungere docstring a nuove funzioni/classi
2. Aggiornare README se necessario
3. Aggiungere type hints
4. Commentare logica complessa

## Workflow Operativo

### Regola 7: Comunicazione Progressi
Durante task lunghi:
- Informare l'utente sui progressi
- Mostrare output di comandi importanti
- Segnalare eventuali blocchi o problemi

### Regola 8: Gestione Errori
In caso di errori:
1. Non nascondere errori - mostrarli all'utente
2. Proporre soluzioni alternative
3. Non procedere con operazioni incerte
4. Chiedere conferma per azioni irreversibili

## Testing

### Regola 9: Verifica Soluzioni
Dopo implementazioni:
- Testare le modifiche se possibile
- Verificare che non ci siano regressioni
- Assicurarsi che il codice funzioni come previsto

## Decisioni Architetturali

### Regola 10: Pattern Esistenti
- Seguire pattern e convenzioni esistenti nel codebase
- Mantenere consistenza con codice circostante
- Usare librerie già presenti nel progetto
- Non assumere disponibilità di librerie non verificate

---

**Ultimo aggiornamento:** 2026-02-07
**Versione:** 1.0
