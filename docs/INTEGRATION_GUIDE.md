# Ingestion service: integration guide

This guide is for whoever connects an API gateway or a back office to the ingestion service (`app/ingesta/servicio.py`). Part 1 explains what the service does and why. Part 2 is the reference for each route.

The service, its routes and its JSON fields use Spanish names, because the system is built for Spanish-language documents and for teams that work in Spanish. This guide keeps those names as they are and explains them.

---

# Part 1. How it works

## 1.1 The problem

A university keeps dozens of regulations as PDF files. A student who wants to know the requirements to graduate has to open them one by one and search by hand.

The goal is to let that student ask in plain Spanish and get the correct answer together with the regulation and article it came from.

The hard part is that language models, when they do not know an answer, usually do not say so; they produce a plausible one. In regulations a made-up requirement or deadline is not a small error: a student who follows it can lose a procedure.

The design described here exists to prevent that.

## 1.2 The two phases

**Curation.** Each regulation is cut into pieces that make sense on their own (an article, a chapter) and every piece gets a label saying where it came from, for example `Título Segundo > Capítulo I > Artículo 15`. A piece with its label is a **fragment**.

**Query.** When a question arrives, the system finds the fragments closest in meaning to the question, gives them to the language model and asks it to answer using only that material and to cite the label.

The ingestion service covers the curation phase for one document at a time: it turns a PDF into fragments and stores them.

## 1.3 Search by meaning

Keyword search fails when the student and the regulation use different words: "¿cómo me titulo?" and "proceso de titulación" share no word. Each fragment is therefore also stored as a vector (an embedding) that represents its meaning, so that texts saying the same thing end up close to each other.

Vectors are stored in PostgreSQL with the `pgvector` extension and an HNSW index per model.

The project does not use one embedding model but thirteen, plus Doc2Vec. Each has its own column. Comparing them on Spanish legal text is part of the research, and the ingestion request lets you choose which ones to compute.

## 1.4 The pipeline, stage by stage

```
   PDF
    │
    ▼
┌──────────────────────────────┐
│ 1. Extraction                │  pipeline/01_extraccion.py
│    text + font size + bold   │
└──────────────────────────────┘
    │  JSON of line blocks
    ▼
┌──────────────────────────────┐
│ 2. Semantic tree             │  pipeline/02_arbol_semantico.py
│    rebuild the hierarchy     │
└──────────────────────────────┘
    │  JSON with path: Título > Capítulo > Artículo
    ▼
┌──────────────────────────────┐
│ 3. Chunking + fidelity gate  │  pipeline/03_chunker_semantico.py
│    split and clean           │
└──────────────────────────────┘
    │  JSON of final fragments
    ▼
┌──────────────────────────────┐
│ 4. Load                      │  app/ingesta/pipeline_documento.py
│    insert and vectorize      │
└──────────────────────────────┘
    │
    ▼
 PostgreSQL + pgvector
```

### Stage 1. Extraction

The script first checks whether the document has a usable text layer, sampling its first pages with content. Older regulations are often scanned images.

- If it does, the text is read with pdfplumber, keeping font size and bold. A page whose text is still unusable is sent to OCR on its own.
- If it does not, the whole document goes through Tesseract OCR in Spanish at 400 dpi, after OpenCV preprocessing.
- Pages laid out in two columns are detected by the gap between columns and split, so lines from the left and right columns are not merged.

Font size and bold are kept for every line. They look irrelevant and are what makes stage 2 possible.

### Stage 2. Semantic tree

A regulation is not flat text. It has titles, chapters, sections and articles, and an article means different things depending on the chapter it sits under.

The script combines two signals:

1. **Typography.** The most common font size is taken as body text; larger or bold lines are heading candidates.
2. **Legal vocabulary.** Patterns such as `TÍTULO`, `CAPÍTULO`, `SECCIÓN`, `Artículo` and Roman numerals.

Each piece of text ends up with its full path, for example `Título Segundo > Capítulo I > Artículo 15`.

A heading candidate that ends in `,`, `;`, `:` or `.` is not promoted to a level. The rule exists to skip in-text references such as "Título Décimo de la Ley Orgánica;". Its side effect is that documents writing article headings as `Artículo 13.` keep those articles inside the chapter fragment. Gazettes (`99_GACETAS_HISTORICAS`) are treated as narrative documents and are not forced into a legal hierarchy.

### Stage 3. Chunking and the fidelity gate

Long nodes are split with a recursive splitter (1,500 characters, 200 overlap) that respects paragraph boundaries. Splitting by a fixed number of words was tried first and failed: an obligation ended up in one fragment and its exception in the next.

Then comes cleanup. OCR output carries noise: broken words, repeated page headers, misread characters. A local model (`llama3.2:3b` by default) is asked to fix only that.

The risk is that a language model, besides cleaning, may summarize, paraphrase or add text. In a regulation that changes the rule.

The **fidelity gate** compares each cleaned fragment with its input:

| Condition | Threshold |
|---|---|
| `difflib.SequenceMatcher` ratio, original vs. cleaned | at least **0.55** |
| Length ratio, cleaned / original | between **0.4 and 2.0** |

If both hold, the cleaned version is kept. If not, the cleanup is **discarded, the original text is kept and the case is logged** to `_fidelity_flags.jsonl` for human review.

The decision is a fixed rule, not another model. Anyone can rerun it and get the same result. Using a model to check a model would bring back the risk the gate is meant to remove.

A real case from the example in `examples/`: asked to clean the heading "De la Estructura y Organización de la Universidad", the model returned the heading followed by two sentences about departments and department heads that do not appear in the source. Similarity was 0.31 and the cleanup was rejected.

### Stage 4. Load

Each fragment is inserted into `benchmark_chunks` with its source document, its hierarchical path and its content. Then the vectors for the requested models are computed and written to their columns.

`pipeline/04_vectorizacion_masiva.py`, used to rebuild the whole corpus, starts with `TRUNCATE`. That is fine for a full rebuild and would wipe the corpus if used to add one document. The ingestion service does not call it; it performs its own incremental insert.

## 1.5 Watching the pipeline run

`despliegue/demo_pipeline.py` runs stages 1 to 3 on one PDF and prints what went in and out of each stage. It does not touch the database.

```bash
python despliegue/demo_pipeline.py examples/estatuto_general/estatuto_general_extracto.pdf --categoria 00_MARCO_CONSTITUCIONAL
```

It pauses after each stage until you press Enter. To run straight through and keep the intermediate files:

```bash
python despliegue/demo_pipeline.py examples/estatuto_general/estatuto_general_extracto.pdf --categoria 00_MARCO_CONSTITUCIONAL --sin-pausas --conservar
```

Outputs go to `data/demo_pipeline/`.

**After stage 1** it reports how many lines were extracted, with which method, and the most common font size, and shows real blocks:

```json
{"texto": "Título Segundo", "font_size": 11.0, "is_bold": true, "origen": "nativo"}
```

`origen` is `nativo` for the text layer and `ocr` for Tesseract.

**After stage 2** it lists the detected paths and shows a node:

```json
{
  "documento": "estatuto_general_extracto",
  "ruta_jerarquica": ["Título Segundo", "Capítulo I", "Artículo 15"],
  "texto_fragmento": ". El desarrollo de las Unidades Académicas tendrá en la planeación integral su\neje central, el cual será: ...",
  "tipo_documento": "articulado"
}
```

**After stage 3** it reports the final count and how many cleanups the gate rejected, with a warning above 10 %. It shows one rejected case side by side and a final fragment:

```json
{
  "documento_origen": "estatuto_general_extracto",
  "ruta_jerarquica": ["Título Segundo", "Capítulo I", "Artículo 15"],
  "texto_para_vectorizar": "DOCUMENTO: estatuto_general_extracto\nJERARQUÍA: Título Segundo > Capítulo I > Artículo 15\nCONTENIDO:\nEl desarrollo de las Unidades Académicas tendrá en la planeación integral su eje central, ...",
  "chunk_numero": 1,
  "chunk_total": 1
}
```

The vectorized text includes the document name and the path, so the embedding carries the context of the article and not only its literal content.

The run ends with the funnel for the document. For the example extract:

```
  1. Lines extracted               148
  2. Fragments with hierarchy       13
  3. Final fragments                13
     Cleanups rejected by the gate   3
```

## 1.6 Files

```
app/ingesta/
├── config.py              Everything configurable: database, model, folders, token
├── pipeline_documento.py  Runs the 4 stages on one document
└── servicio.py            HTTP routes and the job queue

despliegue/
├── migracion_base_de_datos.sql   Idempotent database migration
├── verificar_entorno.py          Environment check
└── demo_pipeline.py              The demo above
```

Every value in `config.py` can be overridden with an environment variable (see `.env.example`).

## 1.7 Why processing is asynchronous

Processing a document is not instant. A medium-sized regulation takes five to fifteen minutes on CPU; a scanned one that needs OCR can take half an hour, and each extra embedding model adds time. A request that waited for the result would be cut by any proxy.

So:

1. You upload the PDF and the service answers **immediately** with a job id and status **202**.
2. The document enters a queue and documents are processed one at a time, so the embedding models do not compete for memory.
3. You poll the job with its id.

---

# Part 2. Reference

## 2.1 Setup

### Requirements

| Component | Detail |
|---|---|
| Python | 3.11 or newer (tested on 3.12) |
| PostgreSQL | with `pgvector` and the `benchmark_chunks` table |
| Ollama | running, with `qwen2.5:3b` and `llama3.2:3b` pulled |
| Tesseract OCR | with Spanish data (`spa`); needed for scanned PDFs |
| Poppler | needed to rasterize pages |

### Install

```bash
cd grounded-rag-es
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Check the environment

```bash
python despliegue/verificar_entorno.py
```

It checks seven things (Python version, packages, Tesseract and Poppler with Spanish, project files, database connection with columns and indexes, user permissions, Ollama with the model, writable work folders) and prints the fix for each one that fails. It exits with 0 when everything is ready and 1 otherwise, so it can be used in a deploy script.

### Database

```bash
# Back up first
pg_dump -h "$RAG_DB_HOST" -U "$RAG_DB_USER" -d "$RAG_DB_NAME" -t benchmark_chunks -f backup.sql

# Migration: adds missing columns and indexes, deletes nothing, safe to rerun
psql -h "$RAG_DB_HOST" -U "$RAG_DB_USER" -d "$RAG_DB_NAME" -f despliegue/migracion_base_de_datos.sql
```

If the table does not exist yet:

```bash
python pipeline/00_init_db.py
```

### Configuration

Copy `.env.example` to `.env`, fill it in and load it before starting the service:

```bash
set -a; source .env; set +a
```

If `RAG_TOKEN_API` is set, the routes that change data require `Authorization: Bearer <token>`. Never commit the database password or the token.

### Start

Local testing (listens on 127.0.0.1 unless `RAG_HOST_INGESTA` says otherwise):

```bash
python -m app.ingesta.servicio
```

Production, behind a WSGI server. One worker, because the queue lives in memory. Set `RAG_TOKEN_API` before binding to a public interface:

```bash
gunicorn --workers 1 --threads 4 --timeout 120 \
         --bind 0.0.0.0:5001 "app.ingesta.servicio:app"
```

## 2.2 Routes

| Method | Route | What it does | Token |
|---|---|---|---|
| GET | `/ingesta/salud` | Health of service, database and local model | no |
| GET | `/ingesta/categorias` | Valid categories and models | no |
| POST | `/ingesta/documentos` | Upload a PDF and queue it | yes |
| GET | `/ingesta/trabajos/{id}` | Progress of one job | no |
| GET | `/ingesta/trabajos` | List of jobs | no |
| GET | `/ingesta/documentos` | Documents already loaded | no |
| DELETE | `/ingesta/documentos/{nombre}` | Delete a document from the database | yes |

## 2.3 Routes in detail

### GET /ingesta/salud

Health probe. It always answers 200, so you can tell "the service is down" from "the service is up and says the database is down".

```bash
curl http://HOST:5001/ingesta/salud
```

```json
{
  "servicio": "activo",
  "momento": "2026-09-08T18:20:11+00:00",
  "base_de_datos": { "conectada": true, "fragmentos": 6291, "documentos": 105 },
  "modelo_local": { "disponible": true },
  "trabajos": { "en_cola": 0, "procesando": 1 }
}
```

### GET /ingesta/categorias

Returns the valid values. Call it at startup instead of hard-coding categories in the gateway.

```json
{
  "categorias": [
    "00_MARCO_CONSTITUCIONAL",
    "01_NORMATIVA_ACADEMICA_ESTUDIANTIL",
    "02_DERECHOS_HUMANOS_E_INCLUSION",
    "03_INVESTIGACION_Y_POSGRADO",
    "04_GOBERNANZA_Y_ADMINISTRACION",
    "99_GACETAS_HISTORICAS"
  ],
  "modelos_disponibles": ["bge_m3", "beto", "distiluse", "legal_xlm", "maria",
                          "mbert", "me5_base", "me5_large", "minilm", "mpnet",
                          "robertalex", "sim_es", "simcse"],
  "modelos_por_defecto": ["me5_large", "robertalex", "bge_m3"],
  "tamano_maximo_mb": 80
}
```

### POST /ingesta/documentos

Upload the PDF as `multipart/form-data`.

| Field | Required | Description |
|---|---|---|
| `archivo` | yes | The PDF, up to `RAG_MB_MAXIMO` MB (80 by default) |
| `categoria` | yes | One of the valid categories |
| `modelos` | no | Comma-separated model keys. Defaults to the three default models |

```bash
curl -X POST http://HOST:5001/ingesta/documentos \
  -H "Authorization: Bearer $RAG_TOKEN_API" \
  -F "archivo=@Reglamento_Escolar_2026.pdf" \
  -F "categoria=01_NORMATIVA_ACADEMICA_ESTUDIANTIL" \
  -F "modelos=me5_large,robertalex"
```

Response **202**:

```json
{
  "id_trabajo": "a3f9c1d02b77",
  "estado": "en_cola",
  "documento": "Reglamento_Escolar_2026",
  "consultar_en": "/ingesta/trabajos/a3f9c1d02b77",
  "posicion_en_cola": 1
}
```

Errors:

| Code | Cause |
|---|---|
| 400 | Missing file, invalid category or unknown model |
| 401 | Missing or wrong token |
| 413 | File larger than the configured maximum |
| 415 | Not a PDF |

### GET /ingesta/trabajos/{id}

Job progress. Polling every 30 seconds is enough.

```bash
curl http://HOST:5001/ingesta/trabajos/a3f9c1d02b77
```

States: `en_cola` (queued), `procesando` (running), `completado` (done), `fallido` (failed).

While running:

```json
{
  "id": "a3f9c1d02b77",
  "estado": "procesando",
  "documento": "Reglamento_Escolar_2026",
  "categoria": "01_NORMATIVA_ACADEMICA_ESTUDIANTIL",
  "modelos": ["me5_large", "robertalex"],
  "recibido": "2026-09-08T18:20:11+00:00",
  "inicio": "2026-09-08T18:20:12+00:00",
  "bitacora": [
    { "momento": "2026-09-08T18:20:12+00:00", "mensaje": "Documento recibido: Reglamento_Escolar_2026.pdf" },
    { "momento": "2026-09-08T18:22:40+00:00", "mensaje": "Etapa 01_extraccion.py completada" },
    { "momento": "2026-09-08T18:23:05+00:00", "mensaje": "Etapa 02_arbol_semantico.py completada" }
  ]
}
```

When done:

```json
{
  "id": "a3f9c1d02b77",
  "estado": "completado",
  "fin": "2026-09-08T18:31:44+00:00",
  "resultado": {
    "documento": "Reglamento_Escolar_2026",
    "categoria": "01_NORMATIVA_ACADEMICA_ESTUDIANTIL",
    "fragmentos": 184,
    "modelos_vectorizados": ["me5_large", "robertalex"],
    "descartados_por_fidelidad": 3
  }
}
```

`descartados_por_fidelidad` counts fragments whose cleanup was rejected by the gate and that kept their original text. It is not an error. A high value relative to the total usually points to a poor scan.

On failure:

```json
{
  "id": "a3f9c1d02b77",
  "estado": "fallido",
  "error": "El documento 'Reglamento_Escolar_2026' ya existe en la base. Elimínalo primero si quieres reemplazarlo."
}
```

### GET /ingesta/documentos

Inventory of what is loaded.

```json
{
  "total_documentos": 105,
  "total_fragmentos": 6291,
  "documentos": [
    { "documento": "Reglamento_Escolar_2026", "fragmentos": 184, "alta": "2026-09-08T18:31:44+00:00" }
  ]
}
```

### DELETE /ingesta/documentos/{nombre}

Deletes every fragment of a document. To replace a version: delete, then upload again.

```bash
curl -X DELETE \
  -H "Authorization: Bearer $RAG_TOKEN_API" \
  "http://HOST:5001/ingesta/documentos/Reglamento_Escolar_2026"
```

```json
{ "documento": "Reglamento_Escolar_2026", "fragmentos_eliminados": 184 }
```

This cannot be undone. The original PDF stays in `data/01_raw_pdf/`, so it can always be processed again.

## 2.4 Full example

```bash
BASE="http://HOST:5001"
TOKEN="$RAG_TOKEN_API"

# 1. Check that everything answers
curl -s "$BASE/ingesta/salud" | jq '.base_de_datos.conectada'

# 2. Upload and keep the job id
ID=$(curl -s -X POST "$BASE/ingesta/documentos" \
      -H "Authorization: Bearer $TOKEN" \
      -F "archivo=@Reglamento_Escolar_2026.pdf" \
      -F "categoria=01_NORMATIVA_ACADEMICA_ESTUDIANTIL" \
      | jq -r '.id_trabajo')

# 3. Wait until it finishes
while true; do
  ESTADO=$(curl -s "$BASE/ingesta/trabajos/$ID" | jq -r '.estado')
  echo "Estado: $ESTADO"
  [ "$ESTADO" = "completado" ] || [ "$ESTADO" = "fallido" ] && break
  sleep 30
done

# 4. Result
curl -s "$BASE/ingesta/trabajos/$ID" | jq '.resultado'
```

## 2.5 Notes for the gateway

**Timeouts.** Upload answers in under a second; a 30-second timeout is plenty. Status queries are immediate.

**Retries.** Do not automatically retry a `POST` that already returned 202: the document would be queued twice and the second job would fail as a duplicate. If you got no answer, check `/ingesta/trabajos` first.

**Concurrency.** One document at a time, on purpose, because the embedding models compete for memory. You can queue several; they run in order.

**Job persistence.** The job list lives in memory and is lost on restart. Documents already in the database are not. Keep a permanent history on the gateway side if you need one.

**Model choice.** Each extra model adds minutes. Use the three defaults for normal operation and ask for more only for a formal comparison.

## 2.6 Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `salud` says database disconnected | Credentials or network | Check the `RAG_DB_*` variables and that the port is reachable |
| `salud` says model unavailable | Ollama down | Start Ollama and check that `qwen2.5:3b` is pulled |
| Job failed: "no produjo ningún fragmento" | PDF without readable text | Check that Tesseract (with `spa`) and Poppler are installed |
| Job failed: "ya existe en la base" | Duplicate document | Delete it and upload again |
| Stage 3 takes very long | Ollama busy or very large document | Expected for long documents on CPU |
| 413 | File too large | Raise `RAG_MB_MAXIMO` or split the document |

## 2.7 Querying what was loaded

Loaded documents are queried through the search and comparison service (`app/server/app.py`):

```bash
curl -X POST http://HOST:5000/benchmark-RAG/search \
  -H "Content-Type: application/json" \
  -d '{
        "query": "¿Cuáles son los requisitos para titularme?",
        "modelos": ["mE5-Large", "BM25"],
        "llm": "qwen2.5:3b"
      }'
```

For each requested method it returns the generated answer, the three fragments used with their hierarchical path, and retrieval and generation times. Method names: `mBERT`, `Legal-XLM`, `RoBERTalex`, `mE5-Large`, `MPNet`, `SimCSE`, `MiniLM`, `BETO`, `mE5-Base`, `Sim-ES`, `MarIA`, `DistilUSE`, `Doc2Vec`, `BM25`, `TF-IDF`.
