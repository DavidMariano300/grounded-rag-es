import os
import gc
import logging
from pathlib import Path
import time
import torch
import psycopg2
import requests
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/benchmark-RAG'

# Toda la configuración viene de variables de entorno (ver .env.example).
DB_PARAMS = {
    "host": os.getenv("RAG_DB_HOST", "localhost"),
    "port": int(os.getenv("RAG_DB_PORT", "5432")),
    "user": os.getenv("RAG_DB_USER", "postgres"),
    "password": os.getenv("RAG_DB_PASSWORD", ""),
    "dbname": os.getenv("RAG_DB_NAME", "rag_benchmark"),
}

OLLAMA_URL = os.getenv("RAG_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODELO = os.getenv("RAG_OLLAMA_MODELO", "qwen2.5:3b")
RAIZ = Path(__file__).resolve().parent.parent.parent
DOC2VEC_MODEL = os.getenv("RAG_DOC2VEC_MODEL", str(RAIZ / "models" / "modelo_doc2vec.model"))

TRANSFORMER_MODELS = {
    "mBERT": ("bert-base-multilingual-cased", "vector_mbert"),
    "Legal-XLM": ("joelniklaus/legal-xlm-roberta-large", "vector_legal_xlm"),
    "RoBERTalex": ("plantl-gob-es/robertalex", "vector_robertalex"),
    "mE5-Large": ("intfloat/multilingual-e5-large", "vector_me5"),
    "MPNet": ("paraphrase-multilingual-mpnet-base-v2", "vector_mpnet"),
    "SimCSE": ("princeton-nlp/sup-simcse-roberta-base", "vector_simcse"),
    "MiniLM": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "vector_minilm"),
    "BETO": ("dccuchile/bert-base-spanish-wwm-uncased", "vector_beto"),
    "mE5-Base": ("intfloat/multilingual-e5-base", "vector_e5_base"),
    "Sim-ES": ("hiiamsid/sentence_similarity_spanish_es", "vector_sim_es"),
    "MarIA": ("PlanTL-GOB-ES/roberta-base-bne", "vector_maria"),
    "DistilUSE": ("sentence-transformers/distiluse-base-multilingual-cased-v1", "vector_distiluse")
}

def get_db_connection():
    for attempt in range(3):
        try:
            return psycopg2.connect(**DB_PARAMS)
        except psycopg2.OperationalError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

def call_ollama(query, contexto, llm_seleccionado):
    prompt = f"""Actua como un asistente institucional de la UAGro. Responde la siguiente consulta basandote UNICAMENTE en el contexto proporcionado.

CONSULTA: {query}

CONTEXTO NORMATIVO:
{contexto}

INSTRUCCIONES:
1. Responde de forma clara, directa y profesional.
2. Menciona la ruta jerarquica cuando uses informacion del contexto.
3. Si la respuesta no se encuentra en el contexto, indicalo claramente sin añadir informacion externa.
4. No repitas los fragmentos; sintetiza la respuesta en un solo bloque coherente.

RESPUESTA:"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": llm_seleccionado, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 512}},
            timeout=180
        )
        return response.json().get("response", "No se pudo generar una respuesta.")
    except Exception as e:
        return f"Error al procesar la consulta: {str(e)}"

@app.route('/')
@app.route('/benchmark-RAG/')
@app.route('/benchmark-RAG')
def index():
    return render_template('index.html')

@app.route('/benchmark-RAG/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '').strip()
    modelos_seleccionados = data.get('modelos', [])
    llm_seleccionado = data.get('llm', OLLAMA_MODELO)

    if not query:
        return jsonify({"error": "La consulta no puede estar vacia"}), 400
    if not modelos_seleccionados:
        return jsonify({"error": "Seleccione al menos un modelo de recuperacion"}), 400

    final_results = {}
    chunk_tracker = {}
    logger.info(f"BENCHMARK INICIADO | Query: '{query[:80]}...' | LLM: {llm_seleccionado}")

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, ruta_jerarquica, contenido FROM benchmark_chunks WHERE contenido IS NOT NULL")
        corpus_raw = cur.fetchall()
        processed_corpus = [{"id": d[0], "ruta": d[1], "cont": str(d[2]).strip()} for d in corpus_raw if d[2] and str(d[2]).strip()]

        if not processed_corpus:
            return jsonify({"error": "No se encontraron documentos en la base de datos"}), 500

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Dispositivo: {device.upper()}")

        for model_name in modelos_seleccionados:
            try:
                top_3_chunks = []
                t0 = time.time()
                logger.info(f"Evaluando: {model_name}")

                if model_name == "BM25":
                    bm25 = BM25Okapi([d['cont'].lower().split() for d in processed_corpus])
                    scores = bm25.get_scores(query.lower().split())
                    for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]:
                        if scores[idx] > 0:
                            top_3_chunks.append({"id": processed_corpus[idx]['id'], "ruta": processed_corpus[idx]['ruta'], "texto": processed_corpus[idx]['cont']})

                elif model_name == "TF-IDF":
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([d['cont'] for d in processed_corpus])
                    query_vec = vectorizer.transform([query])
                    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
                    for idx in similarities.argsort()[-3:][::-1]:
                        if similarities[idx] > 0:
                            top_3_chunks.append({"id": processed_corpus[idx]['id'], "ruta": processed_corpus[idx]['ruta'], "texto": processed_corpus[idx]['cont']})

                elif model_name == "Doc2Vec":
                    try:
                        model_path = DOC2VEC_MODEL
                        if os.path.exists(model_path):
                            m_d2v = Doc2Vec.load(model_path)
                            q_vec = m_d2v.infer_vector(query.lower().split()).tolist()
                            cur.execute("SELECT id, ruta_jerarquica, contenido FROM benchmark_chunks WHERE vector_doc2vec IS NOT NULL ORDER BY vector_doc2vec <=> %s::vector LIMIT 3", (q_vec,))
                            for r in cur.fetchall():
                                top_3_chunks.append({"id": r[0], "ruta": r[1], "texto": r[2]})
                    except Exception:
                        logger.info("Doc2Vec: modelo no encontrado, omitiendo")
                        pass

                elif model_name in TRANSFORMER_MODELS:
                    hf_id, col_name = TRANSFORMER_MODELS[model_name]
                    logger.info(f"Cargando {hf_id} en {device.upper()}")
                    model = SentenceTransformer(hf_id, device=device)
                    q_vec = model.encode([query], convert_to_numpy=True)[0].tolist()
                    cur.execute(f"SELECT id, ruta_jerarquica, contenido FROM benchmark_chunks WHERE {col_name} IS NOT NULL ORDER BY {col_name} <=> %s::vector LIMIT 3", (q_vec,))
                    for r in cur.fetchall():
                        top_3_chunks.append({"id": r[0], "ruta": r[1], "texto": r[2]})
                    del model; gc.collect()
                    if device == 'cuda': torch.cuda.empty_cache()

                t_retrieval = time.time() - t0
                logger.info(f"Recuperacion: {round(t_retrieval, 2)}s | Chunks: {len(top_3_chunks)}")

                for chunk in top_3_chunks:
                    c_id = chunk["id"]
                    chunk_tracker.setdefault(c_id, {"ruta": chunk["ruta"], "texto": chunk["texto"], "models": []})["models"].append(model_name)

                t1 = time.time()
                contexto_str = "\n\n".join([f"[{c['ruta']}]\n{c['texto']}" for c in top_3_chunks])[:6000]
                llm_ans = call_ollama(query, contexto_str or "Sin contexto relevante.", llm_seleccionado)
                t_llm = time.time() - t1

                final_results[model_name] = {
                    "status": "success",
                    "respuesta_llm": llm_ans,
                    "chunks": [{"ruta": c["ruta"], "texto": c["texto"]} for c in top_3_chunks],
                    "stats": {"retrieval_sec": round(t_retrieval, 2), "llm_sec": round(t_llm, 2), "total_sec": round(t_retrieval + t_llm, 2)}
                }
            except Exception as e:
                logger.error(f"Fallo en {model_name}: {str(e)}")
                if 'model' in locals():
                    try: del model
                    except: pass
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                final_results[model_name] = {"status": "error", "error_msg": str(e)}

        tiempos = {m: d['stats']['retrieval_sec'] for m, d in final_results.items() if d['status'] == 'success'}
        global_stats = {
            "fastest": {"model": min(tiempos, key=tiempos.get) if tiempos else "N/A", "time": min(tiempos.values()) if tiempos else 0},
            "slowest": {"model": max(tiempos, key=tiempos.get) if tiempos else "N/A", "time": max(tiempos.values()) if tiempos else 0},
            "avg_time": round(sum(tiempos.values()) / len(tiempos), 2) if tiempos else 0,
            "consensus": sorted([c for c in chunk_tracker.values() if len(c["models"]) > 1], key=lambda x: len(x["models"]), reverse=True)
        }

        return jsonify({"results": final_results, "global_stats": global_stats})

    except psycopg2.OperationalError as e:
        logger.error(f"Error de base de datos: {e}")
        return jsonify({"error_global": "No se pudo conectar a la base de datos."}), 503
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return jsonify({"error_global": f"Error interno del servidor: {str(e)}"}), 500
    finally:
        if cur: cur.close()
        if conn: conn.close()

if __name__ == "__main__":
    # Depuración apagada por defecto: el depurador de Flask permite ejecutar código.
    app.run(host=os.getenv("RAG_SERVER_HOST", "127.0.0.1"),
            port=int(os.getenv("RAG_SERVER_PORT", "5000")),
            debug=os.getenv("RAG_DEBUG", "0") == "1")