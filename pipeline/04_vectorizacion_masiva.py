import json
import gc
import os
import torch
import psycopg2
from psycopg2 import extras
from pathlib import Path
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# Conexion a PostgreSQL con pgvector, tomada de variables de entorno (ver .env.example)
DB_PARAMS = {
    "host": os.getenv("RAG_DB_HOST", "localhost"),
    "port": int(os.getenv("RAG_DB_PORT", "5432")),
    "user": os.getenv("RAG_DB_USER", "postgres"),
    "password": os.getenv("RAG_DB_PASSWORD", ""),
    "dbname": os.getenv("RAG_DB_NAME", "rag_benchmark"),
}

# Mapeo de modelos densos a columnas en la base de datos
MODELOS_DENSOS = {
    "bert-base-multilingual-cased": "vector_mbert",
    "joelniklaus/legal-xlm-roberta-large": "vector_legal_xlm",
    "plantl-gob-es/robertalex": "vector_robertalex",
    "intfloat/multilingual-e5-large": "vector_me5",
    "paraphrase-multilingual-mpnet-base-v2": "vector_mpnet",
    "princeton-nlp/sup-simcse-roberta-base": "vector_simcse",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "vector_minilm",
    "dccuchile/bert-base-spanish-wwm-uncased": "vector_beto",
    "intfloat/multilingual-e5-base": "vector_e5_base",
    "hiiamsid/sentence_similarity_spanish_es": "vector_sim_es",
    "PlanTL-GOB-ES/roberta-base-bne": "vector_maria",
    "sentence-transformers/distiluse-base-multilingual-cased-v1": "vector_distiluse",
    "BAAI/bge-m3": "vector_bge_m3",
}


def main():
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        print("Purificando entorno: Truncando tabla benchmark_chunks...")
        cur.execute("TRUNCATE TABLE benchmark_chunks RESTART IDENTITY;")
        conn.commit()
        print("Base de datos limpia y lista para nueva ingesta.")

        corpus_memoria = []

        # Resolucion de ruta absoluta independiente del directorio de ejecucion
        script_dir = Path(__file__).resolve().parent
        data_dir = script_dir.parent / "data" / "04_final_chunks"

        if not data_dir.exists():
            raise FileNotFoundError(f"El directorio {data_dir} no existe.")

        archivos_json = list(data_dir.rglob("*.json"))
        print(f"Encontrados {len(archivos_json)} archivos JSON en {data_dir}")

        for archivo in archivos_json:
            with open(archivo, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]

                    for chunk in data:
                        doc = chunk.get("documento_origen", "")
                        
                        ruta_raw = chunk.get("ruta_jerarquica", [])
                        if isinstance(ruta_raw, list):
                            ruta = " > ".join([str(item).strip() for item in ruta_raw if item])
                        else:
                            ruta = str(ruta_raw).strip()
                        
                        if not ruta:
                            ruta = "Sin jerarquia"

                        cont = chunk.get("texto_para_vectorizar", "")
                        if not cont:
                            continue

                        insert_query = """
                            INSERT INTO benchmark_chunks (documento, ruta_jerarquica, contenido)
                            VALUES (%s, %s, %s)
                            RETURNING id;
                        """
                        cur.execute(insert_query, (doc, ruta, cont))
                        id_bd = cur.fetchone()[0]

                        corpus_memoria.append({"id": id_bd, "texto": cont})
                        
                except Exception as e:
                    print(f"Error procesando {archivo}: {e}")

        conn.commit()
        print(f"Ingesta completada. {len(corpus_memoria)} chunks cargados en memoria y BD.")

        if not corpus_memoria:
            print("No se encontraron chunks validos. Finalizando ejecucion.")
            return

        textos_a_vectorizar = [item["texto"] for item in corpus_memoria]
        ids_a_actualizar = [item["id"] for item in corpus_memoria]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Dispositivo de computo detectado: {device}")

        # Vectorizacion secuencial con modelos densos
        for model_name, column_name in MODELOS_DENSOS.items():
            print(f"\n--- Procesando modelo: {model_name} -> Columna: {column_name} ---")
            try:
                model = SentenceTransformer(model_name, device=device)
            except Exception as e:
                print(f"Error cargando el modelo {model_name}: {e}")
                continue

            print("Generando embeddings...")
            embeddings = model.encode(
                textos_a_vectorizar,
                batch_size=8,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            print("Actualizando base de datos...")
            update_query = f"UPDATE benchmark_chunks SET {column_name} = %s WHERE id = %s"
            datos_update = [
                (emb.tolist(), id_bd)
                for emb, id_bd in zip(embeddings, ids_a_actualizar)
            ]
            extras.execute_batch(cur, update_query, datos_update, page_size=100)
            conn.commit()

            print("Liberando memoria VRAM/RAM...")
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Vectorizacion baseline con Doc2Vec
        print("\n--- Procesando Baseline: Doc2Vec -> Columna: vector_doc2vec ---")
        print("Tokenizando documentos para Doc2Vec...")
        documentos_etiquetados = [
            TaggedDocument(words=texto.lower().split(), tags=[str(id_bd)])
            for id_bd, texto in zip(ids_a_actualizar, textos_a_vectorizar)
        ]

        print("Entrenando modelo Doc2Vec (vector_size=300, epochs=20)...")
        modelo_doc2vec = Doc2Vec(
            vector_size=300,
            min_count=2,
            epochs=20,
            workers=4
        )
        modelo_doc2vec.build_vocab(documentos_etiquetados)
        modelo_doc2vec.train(
            documentos_etiquetados,
            total_examples=modelo_doc2vec.corpus_count,
            epochs=modelo_doc2vec.epochs
        )

        print("Guardando modelo en disco (modelo_doc2vec.model)...")
        modelo_path = Path(os.getenv("RAG_DOC2VEC_MODEL", str(script_dir.parent / "models" / "modelo_doc2vec.model")))
        modelo_path.parent.mkdir(parents=True, exist_ok=True)
        modelo_doc2vec.save(str(modelo_path))

        print("Infiere vectores Doc2Vec...")
        vectores_doc2vec = [
            modelo_doc2vec.infer_vector(texto.lower().split()).tolist()
            for texto in textos_a_vectorizar
        ]

        print("Actualizando base de datos con Doc2Vec...")
        update_doc2vec_query = "UPDATE benchmark_chunks SET vector_doc2vec = %s WHERE id = %s"
        datos_update_doc2vec = [
            (vec, id_bd)
            for vec, id_bd in zip(vectores_doc2vec, ids_a_actualizar)
        ]
        extras.execute_batch(cur, update_doc2vec_query, datos_update_doc2vec, page_size=100)
        conn.commit()

        print("\nProceso de vectorizacion masiva finalizado correctamente.")

    except Exception as e:
        print(f"Error critico en el pipeline: {e}")
    finally:
        if conn:
            conn.close()
            print("Conexion a PostgreSQL cerrada.")


if __name__ == "__main__":
    main()