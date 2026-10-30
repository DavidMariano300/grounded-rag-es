# -*- coding: utf-8 -*-
"""Configuración del servicio de ingesta.

Todo lo que cambia entre el equipo de desarrollo y el servidor de producción
vive aquí. No hace falta tocar el resto del código.

Las variables de entorno tienen prioridad sobre estos valores, de modo que en
producción se pueden inyectar sin editar el archivo.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent

# ------------------------------------------------------------ base de datos
DB_PARAMS = {
    "host": os.getenv("RAG_DB_HOST", "localhost"),
    "port": int(os.getenv("RAG_DB_PORT", "5432")),
    "user": os.getenv("RAG_DB_USER", "postgres"),
    "password": os.getenv("RAG_DB_PASSWORD", ""),
    "dbname": os.getenv("RAG_DB_NAME", "rag_benchmark"),
}

# --------------------------------------------------------------- modelo local
OLLAMA_URL = os.getenv("RAG_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODELO = os.getenv("RAG_OLLAMA_MODELO", "qwen2.5:3b")

# ------------------------------------------------------------------- carpetas
DIR_PIPELINE = RAIZ / "pipeline"
DIR_TRABAJOS = Path(os.getenv("RAG_DIR_TRABAJOS", str(RAIZ / "data" / "trabajos_ingesta")))
DIR_ARCHIVO_PDF = Path(os.getenv("RAG_DIR_PDF", str(RAIZ / "data" / "01_raw_pdf")))

# --------------------------------------------------------------------- reglas
MB_MAXIMO = int(os.getenv("RAG_MB_MAXIMO", "80"))
TOKEN_API = os.getenv("RAG_TOKEN_API", "")   # vacío desactiva la autenticación
PUERTO = int(os.getenv("RAG_PUERTO_INGESTA", "5001"))

# Categorías admitidas. Determinan la carpeta destino y sirven para clasificar
# el documento dentro del corpus. Deben coincidir con las carpetas de
# data/01_raw_pdf, porque el pipeline guarda el nombre de la carpeta como
# categoría de cada fragmento en la base.
CATEGORIAS = [
    "00_MARCO_CONSTITUCIONAL",
    "01_NORMATIVA_ACADEMICA_ESTUDIANTIL",
    "02_DERECHOS_HUMANOS_E_INCLUSION",
    "03_INVESTIGACION_Y_POSGRADO",
    "04_GOBERNANZA_Y_ADMINISTRACION",
    "99_GACETAS_HISTORICAS",
]

# Modelos de representación vectorial disponibles, con su columna en la tabla.
# La clave es el nombre corto que se usa en las peticiones.
MODELOS = {
    "bge_m3":     ("BAAI/bge-m3", "vector_bge_m3"),
    "legal_xlm":  ("joelniklaus/legal-xlm-roberta-large", "vector_legal_xlm"),
    "robertalex": ("plantl-gob-es/robertalex", "vector_robertalex"),
    "me5_large":  ("intfloat/multilingual-e5-large", "vector_me5"),
    "me5_base":   ("intfloat/multilingual-e5-base", "vector_e5_base"),
    "mpnet":      ("paraphrase-multilingual-mpnet-base-v2", "vector_mpnet"),
    "minilm":     ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "vector_minilm"),
    "beto":       ("dccuchile/bert-base-spanish-wwm-uncased", "vector_beto"),
    "mbert":      ("bert-base-multilingual-cased", "vector_mbert"),
    "simcse":     ("princeton-nlp/sup-simcse-roberta-base", "vector_simcse"),
    "sim_es":     ("hiiamsid/sentence_similarity_spanish_es", "vector_sim_es"),
    "maria":      ("PlanTL-GOB-ES/roberta-base-bne", "vector_maria"),
    "distiluse":  ("sentence-transformers/distiluse-base-multilingual-cased-v1", "vector_distiluse"),
}

# Modelos que se calculan si la petición no especifica ninguno. Agregar más
# multiplica el tiempo de ingesta.
MODELOS_POR_DEFECTO = ["me5_large", "robertalex", "bge_m3"]
