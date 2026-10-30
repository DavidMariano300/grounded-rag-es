# Crear el archivo en tu directorio de trabajo
#cat > ~/PROYECTO_RAG/db_vectorial/01_init_db.py << 'PYEOF'
"""Inicialización de base de datos PostgreSQL con pgvector.
Este script crea la extensión vector, la tabla principal benchmark_chunks
con todas las columnas de embeddings, e índices HNSW para búsqueda eficiente.
"""
import psycopg2
from pgvector.psycopg2 import register_vector
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_db(
    host: str = os.getenv("RAG_DB_HOST", "localhost"),
    port: int = int(os.getenv("RAG_DB_PORT", "5432")),
    user: str = os.getenv("RAG_DB_USER", "postgres"),
    password: str = os.getenv("RAG_DB_PASSWORD", ""),
    dbname: str = os.getenv("RAG_DB_NAME", "rag_benchmark")
) -> bool:
    """Inicializa la base de datos con la estructura requerida."""
    conn_params = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "dbname": dbname
    }

    conn = None
    cur = None

    try:
        logger.info("Conectando a PostgreSQL...")
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cur = conn.cursor()

        # Crear extensión pgvector
        logger.info("Creando extensión pgvector...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Registrar tipo vector para psycopg2
        register_vector(conn)

        # Crear tabla principal benchmark_chunks
        logger.info("Creando tabla benchmark_chunks...")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS benchmark_chunks (
            id SERIAL PRIMARY KEY,
            documento TEXT NOT NULL,
            ruta_jerarquica TEXT,
            contenido TEXT NOT NULL,
            vector_bge_m3 vector(1024),
            vector_legal_xlm vector(1024),
            vector_robertalex vector(768),
            vector_me5 vector(1024),
            vector_mpnet vector(768),
            vector_simcse vector(768),
            vector_doc2vec vector(300),
            vector_mbert vector(768),
            vector_minilm vector(384),
            vector_beto vector(768),
            vector_e5_base vector(768),
            vector_sim_es vector(768),
            vector_maria vector(768),
            vector_distiluse vector(768),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        cur.execute(create_table_query)
        logger.info("Tabla 'benchmark_chunks' creada/verificada exitosamente.")

        # Crear índices HNSW
        logger.info("Creando índices HNSW...")
        indices = {
            "idx_hnsw_bge": "vector_bge_m3",
            "idx_hnsw_legal_xlm": "vector_legal_xlm",
            "idx_hnsw_robertalex": "vector_robertalex",
            "idx_hnsw_me5": "vector_me5",
            "idx_hnsw_mpnet": "vector_mpnet",
            "idx_hnsw_simcse": "vector_simcse",
            "idx_hnsw_doc2vec": "vector_doc2vec",
            "idx_hnsw_mbert": "vector_mbert",
            "idx_hnsw_minilm": "vector_minilm",
            "idx_hnsw_beto": "vector_beto",
            "idx_hnsw_e5_base": "vector_e5_base",
            "idx_hnsw_sim_es": "vector_sim_es",
            "idx_hnsw_maria": "vector_maria",
            "idx_hnsw_distiluse": "vector_distiluse"
        }

        for idx_name, column in indices.items():
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {idx_name} 
                ON benchmark_chunks 
                USING hnsw ({column} vector_cosine_ops) 
                WITH (m=16, ef_construction=64);
            """)
            logger.info(f"Índice HNSW creado para {column}")

        # Índices auxiliares
        logger.info("Creando índices auxiliares...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_documento ON benchmark_chunks (documento);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_ruta ON benchmark_chunks (ruta_jerarquica);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_created ON benchmark_chunks (created_at);")

        # Función de reinicio
        logger.info("Creando función de reinicio...")
        cur.execute("""
        CREATE OR REPLACE FUNCTION reset_benchmark_chunks()
        RETURNS void AS $$
        BEGIN
            TRUNCATE TABLE benchmark_chunks RESTART IDENTITY CASCADE;
            RAISE NOTICE 'Tabla benchmark_chunks reiniciada.';
        END;
        $$ LANGUAGE plpgsql;
        """)

        # Vista de estadísticas
        logger.info("Creando vista de estadísticas...")
        cur.execute("""
        CREATE OR REPLACE VIEW vw_chunks_stats AS
        SELECT 
            COUNT(*) as total_chunks,
            COUNT(vector_robertalex) FILTER (WHERE vector_robertalex IS NOT NULL) as robertalex_completos,
            COUNT(vector_legal_xlm) FILTER (WHERE vector_legal_xlm IS NOT NULL) as legal_xlm_completos,
            COUNT(vector_doc2vec) FILTER (WHERE vector_doc2vec IS NOT NULL) as doc2vec_completos,
            AVG(LENGTH(contenido)) as avg_content_length
        FROM benchmark_chunks;
        """)

        logger.info("Inicialización de base de datos completada exitosamente.")
        return True

    except psycopg2.OperationalError as e:
        logger.error(f"Error de conexión a PostgreSQL: {e}")
        return False
    except Exception as e:
        logger.error(f"Error durante la inicialización: {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    success = init_db()
    if success:
        print("Base de datos inicializada correctamente.")
    else:
        print("Error al inicializar la base de datos.")
        exit(1)
#PYEOF