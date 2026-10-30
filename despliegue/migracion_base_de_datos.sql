-- ============================================================================
--  Migración de la base para habilitar el servicio de ingesta
--  grounded-rag-es
--  Septiembre de 2026
-- ============================================================================
--
--  Este guion es IDEMPOTENTE: se puede correr varias veces sin causar daño.
--  Todo usa IF NOT EXISTS, de modo que si algo ya existe simplemente se salta.
--
--  NO borra datos. NO modifica filas existentes. Solo agrega lo que falte.
--
--  Cómo aplicarlo:
--      psql -h "$RAG_DB_HOST" -U "$RAG_DB_USER" -d "$RAG_DB_NAME" -f migracion_base_de_datos.sql
--
--  Antes de correrlo conviene sacar un respaldo:
--      pg_dump -h "$RAG_DB_HOST" -U "$RAG_DB_USER" -d "$RAG_DB_NAME" -t benchmark_chunks \
--              -f respaldo_benchmark_chunks_$(date +%F).sql
-- ============================================================================

\echo '--- 1. Extensión pgvector ---'
CREATE EXTENSION IF NOT EXISTS vector;

\echo '--- 2. Estado actual de la tabla ---'
SELECT
    COUNT(*)                     AS filas_actuales,
    COUNT(DISTINCT documento)    AS documentos_actuales
FROM benchmark_chunks;

-- ============================================================================
--  3. Columnas de control
-- ============================================================================
--  El servicio de ingesta registra cuándo entró cada fragmento y cuándo se
--  actualizaron sus vectores. Si la tabla de producción se creó sin estas
--  columnas, aquí se agregan. Las filas que ya existen quedan con la fecha
--  del momento de la migración, lo cual es correcto: no sabemos cuándo se
--  cargaron realmente.
-- ============================================================================

\echo '--- 3. Columnas de control (created_at, updated_at) ---'
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- ============================================================================
--  4. Columnas de vectores que pueden faltar
-- ============================================================================
--  El servicio de búsqueda que está hoy en producción consulta doce columnas.
--  El esquema de referencia declara catorce. Estas dos son las que suelen
--  faltar, y una de ellas (bge_m3) está entre los modelos por defecto del
--  servicio de ingesta, así que es necesaria.
--
--  La dimensión debe coincidir exactamente con la salida del modelo. Si una
--  columna ya existe con otra dimensión, este guion NO la toca: habría que
--  resolverlo a mano, porque cambiarla implicaría recalcular los vectores.
-- ============================================================================

\echo '--- 4. Columnas de vectores ---'
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_bge_m3   vector(1024);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_doc2vec  vector(300);

-- Las doce restantes se declaran también por si la tabla se creó incompleta.
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_legal_xlm  vector(1024);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_me5        vector(1024);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_robertalex vector(768);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_mpnet      vector(768);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_simcse     vector(768);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_mbert      vector(768);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_minilm     vector(384);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_beto       vector(768);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_e5_base    vector(768);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_sim_es     vector(768);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_maria      vector(768);
ALTER TABLE benchmark_chunks ADD COLUMN IF NOT EXISTS vector_distiluse  vector(768);

-- ============================================================================
--  5. Índice sobre el nombre del documento
-- ============================================================================
--  Este es el cambio MÁS IMPORTANTE de toda la migración.
--
--  El servicio de ingesta hace tres operaciones que filtran por documento:
--      a) comprobar si el documento ya existe, antes de insertarlo
--      b) listar el inventario agrupando por documento
--      c) borrar todos los fragmentos de un documento
--
--  Sin índice, cada una de esas operaciones recorre la tabla completa. Con
--  seis mil filas todavía se tolera, pero el corpus crece con cada carga.
-- ============================================================================

\echo '--- 5. Índice sobre documento ---'
CREATE INDEX IF NOT EXISTS idx_benchmark_chunks_documento
    ON benchmark_chunks (documento);

-- ============================================================================
--  6. Índices de búsqueda vectorial
-- ============================================================================
--  HNSW acelera la búsqueda por similitud. Sin él, una consulta compara el
--  vector de la pregunta contra todos los fragmentos uno por uno.
--
--  Aviso: construir estos índices sobre una tabla con datos tarda varios
--  minutos por índice y consume memoria. Conviene correr esta sección en una
--  ventana de mantenimiento, no en horario de uso.
--
--  Si prefieres aplazarla, la migración funciona igual: los índices vectoriales
--  son de rendimiento, no de correctitud.
-- ============================================================================

\echo '--- 6. Índices HNSW (puede tardar varios minutos) ---'
CREATE INDEX IF NOT EXISTS idx_hnsw_bge       ON benchmark_chunks USING hnsw (vector_bge_m3   vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_hnsw_legal_xlm ON benchmark_chunks USING hnsw (vector_legal_xlm vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_hnsw_me5       ON benchmark_chunks USING hnsw (vector_me5      vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_hnsw_robertalex ON benchmark_chunks USING hnsw (vector_robertalex vector_cosine_ops);

-- ============================================================================
--  7. Verificación posterior
-- ============================================================================

\echo '--- 7. Columnas resultantes ---'
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'benchmark_chunks'
ORDER BY ordinal_position;

\echo '--- 8. Índices resultantes ---'
SELECT indexname
FROM pg_indexes
WHERE tablename = 'benchmark_chunks'
ORDER BY indexname;

\echo '--- Migración terminada ---'
