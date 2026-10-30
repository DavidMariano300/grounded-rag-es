# -*- coding: utf-8 -*-
"""Comprueba que el servidor está listo para correr el servicio de ingesta.

Revisa siete cosas y dice, para cada una, si está bien o qué hacer si no.
No modifica nada: solo mira y reporta.

Uso:
    python despliegue/verificar_entorno.py

Conviene correrlo dos veces: antes de aplicar la migración, para saber qué
falta, y después, para confirmar que quedó.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

VERDE, ROJO, AMBAR, FIN = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
_resultados: list[tuple[str, bool]] = []


def titulo(t: str) -> None:
    print(f"\n{'=' * 68}\n  {t}\n{'=' * 68}")


def bien(t: str) -> None:
    print(f"  {VERDE}[ OK ]{FIN} {t}")


def mal(t: str, arreglo: str = "") -> None:
    print(f"  {ROJO}[FALTA]{FIN} {t}")
    if arreglo:
        print(f"         {AMBAR}Solución:{FIN} {arreglo}")


def aviso(t: str) -> None:
    print(f"  {AMBAR}[AVISO]{FIN} {t}")


def registrar(nombre: str, ok: bool) -> None:
    _resultados.append((nombre, ok))


# ------------------------------------------------------------------ 1. Python
def revisar_python() -> None:
    titulo("1. Versión de Python")
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    if ok:
        bien(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        mal(f"Python {v.major}.{v.minor}, se requiere 3.10 o superior",
            "Instala una versión más reciente o usa un entorno virtual con ella")
    registrar("Python 3.10+", ok)


# ------------------------------------------------------------ 2. Dependencias
def revisar_paquetes() -> None:
    titulo("2. Paquetes de Python")
    requeridos = {
        "flask": "Servidor web del servicio",
        "psycopg2": "Conexión a PostgreSQL",
        "requests": "Llamadas al modelo local",
        "torch": "Cálculo de vectores",
        "sentence_transformers": "Modelos de representación",
        "pdfplumber": "Lectura de PDF con texto",
        "pypdf": "Inspección de PDF",
        "pdf2image": "Conversión de páginas a imagen",
        "pytesseract": "Reconocimiento óptico",
        "langchain_text_splitters": "Segmentación de texto",
        "tqdm": "Barras de avance",
    }
    faltan = []
    for modulo, para_que in requeridos.items():
        try:
            __import__(modulo)
            bien(f"{modulo:<26} {para_que}")
        except ImportError:
            mal(f"{modulo:<26} {para_que}")
            faltan.append(modulo)
    if faltan:
        print(f"\n         {AMBAR}Solución:{FIN} pip install -r requirements.txt")
    registrar("Paquetes de Python", not faltan)


# ----------------------------------------------------- 3. Programas del sistema
def revisar_binarios() -> None:
    titulo("3. Programas del sistema")
    binarios = {
        "tesseract": ("Reconocimiento óptico para PDF escaneados",
                      "sudo apt install tesseract-ocr tesseract-ocr-spa"),
        "pdftoppm": ("Convierte páginas de PDF a imagen (paquete poppler)",
                     "sudo apt install poppler-utils"),
    }
    todo_ok = True
    for binario, (para_que, arreglo) in binarios.items():
        ruta = shutil.which(binario)
        if ruta:
            bien(f"{binario:<12} {para_que}")
        else:
            mal(f"{binario:<12} {para_que}", arreglo)
            todo_ok = False

    if shutil.which("tesseract"):
        try:
            salida = subprocess.run(["tesseract", "--list-langs"],
                                    capture_output=True, text=True, timeout=20)
            idiomas = (salida.stdout + salida.stderr).lower()
            if "spa" in idiomas:
                bien("tesseract    idioma español instalado")
            else:
                mal("tesseract    falta el idioma español",
                    "sudo apt install tesseract-ocr-spa")
                todo_ok = False
        except Exception as exc:                      # noqa: BLE001
            aviso(f"No se pudo consultar los idiomas de tesseract: {exc}")
    registrar("Programas del sistema", todo_ok)


# ---------------------------------------------------------- 4. Base de datos
def revisar_base() -> None:
    titulo("4. Base de datos")
    try:
        from app.ingesta import config
    except Exception as exc:                          # noqa: BLE001
        mal(f"No se pudo leer la configuración: {exc}")
        registrar("Base de datos", False)
        return

    destino = (f"{config.DB_PARAMS['host']}:{config.DB_PARAMS['port']}"
               f"/{config.DB_PARAMS['dbname']}")
    print(f"  Destino configurado: {destino}")
    if not config.DB_PARAMS["password"]:
        aviso("La contraseña viene vacía. Define RAG_DB_PASSWORD antes de arrancar.")

    try:
        import psycopg2
        conn = psycopg2.connect(**config.DB_PARAMS, connect_timeout=8)
    except Exception as exc:                          # noqa: BLE001
        mal(f"No se pudo conectar: {exc}",
            "Revisa las variables RAG_DB_* y que el puerto esté abierto")
        registrar("Base de datos", False)
        return

    bien("Conexión establecida")
    todo_ok = True
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    if cur.fetchone():
        bien("Extensión pgvector activa")
    else:
        mal("Falta la extensión pgvector", "Corre la migración o: CREATE EXTENSION vector;")
        todo_ok = False

    cur.execute("SELECT to_regclass('public.benchmark_chunks')")
    if not cur.fetchone()[0]:
        mal("No existe la tabla benchmark_chunks", "python pipeline/00_init_db.py")
        conn.close()
        registrar("Base de datos", False)
        return
    bien("Tabla benchmark_chunks presente")

    cur.execute("SELECT COUNT(*), COUNT(DISTINCT documento) FROM benchmark_chunks")
    filas, documentos = cur.fetchone()
    print(f"         Contenido actual: {filas} fragmentos de {documentos} documentos")

    cur.execute("SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'benchmark_chunks'")
    columnas = {c[0] for c in cur.fetchall()}

    for col in ("documento", "ruta_jerarquica", "contenido"):
        if col in columnas:
            bien(f"Columna {col}")
        else:
            mal(f"Falta la columna {col}", "Revisa el esquema, es indispensable")
            todo_ok = False

    for col in ("created_at", "updated_at"):
        if col in columnas:
            bien(f"Columna {col}")
        else:
            mal(f"Falta la columna {col}", "Aplica despliegue/migracion_base_de_datos.sql")
            todo_ok = False

    from app.ingesta import config as cfg
    necesarias = {cfg.MODELOS[m][1] for m in cfg.MODELOS_POR_DEFECTO}
    for col in sorted(necesarias):
        if col in columnas:
            bien(f"Columna de vectores {col} (modelo por defecto)")
        else:
            mal(f"Falta {col}, que usa un modelo por defecto",
                "Aplica despliegue/migracion_base_de_datos.sql")
            todo_ok = False

    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'benchmark_chunks'")
    indices = {i[0] for i in cur.fetchall()}
    if "idx_benchmark_chunks_documento" in indices:
        bien("Índice sobre documento")
    else:
        mal("Falta el índice sobre documento",
            "Aplica la migración. Sin él, cada alta y cada borrado recorren toda la tabla")
        todo_ok = False

    hnsw = [i for i in indices if i.startswith("idx_hnsw")]
    if hnsw:
        bien(f"Índices vectoriales presentes: {len(hnsw)}")
    else:
        aviso("No hay índices vectoriales. La búsqueda funcionará, pero más lenta.")

    try:
        cur.execute("SELECT has_table_privilege(current_user, 'benchmark_chunks', 'INSERT'), "
                    "has_table_privilege(current_user, 'benchmark_chunks', 'DELETE')")
        puede_insertar, puede_borrar = cur.fetchone()
        if puede_insertar and puede_borrar:
            bien("El usuario puede insertar y borrar")
        else:
            mal("Al usuario le faltan permisos de escritura",
                "GRANT INSERT, UPDATE, DELETE ON benchmark_chunks TO <usuario>;")
            todo_ok = False
    except Exception as exc:                          # noqa: BLE001
        aviso(f"No se pudieron comprobar los permisos: {exc}")

    conn.close()
    registrar("Base de datos", todo_ok)


# ---------------------------------------------------------- 5. Modelo local
def revisar_ollama() -> None:
    titulo("5. Modelo de lenguaje local")
    try:
        from app.ingesta import config
        import requests
    except Exception as exc:                          # noqa: BLE001
        mal(f"No se pudo preparar la comprobación: {exc}")
        registrar("Modelo local", False)
        return

    raiz = config.OLLAMA_URL.rsplit("/api/", 1)[0]
    print(f"  Destino configurado: {raiz}")
    try:
        r = requests.get(f"{raiz}/api/tags", timeout=8)
        if r.status_code != 200:
            mal(f"Ollama respondió con código {r.status_code}", "Revisa que el servicio esté activo")
            registrar("Modelo local", False)
            return
        bien("Ollama responde")
        modelos = [m["name"] for m in r.json().get("models", [])]
        buscado = config.OLLAMA_MODELO
        if any(buscado.split(":")[0] in m for m in modelos):
            bien(f"Modelo {buscado} disponible")
            registrar("Modelo local", True)
        else:
            mal(f"No se encontró el modelo {buscado}", f"ollama pull {buscado}")
            print(f"         Modelos presentes: {', '.join(modelos) or 'ninguno'}")
            registrar("Modelo local", False)
    except Exception as exc:                          # noqa: BLE001
        mal(f"No se pudo conectar con Ollama: {exc}",
            "Levanta el servicio y revisa RAG_OLLAMA_URL")
        registrar("Modelo local", False)


# --------------------------------------------------------------- 6. Archivos
def revisar_archivos() -> None:
    titulo("6. Archivos del proyecto")
    necesarios = [
        "pipeline/01_extraccion.py",
        "pipeline/02_arbol_semantico.py",
        "pipeline/03_chunker_semantico.py",
        "pipeline/00_init_db.py",
        "app/ingesta/config.py",
        "app/ingesta/pipeline_documento.py",
        "app/ingesta/servicio.py",
        "requirements.txt",
    ]
    todo_ok = True
    for rel in necesarios:
        if (RAIZ / rel).exists():
            bien(rel)
        else:
            mal(rel, "Falta subir este archivo al servidor")
            todo_ok = False
    registrar("Archivos del proyecto", todo_ok)


# -------------------------------------------------------------- 7. Escritura
def revisar_carpetas() -> None:
    titulo("7. Permisos de escritura")
    try:
        from app.ingesta import config
    except Exception as exc:                          # noqa: BLE001
        mal(f"No se pudo leer la configuración: {exc}")
        registrar("Permisos de escritura", False)
        return

    todo_ok = True
    for carpeta in (config.DIR_TRABAJOS, config.DIR_ARCHIVO_PDF):
        try:
            carpeta.mkdir(parents=True, exist_ok=True)
            prueba = carpeta / ".prueba_escritura"
            prueba.write_text("ok", encoding="utf-8")
            prueba.unlink()
            bien(f"Se puede escribir en {carpeta}")
        except Exception as exc:                      # noqa: BLE001
            mal(f"No se puede escribir en {carpeta}: {exc}",
                f"sudo chown -R $USER {carpeta}")
            todo_ok = False
    registrar("Permisos de escritura", todo_ok)


def main() -> int:
    print("\n" + "=" * 68)
    print("  VERIFICACIÓN DEL ENTORNO · Servicio de ingesta")
    print("  Módulo normativo, Plataforma Integral de IA de la UAGro")
    print("=" * 68)

    revisar_python()
    revisar_paquetes()
    revisar_binarios()
    revisar_archivos()
    revisar_base()
    revisar_ollama()
    revisar_carpetas()

    titulo("Resumen")
    for nombre, ok in _resultados:
        marca = f"{VERDE}correcto{FIN}" if ok else f"{ROJO}requiere atención{FIN}"
        print(f"  {nombre:<28} {marca}")

    fallas = [n for n, ok in _resultados if not ok]
    print()
    if fallas:
        print(f"  {ROJO}El entorno todavía no está listo.{FIN}")
        print(f"  Pendientes: {', '.join(fallas)}")
        return 1
    print(f"  {VERDE}Todo en orden. El servicio se puede arrancar.{FIN}")
    print("  python -m app.ingesta.servicio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
