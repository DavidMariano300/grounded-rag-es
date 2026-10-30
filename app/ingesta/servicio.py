# -*- coding: utf-8 -*-
"""Servicio de ingesta de documentos normativos.

Expone las rutas que el API Gateway necesita para dar de alta un PDF y que
este atraviese solo el proceso ETL hasta quedar consultable en la base
vectorial.

El procesamiento no es instantáneo: un documento tarda de varios minutos a
media hora según su tamaño y la cantidad de modelos solicitados. Por eso la
carga responde de inmediato con un identificador de trabajo y el avance se
consulta aparte.

Arranque:
    python -m app.ingesta.servicio
"""
from __future__ import annotations

import os
import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from . import config
from .pipeline_documento import ErrorPipeline, procesar

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MB_MAXIMO * 1024 * 1024

_trabajos: Dict[str, Dict[str, Any]] = {}
_cola: "queue.Queue[str]" = queue.Queue()
_candado = threading.Lock()


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _actualizar(id_trabajo: str, **campos: Any) -> None:
    with _candado:
        _trabajos[id_trabajo].update(campos)


def _anotar(id_trabajo: str, mensaje: str) -> None:
    with _candado:
        _trabajos[id_trabajo]["bitacora"].append({"momento": _ahora(), "mensaje": mensaje})


def _autorizado() -> bool:
    """Valida el token si el servicio fue configurado con uno."""
    if not config.TOKEN_API:
        return True
    cabecera = request.headers.get("Authorization", "")
    return cabecera == f"Bearer {config.TOKEN_API}"


def _trabajador() -> None:
    """Consume la cola y procesa un documento a la vez."""
    while True:
        id_trabajo = _cola.get()
        trabajo = _trabajos[id_trabajo]
        try:
            _actualizar(id_trabajo, estado="procesando", inicio=_ahora())
            resultado = procesar(
                ruta_pdf=Path(trabajo["ruta_pdf"]),
                categoria=trabajo["categoria"],
                modelos=trabajo["modelos"],
                dir_trabajo=Path(trabajo["dir_trabajo"]),
                aviso=lambda m, j=id_trabajo: _anotar(j, m),
            )
            _actualizar(id_trabajo, estado="completado", fin=_ahora(), resultado=resultado)
            _anotar(id_trabajo, "Proceso terminado")
        except ErrorPipeline as exc:
            _actualizar(id_trabajo, estado="fallido", fin=_ahora(), error=str(exc))
            _anotar(id_trabajo, f"Error: {exc}")
        except Exception as exc:                      # noqa: BLE001
            _actualizar(id_trabajo, estado="fallido", fin=_ahora(),
                        error=f"Error inesperado: {exc}",
                        traza=traceback.format_exc()[-2000:])
            _anotar(id_trabajo, f"Error inesperado: {exc}")
        finally:
            _cola.task_done()


threading.Thread(target=_trabajador, daemon=True, name="ingesta").start()


# ------------------------------------------------------------------- rutas
@app.get("/ingesta/salud")
def salud():
    """Comprueba que el servicio, la base de datos y el modelo local responden."""
    estado: Dict[str, Any] = {"servicio": "activo", "momento": _ahora()}
    try:
        import psycopg2
        conn = psycopg2.connect(**config.DB_PARAMS, connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT documento) FROM benchmark_chunks")
        fragmentos, documentos = cur.fetchone()
        conn.close()
        estado["base_de_datos"] = {"conectada": True, "fragmentos": fragmentos,
                                   "documentos": documentos}
    except Exception as exc:                          # noqa: BLE001
        estado["base_de_datos"] = {"conectada": False, "detalle": str(exc)}

    try:
        import requests
        raiz = config.OLLAMA_URL.rsplit("/api/", 1)[0]
        r = requests.get(f"{raiz}/api/tags", timeout=5)
        estado["modelo_local"] = {"disponible": r.status_code == 200}
    except Exception as exc:                          # noqa: BLE001
        estado["modelo_local"] = {"disponible": False, "detalle": str(exc)}

    with _candado:
        estado["trabajos"] = {
            "en_cola": _cola.qsize(),
            "procesando": sum(1 for t in _trabajos.values() if t["estado"] == "procesando"),
        }
    return jsonify(estado)


@app.get("/ingesta/categorias")
def categorias():
    """Lista las categorías admitidas y los modelos disponibles."""
    return jsonify({
        "categorias": config.CATEGORIAS,
        "modelos_disponibles": sorted(config.MODELOS),
        "modelos_por_defecto": config.MODELOS_POR_DEFECTO,
        "tamano_maximo_mb": config.MB_MAXIMO,
    })


@app.post("/ingesta/documentos")
def subir_documento():
    """Recibe un PDF y encola su procesamiento.

    Formulario de tipo multipart con los campos:
        archivo    obligatorio, el PDF
        categoria  obligatorio, una de las categorías admitidas
        modelos    opcional, claves separadas por coma
    """
    if not _autorizado():
        return jsonify({"error": "Token invalido o ausente"}), 401

    if "archivo" not in request.files:
        return jsonify({"error": "Falta el campo 'archivo'"}), 400
    archivo = request.files["archivo"]
    if not archivo.filename:
        return jsonify({"error": "El archivo viene sin nombre"}), 400
    if not archivo.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Solo se admiten archivos PDF"}), 415

    categoria = (request.form.get("categoria") or "").strip()
    if categoria not in config.CATEGORIAS:
        return jsonify({"error": "Categoria no valida",
                        "categorias_admitidas": config.CATEGORIAS}), 400

    crudo = (request.form.get("modelos") or "").strip()
    modelos: List[str] = ([m.strip() for m in crudo.split(",") if m.strip()]
                          if crudo else list(config.MODELOS_POR_DEFECTO))
    desconocidos = [m for m in modelos if m not in config.MODELOS]
    if desconocidos:
        return jsonify({"error": "Modelos no reconocidos", "modelos": desconocidos,
                        "disponibles": sorted(config.MODELOS)}), 400

    id_trabajo = uuid.uuid4().hex[:12]
    dir_trabajo = config.DIR_TRABAJOS / id_trabajo
    dir_trabajo.mkdir(parents=True, exist_ok=True)

    nombre = secure_filename(archivo.filename)
    ruta_pdf = dir_trabajo / nombre
    archivo.save(ruta_pdf)

    destino_archivo = config.DIR_ARCHIVO_PDF / categoria
    destino_archivo.mkdir(parents=True, exist_ok=True)
    (destino_archivo / nombre).write_bytes(ruta_pdf.read_bytes())

    with _candado:
        _trabajos[id_trabajo] = {
            "id": id_trabajo, "estado": "en_cola", "documento": Path(nombre).stem,
            "archivo": nombre, "categoria": categoria, "modelos": modelos,
            "ruta_pdf": str(ruta_pdf), "dir_trabajo": str(dir_trabajo),
            "recibido": _ahora(), "bitacora": [],
        }
    _cola.put(id_trabajo)

    return jsonify({
        "id_trabajo": id_trabajo,
        "estado": "en_cola",
        "documento": Path(nombre).stem,
        "consultar_en": f"/ingesta/trabajos/{id_trabajo}",
        "posicion_en_cola": _cola.qsize(),
    }), 202


@app.get("/ingesta/trabajos/<id_trabajo>")
def estado_trabajo(id_trabajo: str):
    """Devuelve el avance de un trabajo de ingesta."""
    with _candado:
        trabajo = _trabajos.get(id_trabajo)
        if not trabajo:
            return jsonify({"error": "Trabajo no encontrado"}), 404
        respuesta = {k: v for k, v in trabajo.items()
                     if k not in ("ruta_pdf", "dir_trabajo")}
    return jsonify(respuesta)


@app.get("/ingesta/trabajos")
def listar_trabajos():
    """Lista los trabajos registrados desde que arrancó el servicio."""
    with _candado:
        lista = [{"id": t["id"], "documento": t["documento"], "estado": t["estado"],
                  "recibido": t["recibido"]} for t in _trabajos.values()]
    return jsonify({"total": len(lista), "trabajos": sorted(
        lista, key=lambda x: x["recibido"], reverse=True)})


@app.get("/ingesta/documentos")
def listar_documentos():
    """Lista los documentos que ya están cargados en la base vectorial."""
    try:
        import psycopg2
        conn = psycopg2.connect(**config.DB_PARAMS)
        cur = conn.cursor()
        cur.execute(
            "SELECT documento, COUNT(*) AS fragmentos, MIN(created_at) AS alta "
            "FROM benchmark_chunks GROUP BY documento ORDER BY documento"
        )
        filas = cur.fetchall()
        conn.close()
    except Exception as exc:                          # noqa: BLE001
        return jsonify({"error": f"No se pudo consultar la base: {exc}"}), 503

    return jsonify({
        "total_documentos": len(filas),
        "total_fragmentos": sum(f[1] for f in filas),
        "documentos": [{"documento": f[0], "fragmentos": f[1],
                        "alta": f[2].isoformat() if f[2] else None} for f in filas],
    })


@app.delete("/ingesta/documentos/<path:documento>")
def eliminar_documento(documento: str):
    """Elimina de la base todos los fragmentos de un documento."""
    if not _autorizado():
        return jsonify({"error": "Token invalido o ausente"}), 401
    try:
        import psycopg2
        conn = psycopg2.connect(**config.DB_PARAMS)
        cur = conn.cursor()
        cur.execute("DELETE FROM benchmark_chunks WHERE documento = %s", (documento,))
        borrados = cur.rowcount
        conn.commit()
        conn.close()
    except Exception as exc:                          # noqa: BLE001
        return jsonify({"error": f"No se pudo eliminar: {exc}"}), 503

    if borrados == 0:
        return jsonify({"error": "El documento no existe en la base"}), 404
    return jsonify({"documento": documento, "fragmentos_eliminados": borrados})


@app.errorhandler(413)
def demasiado_grande(_):
    return jsonify({"error": f"El archivo supera los {config.MB_MAXIMO} MB"}), 413


if __name__ == "__main__":
    config.DIR_TRABAJOS.mkdir(parents=True, exist_ok=True)
    print(f"Servicio de ingesta escuchando en el puerto {config.PUERTO}")
    print(f"Base de datos: {config.DB_PARAMS['host']}:{config.DB_PARAMS['port']}"
          f"/{config.DB_PARAMS['dbname']}")
    print(f"Autenticacion por token: {'activada' if config.TOKEN_API else 'desactivada'}")
    # Solo local por defecto; para exponerlo a otra máquina, fijar RAG_TOKEN_API y RAG_HOST_INGESTA.
    app.run(host=os.getenv("RAG_HOST_INGESTA", "127.0.0.1"), port=config.PUERTO, threaded=True)
