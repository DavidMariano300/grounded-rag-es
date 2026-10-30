# -*- coding: utf-8 -*-
"""Ejecuta el proceso ETL completo sobre un solo documento PDF.

El pipeline original está pensado para procesar carpetas enteras y, en su
última etapa, vacía la tabla antes de recargarla. Eso sirve para reconstruir el
corpus desde cero, pero no para dar de alta un documento nuevo sin tocar los
que ya están.

Este módulo resuelve ese caso. Aísla cada documento en su propia carpeta de
trabajo, reutiliza sin modificar las tres primeras etapas del pipeline y
sustituye la cuarta por una inserción incremental.

Etapas:
    1. Extracción      PDF            -> JSON con líneas y atributos visuales
    2. Árbol semántico JSON           -> JSON con ruta jerárquica
    3. Fragmentación   JSON jerárquico-> JSON de fragmentos finales
    4. Carga           JSON           -> PostgreSQL, sin borrar lo existente
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

from . import config


class ErrorPipeline(Exception):
    """Falla controlada dentro de alguna etapa del proceso."""


def _correr_etapa(script: str, dir_trabajo: Path, aviso: Callable[[str], None]) -> None:
    """Ejecuta un script del pipeline con la carpeta de trabajo como raíz.

    Las tres primeras etapas leen y escriben en rutas relativas del tipo
    ./data/..., de modo que fijar el directorio de trabajo basta para aislarlas.
    """
    ruta = config.DIR_PIPELINE / script
    if not ruta.exists():
        raise ErrorPipeline(f"No se encontró el script {script}")

    proceso = subprocess.run(
        [sys.executable, str(ruta)],
        cwd=str(dir_trabajo),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proceso.returncode != 0:
        cola = (proceso.stderr or proceso.stdout or "").strip().splitlines()[-12:]
        raise ErrorPipeline(f"La etapa {script} terminó con error:\n" + "\n".join(cola))
    aviso(f"Etapa {script} completada")


def _leer_fragmentos(dir_trabajo: Path) -> List[Dict[str, Any]]:
    """Reúne los fragmentos finales que dejó la etapa de fragmentación."""
    carpeta = dir_trabajo / "data" / "04_final_chunks"
    fragmentos: List[Dict[str, Any]] = []
    for archivo in carpeta.rglob("*.json"):
        if archivo.name.startswith("_"):
            continue
        try:
            datos = json.loads(archivo.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(datos, dict):
            datos = [datos]
        for fragmento in datos:
            texto = (fragmento.get("texto_para_vectorizar") or "").strip()
            if not texto:
                continue
            ruta = fragmento.get("ruta_jerarquica", [])
            if isinstance(ruta, list):
                ruta = " > ".join(str(x).strip() for x in ruta if x)
            fragmentos.append({
                "documento": fragmento.get("documento_origen", ""),
                "ruta": str(ruta).strip() or "Sin jerarquia",
                "contenido": texto,
            })
    return fragmentos


def _contar_descartes(dir_trabajo: Path) -> int:
    """Cuenta los fragmentos que la barrera de fidelidad rechazó."""
    marca = dir_trabajo / "data" / "04_final_chunks" / "_fidelity_flags.jsonl"
    if not marca.exists():
        return 0
    return sum(1 for linea in marca.read_text(encoding="utf-8").splitlines() if linea.strip())


def _insertar(fragmentos: List[Dict[str, Any]], documento: str,
              modelos: List[str], aviso: Callable[[str], None]) -> Dict[str, Any]:
    """Inserta los fragmentos y calcula sus vectores, sin borrar lo existente."""
    import psycopg2
    import torch
    from sentence_transformers import SentenceTransformer

    conn = psycopg2.connect(**config.DB_PARAMS)
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM benchmark_chunks WHERE documento = %s", (documento,))
        if cur.fetchone()[0] > 0:
            raise ErrorPipeline(
                f"El documento '{documento}' ya existe en la base. "
                f"Elimínalo primero si quieres reemplazarlo."
            )

        ids: List[int] = []
        for fragmento in fragmentos:
            cur.execute(
                "INSERT INTO benchmark_chunks (documento, ruta_jerarquica, contenido) "
                "VALUES (%s, %s, %s) RETURNING id",
                (fragmento["documento"] or documento, fragmento["ruta"], fragmento["contenido"]),
            )
            ids.append(cur.fetchone()[0])
        conn.commit()
        aviso(f"{len(ids)} fragmentos insertados")

        textos = [f["contenido"] for f in fragmentos]
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        vectorizados: List[str] = []

        for clave in modelos:
            if clave not in config.MODELOS:
                aviso(f"Modelo desconocido, se omite: {clave}")
                continue
            nombre_hf, columna = config.MODELOS[clave]
            try:
                aviso(f"Vectorizando con {clave}")
                modelo = SentenceTransformer(nombre_hf, device=dispositivo)
                vectores = modelo.encode(textos, batch_size=8, convert_to_numpy=True,
                                         show_progress_bar=False)
                cur.executemany(
                    f"UPDATE benchmark_chunks SET {columna} = %s, updated_at = NOW() WHERE id = %s",
                    [(v.tolist(), i) for v, i in zip(vectores, ids)],
                )
                conn.commit()
                vectorizados.append(clave)
                del modelo, vectores
                if dispositivo == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                conn.rollback()
                aviso(f"Falló la vectorización con {clave}: {exc}")

        return {"fragmentos": len(ids), "ids": ids, "modelos_vectorizados": vectorizados}
    finally:
        conn.close()


def procesar(ruta_pdf: Path, categoria: str, modelos: List[str],
             dir_trabajo: Path, aviso: Callable[[str], None]) -> Dict[str, Any]:
    """Procesa un PDF de principio a fin y devuelve el resumen del resultado.

    Args:
        ruta_pdf: Archivo PDF ya guardado en disco.
        categoria: Carpeta temática a la que pertenece el documento.
        modelos: Claves de los modelos con los que se va a vectorizar.
        dir_trabajo: Carpeta aislada para este trabajo.
        aviso: Función que recibe mensajes de avance.

    Returns:
        Diccionario con el conteo de fragmentos, los modelos aplicados y los
        fragmentos que la barrera de fidelidad descartó.
    """
    entrada = dir_trabajo / "data" / "01_raw_pdf" / categoria
    entrada.mkdir(parents=True, exist_ok=True)
    destino = entrada / ruta_pdf.name
    shutil.copy2(ruta_pdf, destino)
    aviso(f"Documento recibido: {ruta_pdf.name}")

    _correr_etapa("01_extraccion.py", dir_trabajo, aviso)
    _correr_etapa("02_arbol_semantico.py", dir_trabajo, aviso)
    _correr_etapa("03_chunker_semantico.py", dir_trabajo, aviso)

    fragmentos = _leer_fragmentos(dir_trabajo)
    if not fragmentos:
        raise ErrorPipeline(
            "El pipeline no produjo ningún fragmento. Revisa que el PDF tenga "
            "texto legible o que el reconocimiento óptico esté instalado."
        )
    aviso(f"{len(fragmentos)} fragmentos generados")

    resultado = _insertar(fragmentos, ruta_pdf.stem, modelos, aviso)
    resultado["descartados_por_fidelidad"] = _contar_descartes(dir_trabajo)
    resultado["categoria"] = categoria
    resultado["documento"] = ruta_pdf.stem
    return resultado
