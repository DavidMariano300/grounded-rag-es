"""Reporte automático de calidad del pipeline ETL.

Recorre ``data/03_semantic_tree`` y ``data/04_final_chunks`` en modo solo
lectura y agrega señales que, en la práctica, delataron los problemas de
segmentación encontrados durante la revisión manual del pipeline: fragmentos
gigantes (el detector de encabezados no encontró ninguna división real),
rutas jerárquicas vacías, y chunks que el LLM reescribió por debajo del
umbral de fidelidad. No corrige nada: es un mecanismo de alerta para no tener
que inspeccionar documento por documento a mano.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Un fragmento más largo que esto normalmente indica que el detector de
# encabezados (TÍTULO/CAPÍTULO/ARTÍCULO) no encontró ninguna división real
# dentro del documento (ver es_documento_narrativo en 02_arbol_semantico.py).
UMBRAL_FRAGMENTO_SOSPECHOSO: int = 5000


def analizar_semantic_tree(carpeta: Path) -> Dict[str, Dict[str, Any]]:
    """Agrega estadísticas por categoría a partir de ``data/03_semantic_tree``.

    Args:
        carpeta: Ruta base de los árboles semánticos generados en la etapa 2.

    Returns:
        Diccionario categoría -> estadísticas agregadas.
    """
    stats_por_categoria: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "documentos": 0,
            "fragmentos": 0,
            "fragmentos_sin_ruta": 0,
            "fragmentos_sospechosos": 0,
            "longitudes": [],
            "documentos_con_fragmento_sospechoso": [],
        }
    )

    for archivo in sorted(carpeta.rglob("*.json")):
        categoria: str = archivo.parent.name
        try:
            fragmentos: List[Dict[str, Any]] = json.loads(archivo.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        datos: Dict[str, Any] = stats_por_categoria[categoria]
        datos["documentos"] += 1
        tiene_sospechoso: bool = False

        for fragmento in fragmentos:
            texto: str = fragmento.get("texto_fragmento", "")
            datos["fragmentos"] += 1
            datos["longitudes"].append(len(texto))
            if not fragmento.get("ruta_jerarquica"):
                datos["fragmentos_sin_ruta"] += 1

            # Los fragmentos narrativos (actas/gacetas) son un único bloque
            # grande por diseño (ver segmentar_documento_narrativo en
            # 02_arbol_semantico.py): el fraccionamiento real ocurre después,
            # en la etapa de chunking. Marcarlos como "sospechosos" por su
            # longitud sería una falsa alarma en cada corrida.
            es_narrativo: bool = fragmento.get("tipo_documento") == "narrativo"
            if not es_narrativo and len(texto) > UMBRAL_FRAGMENTO_SOSPECHOSO:
                datos["fragmentos_sospechosos"] += 1
                tiene_sospechoso = True

        if tiene_sospechoso:
            datos["documentos_con_fragmento_sospechoso"].append(archivo.stem)

    return stats_por_categoria


def contar_flags_fidelidad(carpeta_chunks: Path) -> List[Dict[str, Any]]:
    """Lee el log de auditoría de fidelidad LLM (``_fidelity_flags.jsonl``), si existe.

    Args:
        carpeta_chunks: Carpeta base de los chunks finales (etapa 3/4).

    Returns:
        Lista de registros de chunks que reprobaron la barrera de fidelidad.
    """
    ruta_log: Path = carpeta_chunks / "_fidelity_flags.jsonl"
    if not ruta_log.exists():
        return []

    registros: List[Dict[str, Any]] = []
    for linea in ruta_log.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea:
            continue
        try:
            registros.append(json.loads(linea))
        except json.JSONDecodeError:
            continue
    return registros


def generar_reporte_markdown(
    stats_por_categoria: Dict[str, Dict[str, Any]],
    flags_fidelidad: List[Dict[str, Any]],
) -> str:
    """Construye el reporte en Markdown a partir de las estadísticas agregadas.

    Args:
        stats_por_categoria: Salida de ``analizar_semantic_tree``.
        flags_fidelidad: Salida de ``contar_flags_fidelidad``.

    Returns:
        Contenido Markdown completo del reporte.
    """
    lineas: List[str] = ["# Reporte de calidad del pipeline ETL", ""]

    lineas.append("## Por categoría (data/03_semantic_tree)")
    lineas.append("")
    lineas.append(
        "| Categoría | Documentos | Fragmentos | % sin ruta | "
        f"Sospechosos (>{UMBRAL_FRAGMENTO_SOSPECHOSO} car.) | "
        "Longitud prom. | Longitud máx. |"
    )
    lineas.append("|---|---:|---:|---:|---:|---:|---:|")

    documentos_a_revisar: List[str] = []

    for categoria in sorted(stats_por_categoria):
        datos: Dict[str, Any] = stats_por_categoria[categoria]
        longitudes: List[int] = datos["longitudes"]
        total_fragmentos: int = datos["fragmentos"] or 1
        pct_sin_ruta: float = 100 * datos["fragmentos_sin_ruta"] / total_fragmentos
        promedio: float = sum(longitudes) / len(longitudes) if longitudes else 0.0
        maximo: int = max(longitudes) if longitudes else 0

        lineas.append(
            f"| {categoria} | {datos['documentos']} | {datos['fragmentos']} | "
            f"{pct_sin_ruta:.1f}% | {datos['fragmentos_sospechosos']} | "
            f"{promedio:.0f} | {maximo} |"
        )
        for doc in datos["documentos_con_fragmento_sospechoso"]:
            documentos_a_revisar.append(f"{categoria}/{doc}")

    lineas.append("")
    lineas.append("## Documentos con fragmentos sospechosos")
    lineas.append("")
    if documentos_a_revisar:
        for doc in documentos_a_revisar:
            lineas.append(f"- {doc}")
    else:
        lineas.append("Ninguno.")

    lineas.append("")
    lineas.append("## Flags de fidelidad LLM (data/04_final_chunks/_fidelity_flags.jsonl)")
    lineas.append("")
    if flags_fidelidad:
        lineas.append(
            f"Total de chunks que reprobaron la barrera de fidelidad: {len(flags_fidelidad)}"
        )
        lineas.append("")
        lineas.append("| Documento | Chunk | Similitud |")
        lineas.append("|---|---:|---:|")
        for registro in flags_fidelidad:
            lineas.append(
                f"| {registro.get('documento_origen', '?')} | "
                f"{registro.get('chunk_numero', '?')} | "
                f"{registro.get('ratio_similitud', '?')} |"
            )
    else:
        lineas.append("Ninguno registrado (o el pipeline de chunking aún no se ha ejecutado).")

    return "\n".join(lineas) + "\n"


def main() -> None:
    """Genera y guarda el reporte de calidad del pipeline ETL.

    Returns:
        None.
    """
    carpeta_semantic_tree = Path("./data/03_semantic_tree")
    carpeta_chunks = Path("./data/04_final_chunks")
    ruta_salida = Path("./data/quality_report.md")

    if not carpeta_semantic_tree.exists():
        print(f"No existe {carpeta_semantic_tree}; nada que analizar.")
        return

    stats_por_categoria: Dict[str, Dict[str, Any]] = analizar_semantic_tree(carpeta_semantic_tree)
    flags_fidelidad: List[Dict[str, Any]] = contar_flags_fidelidad(carpeta_chunks)
    reporte: str = generar_reporte_markdown(stats_por_categoria, flags_fidelidad)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(reporte, encoding="utf-8")

    print(reporte)
    print(f"Reporte guardado en: {ruta_salida}")


if __name__ == "__main__":
    main()
