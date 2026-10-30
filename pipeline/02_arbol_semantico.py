"""Construcción de árbol jerárquico semántico para documentos normativos.

Este módulo transforma archivos JSON previamente extraídos en una colección de
fragmentos enriquecidos con ruta jerárquica, utilizando metadatos visuales del
texto y heurísticas léxico-estructurales orientadas a documentos legales y
administrativos.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Patrón de encabezado legal compartido entre el detector de nodos jerárquicos
# y el clasificador narrativo-vs-articulado (evita duplicar la regex).
PATRON_ENCABEZADO_LEGAL: str = (
    r"(?i)^\s*(T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|ART[ÍI]CULO|MARCO|"
    r"PRESENTACI[ÓO]N|GLOSARIO|PRINCIPIOS|OBJETIVOS?|FASES|COMPETENCIA|"
    r"ANEXO|[IVXLCDMlM]+\.)\b"
)

# Categoría cuyos documentos son actas/gacetas en prosa, no reglamentos.
CATEGORIA_NARRATIVA: str = "99_GACETAS_HISTORICAS"


def obtener_tamano_fuente_base(bloques: List[Dict[str, Any]]) -> int:
    """Calcula el tamaño de fuente base del documento mediante la moda.

    Args:
        bloques: Lista de bloques textuales que incluyen metadatos de fuente.

    Returns:
        Tamaño de fuente modal del documento. Si no existen tamaños válidos,
        retorna el valor por defecto 8.
    """
    tamanos: List[float] = [
        b["font_size"]
        for b in bloques
        if "font_size" in b and isinstance(b.get("font_size"), (int, float))
    ]
    if not tamanos:
        return 8

    frecuencia: Counter = Counter(tamanos)
    tamano_base: int = frecuencia.most_common(1)[0][0]
    logger.info("Tamaño de fuente base determinado: %s", tamano_base)
    return tamano_base


def es_basura_indice(texto: str) -> bool:
    """Evalúa si un texto corresponde a ruido típico de índices o paginación.

    Args:
        texto: Cadena textual a evaluar.

    Returns:
        True si el texto coincide con patrones de ruido estructural; en caso
        contrario, False.
    """
    texto_limpio: str = texto.strip()

    patrones_basura: List[str] = [
        r"^[ \*\.\-\•]+\s+[\d\.\-]+\s*$",
        r"^[Pp]ágina\s+\d+",
        r"^OA[.,]?\s*\d+\s*AY$",
        r"(?i)(cnica|coccion|connic|cococ|occnn|nininn|nnn|cnc|ncn|nni)",
    ]

    for patron in patrones_basura:
        if re.search(patron, texto_limpio):
            return True
    return False


def es_nodo_jerarquico_valido(bloque: Dict[str, Any], tamano_base: float) -> bool:
    """Valida si un bloque puede ser interpretado como encabezado jerárquico.

    La decisión se fundamenta en reglas de exclusión léxica, restricciones de
    longitud, un filtro regex orientado al dominio legal y una validación visual
    basada en capitalización, puntuación de continuidad y tamaño tipográfico.

    Args:
        bloque: Bloque textual con metadatos visuales.
        tamano_base: Tamaño de fuente base del documento.

    Returns:
        True si el bloque satisface las heurísticas de encabezado; False en caso
        contrario.
    """
    font_size: float = bloque.get("font_size", tamano_base)
    texto: str = bloque.get("texto", "").strip()
    texto_upper: str = texto.upper()

    if any(
        texto_upper.startswith(grado)
        for grado in ["M.C.", "M.A.", "DR.", "DRA.", "LIC.", "C.P.", "ING.", "S.C."]
    ):
        return False

    if "DEFENSORÍA" in texto_upper:
        return False

    if len(texto.split()) > 10:
        return False

    match_regex = re.match(PATRON_ENCABEZADO_LEGAL, texto)

    if match_regex:
        # El vocabulario legal (TÍTULO/CAPÍTULO/SECCIÓN/ARTÍCULO/...) es la
        # señal más fuerte de encabezado: se acepta sin exigir tamaño de
        # fuente mayor al del cuerpo. En documentos como el Estatuto General,
        # los encabezados comparten el mismo tamaño que el cuerpo y solo se
        # distinguen por negrita (o ni eso); exigir tamaño mayor los rechazaba
        # a todos y colapsaba el documento en un fragmento gigante. Se
        # conserva el filtro de puntuación de continuación para descartar
        # menciones dentro de una enumeración (p. ej. "Título Décimo de la
        # Ley Orgánica;") o números de artículo dentro del cuerpo ("Artículo
        # 13."), que deliberadamente no se promueven a su propio nivel.
        if texto.endswith((",", ";", ":", ".")):
            return False
        return True

    # Sin coincidencia léxica, solo queda la señal visual (mayúsculas). Por sí
    # sola no basta: frases sueltas en mayúsculas dentro de una cita textual
    # (actas), o palabras del masthead de portada ("GACETA", "UNIVERSITARIA"),
    # también están en mayúsculas sin ser encabezados. Se exige además respaldo
    # tipográfico (tamaño de fuente mayor al del cuerpo).
    if not texto.isupper():
        return False

    texto_evaluacion: str = texto.replace("Il.", "II.").replace("IM.", "III.")
    if not texto_evaluacion.isupper():
        return False

    # El respaldo tipográfico solo es confiable en bloques nativos: el
    # font_size de un bloque de OCR es una altura en píxeles a la resolución
    # de rasterizado, no un tamaño en puntos, y prácticamente cualquier
    # palabra de una portada escaneada ("GACETA", "UNIVERSITARIA") supera
    # trivialmente el umbral con esa unidad. Sin este filtro, cualquier
    # documento con una portada que caiga en el fallback OCR por página
    # heredaba encabezados falsos de una sola palabra.
    if bloque.get("origen", "nativo") != "nativo":
        return False

    return font_size >= tamano_base + 1.5


def clasificar_nivel_encabezado(bloque: Dict[str, Any], tamano_base: int) -> int:
    """Asigna un nivel jerárquico al encabezado según su semántica textual.

    Args:
        bloque: Bloque textual identificado como posible encabezado.
        tamano_base: Tamaño de fuente base del documento.

    Returns:
        Nivel jerárquico entero donde 1 representa la división de mayor nivel.
    """
    texto: str = bloque.get("texto", "").strip().upper()

    if texto.startswith("TÍTULO") or texto.startswith("TITULO") or texto.startswith("MARCO"):
        return 1
    elif texto.startswith("CAPÍTULO") or texto.startswith("CAPITULO"):
        return 2
    elif texto.startswith("SECCIÓN") or texto.startswith("SECCION") or re.match(r"^[IVXLCDM]+\.", texto):
        return 3
    elif texto.startswith("ARTÍCULO") or texto.startswith("ARTICULO"):
        return 4
    elif texto.isupper():
        return 2
    else:
        return 5


def es_documento_narrativo(bloques: List[Dict[str, Any]], categoria_macro: str) -> bool:
    """Determina si un documento es prosa narrativa (actas/gacetas) en vez de un
    reglamento con estructura TÍTULO/CAPÍTULO/ARTÍCULO.

    Forzar el detector de encabezados legal sobre actas de sesión genera
    encabezados falsos a partir de frases en mayúsculas dentro de citas
    textuales, colapsando el documento en fragmentos gigantes con rutas
    jerárquicas sin sentido. Este chequeo evita esa ruta para documentos que,
    por categoría o por baja densidad de marcadores legales, no tienen esa
    estructura.

    Args:
        bloques: Lista de bloques textuales del documento.
        categoria_macro: Categoría macro asignada durante la extracción.

    Returns:
        True si el documento debe tratarse como narrativo; False si debe
        procesarse con el detector de jerarquía legal.
    """
    if categoria_macro == CATEGORIA_NARRATIVA:
        return True

    total_palabras: int = sum(len(b.get("texto", "").split()) for b in bloques)
    if total_palabras < 300:
        return False

    coincidencias: int = sum(
        1
        for b in bloques
        if re.match(PATRON_ENCABEZADO_LEGAL, b.get("texto", "").strip())
    )
    umbral_minimo: float = total_palabras / 300
    return coincidencias < umbral_minimo


def extraer_metadato_sesion(bloques: List[Dict[str, Any]]) -> List[str]:
    """Extrae un contexto ligero (tipo y fecha de sesión) para documentos narrativos.

    No intenta reconstruir una jerarquía legal; solo captura la línea de tipo de
    sesión ("Sesión ordinaria...") y la fecha si aparecen entre los primeros
    bloques, para dar contexto mínimo a los fragmentos sin arriesgar falsos
    encabezados.

    Args:
        bloques: Lista de bloques textuales del documento.

    Returns:
        Lista de 0, 1 o 2 elementos para usar como ruta_jerarquica.
    """
    patron_sesion = re.compile(
        r"(?i)sesi[óo]n\s+(ordinaria|extraordinaria|solemne|permanente|virtual|presencial)"
    )
    patron_fecha = re.compile(r"(?i)\d{1,2}\s+de\s+\w+\s+de\s+\d{4}")

    contexto: List[str] = []
    for bloque in bloques[:20]:
        texto: str = bloque.get("texto", "").strip()
        if not texto:
            continue
        if not contexto and patron_sesion.search(texto):
            contexto.append(texto)
        elif len(contexto) == 1 and patron_fecha.search(texto):
            contexto.append(texto)
            break

    return contexto


def segmentar_documento_narrativo(
    bloques: List[Dict[str, Any]], nombre_documento: str
) -> List[Dict[str, Any]]:
    """Construye el fragmento continuo de un documento narrativo (acta/gaceta).

    Deliberadamente no aplica ``es_nodo_jerarquico_valido`` ni
    ``clasificar_nivel_encabezado``: sobre prosa narrativa esas heurísticas
    producen encabezados falsos. El fraccionamiento en chunks de tamaño
    manejable ocurre después, en la etapa de chunking semántico.

    Args:
        bloques: Lista de bloques textuales del documento.
        nombre_documento: Identificador del documento de origen.

    Returns:
        Lista con un único fragmento (o vacía si no hay contenido útil).
    """
    ruta_jerarquica: List[str] = extraer_metadato_sesion(bloques)
    lineas: List[str] = []

    for bloque in bloques:
        texto_limpio: str = limpiar_texto(bloque.get("texto", ""))
        if not texto_limpio or es_basura_indice(texto_limpio):
            continue
        lineas.append(texto_limpio)

    contenido: str = "\n".join(lineas).strip()
    if not contenido:
        return []

    return [
        {
            "documento": nombre_documento,
            "ruta_jerarquica": ruta_jerarquica,
            "texto_fragmento": contenido,
            "tipo_documento": "narrativo",
        }
    ]


def segmentar_documento_articulado(
    bloques: List[Dict[str, Any]], nombre_documento: str
) -> List[Dict[str, Any]]:
    """Construye los fragmentos jerarquizados de un documento con estructura legal.

    Args:
        bloques: Lista de bloques textuales del documento.
        nombre_documento: Identificador del documento de origen.

    Returns:
        Lista de fragmentos con ``ruta_jerarquica`` asignada.
    """
    tamano_base: int = obtener_tamano_fuente_base(bloques)
    niveles_activos: List[Optional[str]] = [None] * NUM_NIVELES_JERARQUICOS
    fragmentos: List[Dict[str, Any]] = []
    contenido_acumulado: str = ""

    for bloque in bloques:
        texto_original: str = bloque.get("texto", "")
        texto_limpio: str = limpiar_texto(texto_original)

        if es_basura_indice(texto_limpio):
            continue

        if not texto_limpio:
            continue

        if es_nodo_jerarquico_valido(bloque, tamano_base):
            if contenido_acumulado.strip():
                fragmentos.append(
                    {
                        "documento": nombre_documento,
                        "ruta_jerarquica": materializar_ruta(niveles_activos),
                        "texto_fragmento": contenido_acumulado.strip(),
                        "tipo_documento": "articulado",
                    }
                )
                contenido_acumulado = ""

            nivel: int = clasificar_nivel_encabezado(bloque, tamano_base)
            niveles_activos = actualizar_ruta_jerarquica(
                niveles_activos,
                texto_limpio,
                nivel,
            )
        else:
            if contenido_acumulado:
                contenido_acumulado += "\n" + texto_limpio
            else:
                contenido_acumulado = texto_limpio

    if contenido_acumulado.strip():
        fragmentos.append(
            {
                "documento": nombre_documento,
                "ruta_jerarquica": materializar_ruta(niveles_activos),
                "texto_fragmento": contenido_acumulado.strip(),
                "tipo_documento": "articulado",
            }
        )

    return fragmentos


def limpiar_texto(texto: str) -> str:
    """Normaliza texto descartando ruido mínimo y patrones de paginación.

    Args:
        texto: Texto de entrada a depurar.

    Returns:
        Texto limpio o cadena vacía si se detecta ruido descartable.
    """
    texto = texto.strip()
    if re.match(r"^OA[.,]?\s*\d+\s*AY$", texto, re.IGNORECASE):
        return ""
    if len(texto) <= 2:
        return ""
    return texto


# Niveles jerárquicos que asigna clasificar_nivel_encabezado: 1=Título,
# 2=Capítulo, 3=Sección, 4=Artículo, 5=genérico.
NUM_NIVELES_JERARQUICOS: int = 5


def actualizar_ruta_jerarquica(
    niveles_activos: List[Optional[str]], nuevo_encabezado: str, nivel: int
) -> List[Optional[str]]:
    """Actualiza el estado de niveles jerárquicos activos con un nuevo encabezado.

    Se representa el estado como una lista de tamaño fijo, una posición por
    nivel (1..NUM_NIVELES_JERARQUICOS), en vez de una lista compacta truncada
    por posición (``ruta[:nivel-1]``). La mayoría de los reglamentos de este
    corpus saltan directo de Capítulo (nivel 2) a Artículo (nivel 4) sin
    Sección (nivel 3); con una lista compacta, dos artículos consecutivos bajo
    el mismo capítulo quedaban anidados uno dentro del otro en vez de que el
    segundo reemplazara al primero, porque la posición en la lista ya no
    correspondía al nivel nominal. Con slots fijos por nivel, el reemplazo es
    correcto sin importar qué niveles intermedios estén ausentes.

    Args:
        niveles_activos: Estado previo (una entrada por nivel, o None si ese
            nivel no está activo).
        nuevo_encabezado: Encabezado recientemente identificado.
        nivel: Nivel jerárquico del nuevo encabezado (1 = mayor jerarquía).

    Returns:
        Nuevo estado de niveles activos: se fija el nivel indicado y se
        limpian todos los niveles más profundos.
    """
    nuevos_niveles: List[Optional[str]] = list(niveles_activos)
    indice: int = nivel - 1
    if indice >= len(nuevos_niveles):
        nuevos_niveles.extend([None] * (indice + 1 - len(nuevos_niveles)))

    nuevos_niveles[indice] = nuevo_encabezado
    for i in range(indice + 1, len(nuevos_niveles)):
        nuevos_niveles[i] = None

    return nuevos_niveles


def materializar_ruta(niveles_activos: List[Optional[str]]) -> List[str]:
    """Convierte el estado de niveles activos en una ruta jerárquica compacta.

    Args:
        niveles_activos: Estado de niveles (con huecos ``None`` posibles).

    Returns:
        Lista de encabezados activos, en orden, sin los niveles vacíos.
    """
    return [nivel for nivel in niveles_activos if nivel]


def construir_arbol_semantico(archivo_entrada: Path, carpeta_salida: Path) -> None:
    """Construye los fragmentos semánticos jerarquizados de un archivo JSON.

    Args:
        archivo_entrada: Ruta del archivo JSON con bloques extraídos.
        carpeta_salida: Directorio donde se almacenará el árbol semántico.

    Returns:
        None.
    """
    try:
        with open(archivo_entrada, "r", encoding="utf-8") as file:
            datos: Any = json.load(file)

        bloques: List[Dict[str, Any]] = datos.get("bloques", []) if isinstance(datos, dict) else datos
        categoria_macro: str = datos.get("categoria_macro", "") if isinstance(datos, dict) else ""
        if not bloques:
            logger.warning("No se encontraron bloques en %s", archivo_entrada.name)
            return

        if es_documento_narrativo(bloques, categoria_macro):
            logger.info(
                "%s clasificado como narrativo: se omite detección de encabezados legales.",
                archivo_entrada.name,
            )
            fragmentos: List[Dict[str, Any]] = segmentar_documento_narrativo(
                bloques, archivo_entrada.stem
            )
        else:
            fragmentos = segmentar_documento_articulado(bloques, archivo_entrada.stem)

        nombre_salida: str = archivo_entrada.with_suffix("").name + "_semantic_tree.json"
        ruta_salida: Path = carpeta_salida / nombre_salida
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)

        with open(ruta_salida, "w", encoding="utf-8") as file:
            json.dump(fragmentos, file, ensure_ascii=False, indent=2)

        logger.info(
            "Procesado %s -> %s. Fragmentos generados: %s",
            archivo_entrada.name,
            ruta_salida.name,
            len(fragmentos),
        )

    except FileNotFoundError:
        logger.error("Archivo no encontrado: %s", archivo_entrada)
    except PermissionError:
        logger.error("Permiso denegado para escribir en: %s", carpeta_salida)
    except json.JSONDecodeError:
        logger.error("Error de formato JSON en: %s", archivo_entrada)
    except Exception as exc:
        logger.error("Error procesando %s: %s", archivo_entrada, str(exc))


def main() -> None:
    """Ejecuta la construcción del árbol semántico para todas las categorías.

    Returns:
        None.
    """
    carpeta_entrada: Path = Path("./data/02_extracted_data")
    carpeta_salida: Path = Path("./data/03_semantic_tree")

    if not carpeta_entrada.exists():
        logger.error("La carpeta de entrada no existe: %s", carpeta_entrada)
        return

    try:
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        logger.info("Carpeta de salida creada/verificada: %s", carpeta_salida)
    except PermissionError:
        logger.error("Permiso denegado para crear directorios en: %s", carpeta_salida)
        return

    for subdir in carpeta_entrada.iterdir():
        if not subdir.is_dir():
            continue

        subdir_salida: Path = carpeta_salida / subdir.name
        try:
            subdir_salida.mkdir(exist_ok=True)
        except PermissionError:
            logger.error("Permiso denegado para crear subdirectorio: %s", subdir_salida)
            continue

        logger.info("Procesando categoría: %s", subdir.name)

        for archivo_json in subdir.glob("*.json"):
            logger.info("Procesando archivo: %s", archivo_json.name)
            construir_arbol_semantico(archivo_json, subdir_salida)


if __name__ == "__main__":
    main()