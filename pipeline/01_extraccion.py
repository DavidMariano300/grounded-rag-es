"""Pipeline híbrido de extracción textual para documentos PDF.

Este módulo implementa un enrutador de extracción que discrimina entre
PDF nativos con texto embebido y PDF escaneados o degradados que requieren
reconocimiento óptico de caracteres (OCR). La salida se serializa en formato
JSON preservando atributos estructurales básicos a nivel de línea visual.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdf2image
import pdfplumber
import pypdf
import pytesseract
from tqdm import tqdm

try:
    import cv2
    import numpy as np

    OPENCV_AVAILABLE: bool = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV no está instalado. El preprocesamiento de imágenes no se realizará.")
    print("Instale OpenCV con: pip install opencv-python")

if OPENCV_AVAILABLE:
    from PIL import Image

# Encabezados legales usados únicamente para reconocer tablas de contenido
# que no tienen puntos guía ni número de página (ver es_pagina_indice, vía B).
PATRON_ENCABEZADO_TOC: str = r"(?im)^\s*(T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|ART[ÍI]CULO)\b"


@dataclass
class LineBlock:
    """Representa una unidad textual alineada a una línea visual del documento.

    Attributes:
        texto: Contenido textual asociado a la línea extraída.
        font_size: Tamaño tipográfico promedio estimado para la línea. Para
            bloques nativos es el tamaño real en puntos; para bloques de OCR
            es la altura del texto en píxeles a la resolución de rasterizado
            y NO es comparable directamente con un tamaño en puntos.
        is_bold: Indicador de presencia de estilo tipográfico en negrita.
        origen: "nativo" si el bloque proviene de pdfplumber, "ocr" si
            proviene de Tesseract. Distingue si font_size es una unidad
            confiable (puntos) o no (píxeles), para que la etapa de árbol
            semántico no compare tamaños entre unidades distintas.
    """

    texto: str
    font_size: float
    is_bold: bool
    origen: str = "nativo"


def setup_logging() -> None:
    """Configura el subsistema de registro del módulo.

    Returns:
        None.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def limpiar_texto_ocr(texto: str) -> str:
    """Depura texto OCR mediante heurísticas léxicas y expresiones regulares.

    La función elimina artefactos frecuentes en salidas OCR, incluyendo
    secuencias de símbolos repetidos, ruido asociado a tablas de contenido y
    líneas compuestas exclusivamente por símbolos o dígitos.

    Args:
        texto: Cadena de texto obtenida mediante OCR.

    Returns:
        Texto normalizado. Si la entrada corresponde únicamente a ruido, se
        retorna una cadena vacía.
    """
    if not texto:
        return texto

    texto = re.sub(r"([.\-_\/\\|#*])\1{2,}", " ", texto)
    texto = re.sub(r"([A-Za-z])\1{4,}", r"\1", texto)
    texto = re.sub(
        r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\s*[^\w\s]{2,}\s*\d*(?:\s*[^\w\s]\w*)?",
        " ",
        texto,
    )
    texto = re.sub(r"^[^\w\s]{2,}\s*\d*", " ", texto.strip())
    texto = re.sub(r"\d*\s*[^\w\s]{2,}$", " ", texto.strip())
    texto = re.sub(r"\s+", " ", texto).strip()

    if re.match(r"^[^\w\s]+$", texto) or re.match(r"^\d+$", texto):
        return ""

    return texto


def es_pagina_indice(texto_de_la_pagina: str) -> bool:
    """Determina si una página corresponde a un índice o tabla de contenido.

    Args:
        texto_de_la_pagina: Texto completo extraído de una página.

    Contempla dos vías de detección: (A) índice clásico con puntos guía y
    número de página, y (B) tabla de contenido sin numeración —común en
    documentos exportados desde Word— identificada por una palabra clave al
    inicio de la página seguida de varias líneas con forma de encabezado
    legal (TÍTULO/CAPÍTULO/SECCIÓN/ARTÍCULO). Sin la vía B, esas páginas se
    procesan como cuerpo del documento y cada línea del índice se confunde
    con un encabezado real, corrompiendo la ruta jerárquica de los primeros
    fragmentos del documento.

    Returns:
        True si se detectan patrones léxicos y estructurales compatibles con
        una página de índice; en caso contrario, False.
    """
    # Se normalizan saltos de línea a espacios simples antes de buscar las
    # palabras clave: en pdfplumber cada línea visual termina en "\n", así que
    # una palabra clave que ocupa su propia línea (p. ej. "Contenido\n...")
    # nunca calzaba contra un patrón que exigía espacio literal tras la
    # palabra. Sin esto, ni la vía A ni la vía B llegan a activarse.
    texto_lower: str = re.sub(r"\s+", " ", texto_de_la_pagina.lower())
    palabras_clave_indice: List[str] = [
        "índice ",
        "indice ",
        "contenido ",
        "tabla de contenido ",
        "contenidos ",
        "glosario ",
        "anexo ",
        "apéndice ",
        "apendice ",
    ]

    if not any(palabra in texto_lower for palabra in palabras_clave_indice):
        return False

    # Vía A: índice clásico con puntos guía (o símbolos equivalentes) + número.
    if re.search(r"\w\s*[.\-_\/\\]+\s*\d+", texto_lower) or re.search(
        r"\w\s*[^\w\s]{2,}\s*\d+", texto_lower
    ):
        return True

    # Vía B: tabla de contenido sin número de página. Solo se activa si la
    # palabra clave abre la página (no basta con aparecer en cualquier parte)
    # y hay al menos 3 líneas con forma de encabezado legal, lo que descarta
    # páginas de cuerpo normativo que solo mencionan "anexo" de paso.
    texto_inicio: str = texto_lower.strip()[:40]
    palabra_clave_al_inicio: bool = any(
        texto_inicio.startswith(palabra.strip()) for palabra in palabras_clave_indice
    )
    if palabra_clave_al_inicio:
        coincidencias_encabezado: int = len(
            re.findall(PATRON_ENCABEZADO_TOC, texto_de_la_pagina)
        )
        if coincidencias_encabezado >= 3:
            return True

    return False


def es_pdf_nativo_limpio(
    pdf_path: Path,
    umbral_caracteres: int = 50,
    umbral_alfanum: float = 0.70,
) -> bool:
    """Valida si un PDF puede procesarse como documento nativo limpio.

    La validación considera volumen mínimo de texto, ausencia de artefactos de
    extracción conocidos y proporción alfanumérica suficiente en una muestra
    inicial de páginas.

    Args:
        pdf_path: Ruta del archivo PDF a validar.
        umbral_caracteres: Número mínimo de caracteres requerido para aceptar
            la muestra textual como válida.
        umbral_alfanum: Proporción mínima de caracteres alfanuméricos respecto
            del total de caracteres extraídos.

    Returns:
        True si el documento satisface todos los criterios de validación;
        False si debe enrutarse por OCR.
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            total_text: str = ""
            num_pages: int = len(reader.pages)

            if num_pages == 0:
                return False

            # Se muestrean hasta 3 páginas con contenido sustantivo, escaneando
            # como máximo las primeras 10. Portadas o separadores casi en
            # blanco (frecuentes al inicio de reglamentos y gacetas) no
            # cuentan para la muestra: si se incluyeran, su ausencia de texto
            # diluye la proporción alfanumérica por debajo del umbral y manda
            # el documento COMPLETO por OCR aunque el resto sea 100% nativo.
            max_paginas_a_escanear: int = min(10, num_pages)
            paginas_con_texto: int = 0
            for i in range(max_paginas_a_escanear):
                page = reader.pages[i]
                text = page.extract_text()
                if text and len(text.strip()) >= 20:
                    total_text += text
                    paginas_con_texto += 1
                if paginas_con_texto >= 3:
                    break

            if len(total_text.strip()) < umbral_caracteres:
                logging.debug(
                    "PDF %s tiene pocos caracteres (%s). Es un escaneo.",
                    pdf_path.name,
                    len(total_text.strip()),
                )
                return False

            if "" in total_text or "cid:" in total_text:
                logging.debug(
                    "PDF %s contiene caracteres de error. Mapa ToUnicode corrupto.",
                    pdf_path.name,
                )
                return False

            alfanum_count: int = sum(1 for c in total_text if c.isalnum())
            total_char_count: int = len(total_text)
            if total_char_count == 0:
                return False

            proporcion_alfanum: float = alfanum_count / total_char_count
            if proporcion_alfanum < umbral_alfanum:
                logging.debug(
                    "PDF %s tiene baja proporción de texto alfanumérico (%.2f). "
                    "Es basura óptica.",
                    pdf_path.name,
                    proporcion_alfanum,
                )
                return False

            logging.debug(
                "PDF %s superó todas las pruebas de validación. "
                "Considerado nativo limpio.",
                pdf_path.name,
            )
            return True

    except Exception as exc:
        logging.error("Error al validar PDF nativo %s: %s", pdf_path, exc)
        return False


def es_texto_pagina_aceptable(
    texto: Optional[str],
    umbral_caracteres: int = 50,
    umbral_alfanum: float = 0.70,
) -> bool:
    """Evalúa si el texto extraído de una sola página es utilizable.

    Aplica el mismo criterio de calidad que ``es_pdf_nativo_limpio`` usa a
    nivel de documento (mínimo de caracteres, ausencia de mapa ToUnicode
    corrupto, proporción alfanumérica), pero por página individual. Permite
    detectar páginas puntuales sin texto útil (p. ej. una firma o certificado
    escaneado insertado) dentro de un documento mayormente nativo.

    Args:
        texto: Texto extraído de una única página.
        umbral_caracteres: Mínimo de caracteres para considerar la muestra válida.
        umbral_alfanum: Proporción mínima de caracteres alfanuméricos.

    Returns:
        True si el texto es aceptable; False si la página debe reprocesarse
        con OCR.
    """
    texto = texto or ""
    if len(texto.strip()) < umbral_caracteres:
        return False
    if "" in texto or "cid:" in texto:
        return False

    total_char_count: int = len(texto)
    if total_char_count == 0:
        return False

    alfanum_count: int = sum(1 for c in texto if c.isalnum())
    return (alfanum_count / total_char_count) >= umbral_alfanum


def _agrupar_ocr_por_linea(imagen_preprocesada: Any) -> List[LineBlock]:
    """Ejecuta OCR sobre una imagen de página ya preprocesada y agrupa por línea.

    Lógica compartida entre el enrutamiento OCR de documento completo
    (``extraer_con_pytesseract``) y el fallback OCR por página dentro de la
    ruta nativa (``extraer_pagina_ocr``), para no preprocesar la misma imagen
    dos veces ni duplicar el agrupamiento de bloques de Tesseract.

    Args:
        imagen_preprocesada: Imagen PIL ya pasada por ``preprocesar_imagen_cv2``.

    Returns:
        Lista de ``LineBlock`` reconocidos ópticamente para la página.
    """
    config_ocr: str = "-l spa"
    data: Dict[str, List[Any]] = pytesseract.image_to_data(
        imagen_preprocesada,
        config=config_ocr,
        output_type=pytesseract.Output.DICT,
    )

    grouped_lines: Dict[Tuple[int, int, int], Dict[str, List[Any]]] = {}
    n_boxes: int = len(data["text"])

    for i in range(n_boxes):
        text_val: str = data["text"][i].strip()
        if not text_val:
            continue
        block_id: int = data["block_num"][i]
        par_id: int = data["par_num"][i]
        line_id: int = data["line_num"][i]
        key: Tuple[int, int, int] = (block_id, par_id, line_id)

        if key not in grouped_lines:
            grouped_lines[key] = {"texts": [], "heights": []}

        grouped_lines[key]["texts"].append(text_val)
        height_val: int = data["height"][i]
        if height_val > 0:
            grouped_lines[key]["heights"].append(height_val)

    line_blocks: List[LineBlock] = []
    for _, line_data in grouped_lines.items():
        if not line_data["texts"]:
            continue

        line_text: str = limpiar_texto_ocr(" ".join(line_data["texts"]))
        if not line_text.strip():
            continue

        avg_height: float = (
            sum(line_data["heights"]) / len(line_data["heights"])
            if line_data["heights"]
            else 0.0
        )
        line_blocks.append(
            LineBlock(
                texto=line_text,
                font_size=round(avg_height, 2),
                is_bold=False,
                origen="ocr",
            )
        )

    return line_blocks


def extraer_pagina_ocr(pdf_path: Path, page_num: int, dpi_alto: int = 400) -> List[LineBlock]:
    """Aplica OCR de respaldo a una única página dentro de la ruta nativa.

    Se invoca cuando un documento es mayormente nativo (ruta A) pero una
    página puntual no produce texto utilizable con pdfplumber —por ejemplo
    una imagen, firma o certificado escaneado insertado entre páginas de
    texto—, evitando perder esa página por completo.

    Args:
        pdf_path: Ruta del documento PDF.
        page_num: Número de página (1-indexado) a reprocesar con OCR.
        dpi_alto: Resolución utilizada en la rasterización.

    Returns:
        Lista de ``LineBlock`` reconocidos ópticamente para esa página, o
        lista vacía si la página resulta ser un índice o falla el OCR.
    """
    try:
        paginas = pdf2image.convert_from_path(
            str(pdf_path), dpi=dpi_alto, first_page=page_num, last_page=page_num
        )
        if not paginas:
            return []

        imagen_preprocesada = preprocesar_imagen_cv2(paginas[0])

        texto_pagina_ocr: str = pytesseract.image_to_string(
            imagen_preprocesada, lang="spa"
        ).strip()
        if texto_pagina_ocr and es_pagina_indice(texto_pagina_ocr):
            logging.info(
                "Página %s de %s identificada como índice (OCR de respaldo). "
                "Se omite su procesamiento.",
                page_num,
                pdf_path.name,
            )
            return []

        return _agrupar_ocr_por_linea(imagen_preprocesada)

    except Exception as exc:
        logging.warning(
            "Error al aplicar OCR de respaldo en página %s de %s: %s",
            page_num,
            pdf_path.name,
            exc,
        )
        return []


def _detectar_limite_columnas(
    words: List[Dict[str, Any]], ancho_pagina: float
) -> Optional[float]:
    """Detecta la coordenada x del canal entre dos columnas de texto, si existe.

    Construye un perfil de densidad horizontal (cuántas palabras cubren cada
    franja vertical de la página) y busca un valle relativo —el "canal" entre
    columnas— en el tercio central de la página. Sin esto, páginas a dos
    columnas se agrupan por línea visual (misma coordenada Y) ignorando la
    columna, mezclando en una sola línea de salida el final de una frase de
    la columna izquierda con el inicio de una frase no relacionada de la
    columna derecha. Un LLM "puliendo" ese texto revuelto tiende a inventar
    contenido plausible para darle sentido en vez de solo corregir cortes de
    línea (ver barrera de fidelidad en 03_chunker_semantico.py).

    El canal real rara vez está completamente vacío: texto justificado se
    desborda ocasionalmente hacia el margen entre columnas, y una página de
    una sola columna puede tener una densidad que decae de forma pareja hacia
    un borde (sin ser un canal real). Por eso el valle se compara contra la
    densidad de ventanas *inmediatamente* a su izquierda y derecha (no contra
    el máximo global de cada mitad de página): solo cuenta como canal si hay
    columnas de texto denso a ambos lados, cerca, con un hueco marcado entre
    medio — la forma real de dos columnas, no una caída gradual.

    Args:
        words: Palabras de la página con coordenadas ``x0``/``x1``.
        ancho_pagina: Ancho total de la página en puntos.

    Returns:
        Coordenada x del canal entre columnas, o None si la página parece de
        una sola columna.
    """
    if len(words) < 20 or ancho_pagina <= 0:
        return None

    n_bins: int = 60
    ancho_bin: float = ancho_pagina / n_bins
    densidad: List[float] = [0.0] * n_bins

    for word in words:
        bin_inicio: int = max(0, min(n_bins - 1, int(word["x0"] / ancho_bin)))
        bin_fin: int = max(0, min(n_bins - 1, int(word["x1"] / ancho_bin)))
        for b in range(bin_inicio, bin_fin + 1):
            densidad[b] += 1

    if max(densidad) == 0:
        return None

    # Zona central donde puede estar el canal (evita confundir los márgenes
    # de la página con un canal real) y ventana de comparación local a cada
    # lado del candidato a valle.
    zona_inicio, zona_fin = n_bins // 3, 2 * n_bins // 3
    ancho_ventana: int = 8

    mejor_idx: Optional[int] = None
    mejor_ratio: float = 1.0

    for i in range(zona_inicio, zona_fin):
        ventana_izquierda = densidad[max(0, i - ancho_ventana) : i]
        ventana_derecha = densidad[i + 1 : i + 1 + ancho_ventana]
        if not ventana_izquierda or not ventana_derecha:
            continue

        densidad_lados: float = min(max(ventana_izquierda), max(ventana_derecha))
        if densidad_lados == 0:
            continue

        ratio: float = densidad[i] / densidad_lados
        if ratio < mejor_ratio:
            mejor_ratio = ratio
            mejor_idx = i

    # Se exige que el valle caiga a menos del 40% de la densidad de ambos
    # lados inmediatos para contar como canal real.
    if mejor_idx is None or mejor_ratio > 0.4:
        return None

    return (mejor_idx + 0.5) * ancho_bin


def _agrupar_palabras_por_linea(words: List[Dict[str, Any]]) -> List[LineBlock]:
    """Agrupa palabras de pdfplumber en líneas visuales y arma sus ``LineBlock``.

    Args:
        words: Palabras (ya restringidas a una sola columna si corresponde)
            con atributos ``top``, ``x0``, ``text``, ``fontname`` y ``size``.

    Returns:
        Lista de ``LineBlock`` en orden de lectura de arriba hacia abajo.
    """
    lines_map: Dict[float, List[Dict[str, Any]]] = {}

    for word in words:
        top_rounded: float = round(word["top"], 1)
        if top_rounded not in lines_map:
            lines_map[top_rounded] = []
        lines_map[top_rounded].append(word)

    line_blocks: List[LineBlock] = []
    for _, words_in_line in sorted(lines_map.items()):
        words_in_line.sort(key=lambda w: w["x0"])
        line_text: str = " ".join([w["text"] for w in words_in_line])
        sizes: List[float] = [w["size"] for w in words_in_line if w.get("size", 0) > 0]
        avg_font_size: float = sum(sizes) / len(sizes) if sizes else 0.0
        bold_fonts: List[Dict[str, Any]] = [
            w for w in words_in_line if w.get("fontname") and "bold" in w["fontname"].lower()
        ]
        is_bold: bool = len(bold_fonts) > 0

        if not line_text.strip():
            continue

        line_blocks.append(
            LineBlock(texto=line_text, font_size=round(avg_font_size, 2), is_bold=is_bold)
        )

    return line_blocks


def extraer_con_pdfplumber(pdf_path: Path) -> List[LineBlock]:
    """Extrae bloques de línea desde un PDF nativo usando pdfplumber.

    La estrategia agrupa palabras por coordenada vertical redondeada para
    reconstruir líneas visuales, estimando además tamaño de fuente promedio y
    presencia de negrita a partir de atributos tipográficos disponibles.

    Args:
        pdf_path: Ruta del documento PDF nativo.

    Returns:
        Lista de instancias ``LineBlock`` correspondientes al contenido lineal
        reconstruido del documento.
    """
    all_line_blocks: List[LineBlock] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    texto_pagina_actual = page.extract_text()
                    if texto_pagina_actual and es_pagina_indice(texto_pagina_actual):
                        logging.info(
                            "Página %s de %s identificada como índice (ruta A). "
                            "Se omite su procesamiento.",
                            page_num,
                            pdf_path.name,
                        )
                        continue

                    if not es_texto_pagina_aceptable(texto_pagina_actual):
                        logging.info(
                            "Página %s de %s sin texto nativo utilizable dentro de un "
                            "documento nativo. Aplicando OCR de respaldo solo a esta página.",
                            page_num,
                            pdf_path.name,
                        )
                        all_line_blocks.extend(extraer_pagina_ocr(pdf_path, page_num))
                        continue

                    words: List[Dict[str, Any]] = page.extract_words(
                        extra_attrs=["fontname", "size"]
                    )

                    limite_columnas = _detectar_limite_columnas(words, page.width)
                    if limite_columnas is None:
                        columnas: List[List[Dict[str, Any]]] = [words]
                    else:
                        logging.info(
                            "Página %s de %s: layout a dos columnas detectado (canal en x=%.0f).",
                            page_num,
                            pdf_path.name,
                            limite_columnas,
                        )
                        columna_izquierda = [
                            w for w in words if (w["x0"] + w["x1"]) / 2 < limite_columnas
                        ]
                        columna_derecha = [
                            w for w in words if (w["x0"] + w["x1"]) / 2 >= limite_columnas
                        ]
                        columnas = [columna_izquierda, columna_derecha]

                    for columna_words in columnas:
                        all_line_blocks.extend(_agrupar_palabras_por_linea(columna_words))

                except Exception as exc:
                    logging.warning(
                        "Error al procesar página %s de %s (ruta A): %s",
                        page_num,
                        pdf_path.name,
                        exc,
                    )
                    continue

    except Exception as exc:
        logging.error(
            "Error crítico al abrir o procesar PDF con pdfplumber %s: %s",
            pdf_path,
            exc,
        )
        raise

    return all_line_blocks


def preprocesar_imagen_cv2(imagen_pil: Any) -> Any:
    """Aplica preprocesamiento con OpenCV para mejorar la entrada al OCR.

    El procedimiento preserva la lógica original: conversión a escala de grises
    seguida por binarización mediante el método de Otsu. Si OpenCV no está
    disponible, la imagen se retorna sin transformación.

    Args:
        imagen_pil: Imagen en formato PIL.

    Returns:
        Imagen preprocesada en formato PIL, o la imagen original si OpenCV no
        se encuentra disponible.
    """
    if not OPENCV_AVAILABLE:
        return imagen_pil

    imagen_np = np.array(imagen_pil)
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    _, binarizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imagen_cv2_pil = cv2.cvtColor(binarizada, cv2.COLOR_GRAY2RGB)
    imagen_final_pil = Image.fromarray(imagen_cv2_pil)

    return imagen_final_pil


def extraer_con_pytesseract(pdf_path: Path, dpi_alto: int = 400) -> List[LineBlock]:
    """Extrae bloques de línea desde un PDF escaneado mediante OCR.

    El procedimiento convierte el PDF a imágenes de alta resolución, ejecuta un
    OCR preliminar para descartar páginas de índice y posteriormente agrupa las
    salidas jerárquicas de Tesseract a nivel de línea lógica.

    Args:
        pdf_path: Ruta del documento PDF escaneado o degradado.
        dpi_alto: Resolución utilizada en la rasterización de páginas para OCR.

    Returns:
        Lista de instancias ``LineBlock`` generadas a partir del texto
        reconocido ópticamente.
    """
    all_line_blocks: List[LineBlock] = []

    try:
        pages = pdf2image.convert_from_path(str(pdf_path), dpi=dpi_alto)

        for page_num, imagen_pil in enumerate(pages, start=1):
            try:
                imagen_preprocesada = preprocesar_imagen_cv2(imagen_pil)

                texto_pagina_actual_ocr: str = pytesseract.image_to_string(
                    imagen_preprocesada,
                    lang="spa",
                ).strip()
                if texto_pagina_actual_ocr and es_pagina_indice(texto_pagina_actual_ocr):
                    logging.info(
                        "Página %s de %s identificada como índice (ruta B OCR). "
                        "Se omite su procesamiento.",
                        page_num,
                        pdf_path.name,
                    )
                    continue

                all_line_blocks.extend(_agrupar_ocr_por_linea(imagen_preprocesada))

            except Exception as exc:
                logging.warning(
                    "Error al procesar página %s con OCR en %s: %s",
                    page_num,
                    pdf_path.name,
                    exc,
                )
                continue

    except Exception as exc:
        logging.error("Error crítico al convertir PDF a imágenes %s: %s", pdf_path, exc)
        raise

    return all_line_blocks


def procesar_documento_hibrido(pdf_path: Path) -> List[LineBlock]:
    """Selecciona y ejecuta la ruta de extracción adecuada para un documento.

    Args:
        pdf_path: Ruta del archivo PDF a procesar.

    Returns:
        Lista de bloques lineales extraídos del documento.
    """
    if es_pdf_nativo_limpio(pdf_path):
        logging.info("(Ruta A) Procesando PDF nativo limpio: %s", pdf_path.name)
        return extraer_con_pdfplumber(pdf_path)

    logging.info(
        "(Ruta B) Aplicando OCR a PDF escaneado o problemático: %s",
        pdf_path.name,
    )
    return extraer_con_pytesseract(pdf_path)


def guardar_resultado_json(
    line_blocks: List[LineBlock],
    pdf_path: Path,
    output_dir: Path,
    categoria_macro: str,
) -> None:
    """Serializa el resultado de extracción en un archivo JSON.

    Args:
        line_blocks: Lista de bloques lineales extraídos del documento.
        pdf_path: Ruta del PDF fuente.
        output_dir: Directorio de salida para el archivo JSON.
        categoria_macro: Etiqueta de categoría macro asociada al documento.

    Returns:
        None.
    """
    document_data: Dict[str, Any] = {
        "nombre_documento": pdf_path.name,
        "categoria_macro": categoria_macro,
        "bloques": [
            {
                "texto": block.texto,
                "font_size": block.font_size,
                "is_bold": block.is_bold,
                "origen": block.origen,
            }
            for block in line_blocks
        ],
    }

    output_filename: str = pdf_path.stem + ".json"
    output_path: Path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(document_data, file, ensure_ascii=False, indent=2)

    logging.info(
        "Procesamiento completado para: %s -> %s",
        pdf_path.name,
        output_path.name,
    )


def procesar_single_pdf(pdf_path: Path, output_dir: Path, categoria_macro: str) -> None:
    """Procesa un único PDF y persiste su salida estructurada.

    Args:
        pdf_path: Ruta del documento PDF a procesar.
        output_dir: Directorio donde se almacenará la salida JSON.
        categoria_macro: Categoría macro asignada al documento.

    Returns:
        None.
    """
    logging.info("Procesando (Híbrido): %s", pdf_path.name)
    try:
        line_blocks = procesar_documento_hibrido(pdf_path)
        guardar_resultado_json(line_blocks, pdf_path, output_dir, categoria_macro)
    except Exception as exc:
        logging.error("Error al procesar %s: %s", pdf_path.name, exc)


def main() -> None:
    """Coordina la ejecución integral del pipeline de extracción híbrida.

    Returns:
        None.
    """
    setup_logging()

    input_base_dir: Path = Path("./data/01_raw_pdf")
    output_base_dir: Path = Path("./data/02_extracted_data")

    if not input_base_dir.exists():
        logging.error("Directorio de entrada no encontrado: %s", input_base_dir)
        sys.exit(1)

    output_base_dir.mkdir(parents=True, exist_ok=True)
    pdf_files: List[Path] = list(input_base_dir.rglob("*.pdf"))

    if not pdf_files:
        logging.warning("No se encontraron archivos PDF en %s", input_base_dir)
        return

    logging.info("Encontrados %s archivos PDF para procesar", len(pdf_files))

    for pdf_path in tqdm(pdf_files, desc="Procesando PDFs (Híbrido)"):
        try:
            relative_path: Path = pdf_path.relative_to(input_base_dir)
            categoria_macro: str = relative_path.parts[0]
            output_subdir: Path = output_base_dir / categoria_macro
            output_subdir.mkdir(exist_ok=True)
            procesar_single_pdf(pdf_path, output_subdir, categoria_macro)
        except Exception as exc:
            logging.error("Error fatal al procesar %s: %s", pdf_path, exc)
            continue

    logging.info("Proceso de extracción híbrida completado.")


if __name__ == "__main__":
    main()