"""Chunking semántico asistido por LLM para fragmentos jerarquizados.

Este módulo procesa árboles semánticos previamente construidos y genera chunks
finales listos para vectorización. El pipeline combina segmentación por
caracteres con una fase de depuración textual mediada por un modelo local
servido mediante Ollama.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Umbral mínimo de similitud (difflib) entre el chunk original y la versión
# pulida por el LLM. Por debajo de este valor se descarta el pulido y se
# conserva el texto original: en un corpus normativo, una reescritura que se
# aleja demasiado del original es más riesgosa que una línea mal cortada.
UMBRAL_SIMILITUD_FIDELIDAD: float = 0.55

# Razón longitud(pulido)/longitud(original) considerada plausible para una
# corrección tipográfica; fuera de este rango se asume resumen u omisión.
RANGO_LONGITUD_ACEPTABLE: Tuple[float, float] = (0.4, 2.0)


def _normalizar_para_comparacion(texto: str) -> str:
    """Normaliza texto para comparación de similitud (minúsculas, sin espacios extra)."""
    return re.sub(r"\s+", " ", texto.lower()).strip()


def evaluar_fidelidad(original: str, pulido: str) -> Tuple[bool, float]:
    """Evalúa si el texto pulido por el LLM conserva el sentido del original.

    El LLM puede corregir cortes de línea y errores tipográficos, pero no
    debería resumir, omitir o reinterpretar contenido normativo. Esta barrera
    compara el chunk crudo contra la versión pulida con similitud de
    secuencias (difflib, sin dependencias nuevas) más un chequeo de razón de
    longitud, para detectar reescrituras que se desvían demasiado.

    Args:
        original: Texto del chunk antes del pulido con LLM.
        pulido: Texto devuelto por el LLM, ya sin etiquetas de contexto.

    Returns:
        Tupla ``(es_fiel, ratio_similitud)``.
    """
    if not original.strip() or not pulido.strip():
        return False, 0.0

    ratio_similitud: float = difflib.SequenceMatcher(
        None,
        _normalizar_para_comparacion(original),
        _normalizar_para_comparacion(pulido),
    ).ratio()

    razon_longitud: float = len(pulido) / len(original)
    longitud_aceptable: bool = (
        RANGO_LONGITUD_ACEPTABLE[0] <= razon_longitud <= RANGO_LONGITUD_ACEPTABLE[1]
    )

    es_fiel: bool = ratio_similitud >= UMBRAL_SIMILITUD_FIDELIDAD and longitud_aceptable
    return es_fiel, ratio_similitud


class SemanticChunkerWithLLM:
    """Orquesta el chunking semántico y la depuración asistida por LLM.

    Attributes:
        input_dir: Directorio de entrada con árboles semánticos en JSON.
        output_dir: Directorio de salida para chunks finales procesados.
        ollama_url: Endpoint HTTP del servicio Ollama para generación.
        text_splitter: Segmentador recursivo de texto utilizado para dividir
            fragmentos largos antes de la depuración con LLM.
    """

    def __init__(
        self,
        input_dir: str = "./data/03_semantic_tree/",
        output_dir: str = "./data/04_final_chunks/",
        ollama_url: str = os.getenv("RAG_OLLAMA_URL", "http://localhost:11434/api/generate"),
    ) -> None:
        """Inicializa el componente de chunking semántico.

        Args:
            input_dir: Ruta base de entrada con archivos JSON semánticos.
            output_dir: Ruta base de salida para los chunks generados.
            ollama_url: URL del endpoint de generación de Ollama.

        Returns:
            None.
        """
        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.ollama_url: str = ollama_url
        self.log_fidelidad_path: str = os.path.join(output_dir, "_fidelity_flags.jsonl")

        os.makedirs(output_dir, exist_ok=True)

        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def pulir_chunk_con_llm(self, contexto_bruto: str) -> str:
        """Solicita al LLM local la depuración o descarte de un chunk.

        Args:
            contexto_bruto: Texto contextualizado que incluye documento,
                jerarquía y contenido del chunk.

        Returns:
            Texto depurado por el modelo sin preámbulos ni introducciones.
        """
        try:
            # Prompt reestructurado como normalizador lingüístico para evitar falsos positivos de seguridad
            prompt_instruccion: str = (
                "Actúa como un compilador de texto automatizado especializado en normalización lingüística. "
                "Tu única función es corregir errores tipográficos, líneas cortadas y problemas de formato "
                "en el bloque de texto institucional que te suministraré.\n\n"
                "REGLAS CRÍTICAS DE SALIDA:\n"
                "1. NO incluyas introducciones, preámbulos ni explicaciones (EJEMPLO: Prohibido decir 'Aquí tienes el texto:', 'He procesado el bloque:', etc.).\n"
                "2. NO agregues saludos ni comentarios finales.\n"
                "3. Si el bloque es puro ruido visual (basura óptica, caracteres aleatorios, listas vacías de nombres), responde estrictamente con la palabra: DESCARTAR.\n"
                "4. Si contiene texto válido, reescríbelo uniendo las líneas rotas. Devuelve ÚNICAMENTE el texto limpio directamente.\n\n"
                "Bloque a procesar:\n"
            ) + contexto_bruto

            payload: Dict[str, Any] = {
                "model": os.getenv("RAG_OLLAMA_MODELO_LIMPIEZA", "llama3.2:3b"),
                "prompt": prompt_instruccion,
                "stream": False,
                "options": {
                    "temperature": 0.0  # Temperatura en cero absoluto para consistencia sintáctica total
                }
            }

            logger.info("Llamando a Ollama para procesar chunk...")
            response = requests.post(
                self.ollama_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )

            if response.status_code == 200:
                response_data: Dict[str, Any] = response.json()
                texto_pulido: str = response_data.get("response", contexto_bruto).strip()
                
                # Poda quirúrgica mediante expresiones regulares en caso de que el modelo use preámbulos comunes
                patrones_preambulo = [
                    r"(?i)^aquí\s+tienes\s+.*:\s*\n*",
                    r"(?i)^he\s+procesado\s+.*:\s*\n*",
                    r"(?i)^resultado\s+de\s+.*:\s*\n*",
                    r"(?i)^bloque\s+limpio\s+.*:\s*\n*",
                    r"(?i)^aquí\s+presento\s+.*:\s*\n*"
                ]
                for patron in patrones_preambulo:
                    texto_pulido = re.sub(patron, "", texto_pulido).strip()
                
                return texto_pulido

            logger.warning(
                "Error en la respuesta de Ollama: %s - %s",
                response.status_code,
                response.text,
            )
            return contexto_bruto

        except requests.exceptions.RequestException as exc:
            logger.error("Error de conexión con Ollama: %s", str(exc))
            return contexto_bruto
        except Exception as exc:
            logger.error("Error inesperado al llamar a Ollama: %s", str(exc))
            return contexto_bruto

    def registrar_flag_fidelidad(
        self,
        doc_origen: str,
        chunk_numero: int,
        ratio_similitud: float,
        texto_original: str,
        texto_pulido: str,
    ) -> None:
        """Anexa un registro de auditoría cuando un chunk reprueba la barrera de fidelidad.

        Permite revisar manualmente los casos donde el pulido del LLM se
        descartó por desviarse demasiado del texto original.

        Args:
            doc_origen: Documento de origen del chunk.
            chunk_numero: Número de chunk dentro del fragmento.
            ratio_similitud: Similitud calculada entre original y pulido.
            texto_original: Texto del chunk antes del pulido (el que se conservó).
            texto_pulido: Texto devuelto por el LLM que fue descartado.

        Returns:
            None.
        """
        registro: Dict[str, Any] = {
            "documento_origen": doc_origen,
            "chunk_numero": chunk_numero,
            "ratio_similitud": round(ratio_similitud, 3),
            "texto_original": texto_original,
            "texto_pulido_descartado": texto_pulido,
        }
        with open(self.log_fidelidad_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(registro, ensure_ascii=False) + "\n")

    def procesar_fragmento(self, fragmento: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Procesa un fragmento del árbol semántico y produce chunks finales.

        Args:
            fragmento: Diccionario con el texto fragmentado y su ruta jerárquica.

        Returns:
            Lista de diccionarios con chunks listos para vectorización, o None si
            el fragmento no genera salida utilizable.
        """
        texto_fragmento: str = fragmento.get("texto_fragmento", "")

        if len(texto_fragmento.strip()) < 30:
            logger.debug(
                "Ignorando fragmento muy corto (%s caracteres)",
                len(texto_fragmento),
            )
            return None

        doc_origen: str = fragmento.get(
            "documento",
            fragmento.get(
                "nombre_documento",
                fragmento.get("documento_origen", "Desconocido"),
            ),
        )

        ruta_jerarquica: List[str] = fragmento.get("ruta_jerarquica", [])
        ruta_str: str = " > ".join(ruta_jerarquica) if ruta_jerarquica else "Sin jerarquía"

        chunks: List[str] = self.text_splitter.split_text(texto_fragmento)

        if not chunks:
            logger.warning("No se generaron chunks para el fragmento de %s", doc_origen)
            return None

        logger.info("Generados %s chunks para %s", len(chunks), doc_origen)
        resultados_chunks: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            logger.info("Procesando chunk %s/%s para %s", i + 1, len(chunks), doc_origen)

            contexto_bruto: str = (
                f"DOCUMENTO: {doc_origen}\n"
                f"JERARQUÍA: {ruta_str}\n"
                f"CONTENIDO: {chunk}"
            )

            respuesta_llm: str = self.pulir_chunk_con_llm(contexto_bruto)

            # Control de contingencia contra falsos positivos de censura
            provino_del_llm: bool = False
            if "LO SIENTO" in respuesta_llm.upper() or "NO PUEDO CUMPLIR" in respuesta_llm.upper():
                logger.warning("Falso positivo de seguridad detectado. Salvando texto original.")
                respuesta_limpia = chunk
            elif "DESCARTAR" in respuesta_llm.upper():
                logger.debug(
                    "Chunk %s descartado por filtro LLM para %s",
                    i + 1,
                    doc_origen,
                )
                continue
            else:
                respuesta_limpia = respuesta_llm
                provino_del_llm = True

            etiquetas_a_borrar: List[str] = [
                "DOCUMENTO:",
                "JERARQUÍA:",
                "CONTENIDO:",
                ruta_str,
            ]
            if doc_origen.lower() != "desconocido":
                etiquetas_a_borrar.append(doc_origen)

            for etiqueta in etiquetas_a_borrar:
                patron: str = rf"(?i)\s*{re.escape(etiqueta)}\s*"
                respuesta_limpia = re.sub(patron, "", respuesta_limpia)

            respuesta_limpia = respuesta_limpia.strip()

            if provino_del_llm:
                es_fiel, ratio_similitud = evaluar_fidelidad(chunk, respuesta_limpia)
                if not es_fiel:
                    logger.warning(
                        "Chunk %s/%s de %s reprobó la barrera de fidelidad "
                        "(similitud=%.2f). Se conserva el texto original.",
                        i + 1,
                        len(chunks),
                        doc_origen,
                        ratio_similitud,
                    )
                    self.registrar_flag_fidelidad(
                        doc_origen, i + 1, ratio_similitud, chunk, respuesta_limpia
                    )
                    respuesta_limpia = chunk

            texto_para_vectorizar: str = (
                f"DOCUMENTO: {doc_origen}\n"
                f"JERARQUÍA: {ruta_str}\n"
                f"CONTENIDO:\n{respuesta_limpia}"
            )

            resultado_chunk: Dict[str, Any] = {
                "documento_origen": doc_origen,
                "ruta_jerarquica": ruta_jerarquica,
                "texto_para_vectorizar": texto_para_vectorizar,
                "chunk_numero": i + 1,
                "chunk_total": len(chunks),
            }
            resultados_chunks.append(resultado_chunk)

        return resultados_chunks

    def procesar_archivo_json(self, archivo_path: str) -> List[Dict[str, Any]]:
        """Procesa un archivo JSON completo del árbol semántico.

        Args:
            archivo_path: Ruta del archivo JSON a procesar.

        Returns:
            Lista agregada de chunks generados para el archivo.
        """
        logger.info("Procesando archivo: %s", archivo_path)

        with open(archivo_path, "r", encoding="utf-8") as file:
            datos_arbol: List[Dict[str, Any]] = json.load(file)

        todos_chunks: List[Dict[str, Any]] = []
        for fragmento in datos_arbol:
            resultado_chunks = self.procesar_fragmento(fragmento)
            if resultado_chunks:
                todos_chunks.extend(resultado_chunks)

        logger.info("Total chunks generados para %s: %s", archivo_path, len(todos_chunks))
        return todos_chunks

    def obtener_ruta_salida(self, ruta_entrada: str) -> str:
        """Construye la ruta de salida preservando la estructura relativa.

        Args:
            ruta_entrada: Ruta del archivo de entrada procesado.

        Returns:
            Ruta de salida normalizada con extensión JSON.
        """
        ruta_relativa: str = os.path.relpath(ruta_entrada, self.input_dir)
        ruta_salida: str = os.path.join(self.output_dir, ruta_relativa)
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        nombre_base, _ = os.path.splitext(ruta_salida)
        ruta_salida = f"{nombre_base}.json"
        return ruta_salida

    def ejecutar(self) -> None:
        """Ejecuta el flujo completo de chunking semántico con LLM.

        Returns:
            None.
        """
        logger.info("Iniciando proceso de chunking semántico con LLM...")
        logger.info("Directorio de entrada: %s", self.input_dir)
        logger.info("Directorio de salida: %s", self.output_dir)

        archivos_procesados: int = 0
        total_chunks_generados: int = 0

        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".json"):
                    ruta_entrada: str = os.path.join(root, file)
                    ruta_salida: str = self.obtener_ruta_salida(ruta_entrada)

                    try:
                        chunks_resultantes: List[Dict[str, Any]] = self.procesar_archivo_json(
                            ruta_entrada
                        )

                        with open(ruta_salida, "w", encoding="utf-8") as output_file:
                            json.dump(chunks_resultantes, output_file, ensure_ascii=False, indent=2)

                        archivos_procesados += 1
                        total_chunks_generados += len(chunks_resultantes)

                        logger.info(
                            "Archivo procesado exitosamente: %s (%s chunks)",
                            ruta_salida,
                            len(chunks_resultantes),
                        )

                    except Exception as exc:
                        logger.error("Error procesando %s: %s", ruta_entrada, str(exc))
                        continue

        logger.info("%s", "=" * 60)
        logger.info("PROCESO COMPLETADO")
        logger.info("Archivos procesados: %s", archivos_procesados)
        logger.info("Total chunks generados: %s", total_chunks_generados)
        logger.info("Directorio de salida: %s", self.output_dir)
        logger.info("%s", "=" * 60)


def main() -> None:
    """Define el punto de entrada principal del módulo.

    Returns:
        None.
    """
    chunker = SemanticChunkerWithLLM()
    chunker.ejecutar()


if __name__ == "__main__":
    main()