# -*- coding: utf-8 -*-
"""Demostración paso a paso del proceso ETL sobre un solo PDF.

Corre las tres etapas de transformación y, al terminar cada una, muestra en
pantalla qué entró y qué salió. Sirve para entender el pipeline viéndolo
funcionar, sin tener que leer el código.

No toca la base de datos. Todo ocurre en una carpeta temporal que se puede
borrar al final.

Uso:
    python despliegue/demo_pipeline.py mi_reglamento.pdf
    python despliegue/demo_pipeline.py mi_reglamento.pdf --categoria 01_NORMATIVA_ACADEMICA_ESTUDIANTIL
    python despliegue/demo_pipeline.py mi_reglamento.pdf --sin-pausas
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List

RAIZ = Path(__file__).resolve().parent.parent
PIPELINE = RAIZ / "pipeline"

AZUL, VERDE, AMBAR, ROJO, GRIS, FIN = (
    "\033[94m", "\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[0m")
ANCHO = 78


def titulo(numero: str, texto: str) -> None:
    print(f"\n{AZUL}{'=' * ANCHO}{FIN}")
    print(f"{AZUL}  ETAPA {numero}: {texto}{FIN}")
    print(f"{AZUL}{'=' * ANCHO}{FIN}")


def nota(texto: str) -> None:
    for linea in textwrap.wrap(texto, ANCHO - 4):
        print(f"  {linea}")


def clave(etiqueta: str, valor: Any) -> None:
    print(f"  {GRIS}{etiqueta:<26}{FIN} {valor}")


def caja(titulo_caja: str, contenido: str, color: str = GRIS) -> None:
    print(f"\n  {color}┌─ {titulo_caja} {'─' * max(0, ANCHO - len(titulo_caja) - 8)}{FIN}")
    for linea in contenido.split("\n"):
        for trozo in textwrap.wrap(linea, ANCHO - 8) or [""]:
            print(f"  {color}│{FIN} {trozo}")
    print(f"  {color}└{'─' * (ANCHO - 4)}{FIN}")


def pausa(activa: bool) -> None:
    if activa:
        input(f"\n  {AMBAR}Presiona Enter para continuar...{FIN}")


def correr(script: str, dir_trabajo: Path) -> float:
    inicio = time.time()
    proceso = subprocess.run([sys.executable, str(PIPELINE / script)],
                             cwd=str(dir_trabajo), capture_output=True, text=True,
                             timeout=3600)
    if proceso.returncode != 0:
        cola = (proceso.stderr or proceso.stdout or "").strip().splitlines()[-15:]
        print(f"\n  {ROJO}La etapa falló:{FIN}")
        for linea in cola:
            print(f"    {linea}")
        sys.exit(1)
    return time.time() - inicio


def leer_json(ruta: Path) -> Any:
    return json.loads(ruta.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Demostración del proceso ETL")
    ap.add_argument("pdf", help="Ruta del PDF a procesar")
    ap.add_argument("--categoria", default="01_NORMATIVA_ACADEMICA_ESTUDIANTIL")
    ap.add_argument("--sin-pausas", action="store_true")
    ap.add_argument("--conservar", action="store_true",
                    help="No borra la carpeta temporal al terminar")
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    if not pdf.exists():
        print(f"{ROJO}No se encontró el archivo {pdf}{FIN}")
        sys.exit(1)

    con_pausas = not args.sin_pausas
    trabajo = RAIZ / "data" / "demo_pipeline"
    if trabajo.exists():
        shutil.rmtree(trabajo)
    entrada = trabajo / "data" / "01_raw_pdf" / args.categoria
    entrada.mkdir(parents=True)
    shutil.copy2(pdf, entrada / pdf.name)

    print(f"\n{AZUL}{'=' * ANCHO}{FIN}")
    print(f"{AZUL}  DEMOSTRACIÓN DEL PROCESO ETL{FIN}")
    print(f"{AZUL}  Módulo normativo · Plataforma Integral de IA de la UAGro{FIN}")
    print(f"{AZUL}{'=' * ANCHO}{FIN}")
    clave("Documento", pdf.name)
    clave("Tamaño", f"{pdf.stat().st_size / 1024:.0f} KB")
    clave("Categoría", args.categoria)
    clave("Carpeta de trabajo", trabajo)
    nota("\nNada de esto toca la base de datos. Es solo para ver el proceso.")
    pausa(con_pausas)

    # ------------------------------------------------------------- etapa 1
    titulo("1", "Extracción del texto")
    nota("Abre el PDF y saca el texto. Primero revisa si el texto se puede "
         "copiar; si el PDF es una imagen escaneada, aplica reconocimiento "
         "óptico. Guarda cada línea junto con su tamaño de letra, dato que la "
         "siguiente etapa necesita para saber qué es un encabezado.")
    print()
    segundos = correr("01_extraccion.py", trabajo)
    salida1 = next((trabajo / "data" / "02_extracted_data").rglob("*.json"))
    datos1 = leer_json(salida1)
    bloques: List[Dict[str, Any]] = datos1.get("bloques", [])

    print(f"  {VERDE}Terminó en {segundos:.1f} segundos{FIN}")
    clave("Archivo generado", salida1.name)
    clave("Líneas extraídas", len(bloques))
    origenes = {b.get("origen", "?") for b in bloques}
    clave("Método de extracción", ", ".join(sorted(origenes)))
    tamanos = [b.get("font_size", 0) for b in bloques if b.get("font_size")]
    if tamanos:
        comun = max(set(tamanos), key=tamanos.count)
        clave("Tamaño de letra más común", f"{comun} (es el cuerpo del texto)")

    muestra = [b for b in bloques if b.get("is_bold") or
               (tamanos and b.get("font_size", 0) > comun)][:2] or bloques[:2]
    for b in muestra:
        caja(f"línea con letra de {b.get('font_size')} pt"
             f"{', en negritas' if b.get('is_bold') else ''}",
             json.dumps(b, ensure_ascii=False, indent=2), AMBAR)
    nota("\nEsas líneas con letra más grande o en negritas son las que la "
         "siguiente etapa va a tratar como encabezados.")
    pausa(con_pausas)

    # ------------------------------------------------------------- etapa 2
    titulo("2", "Reconstrucción de la jerarquía")
    nota("Con el tamaño de letra y con patrones del lenguaje legal (TÍTULO, "
         "CAPÍTULO, ARTÍCULO, números romanos) deduce la estructura del "
         "documento. A cada pedazo de texto le pone su dirección completa.")
    print()
    segundos = correr("02_arbol_semantico.py", trabajo)
    salida2 = next((trabajo / "data" / "03_semantic_tree").rglob("*.json"))
    datos2 = leer_json(salida2)
    if isinstance(datos2, dict):
        datos2 = [datos2]

    print(f"  {VERDE}Terminó en {segundos:.1f} segundos{FIN}")
    clave("Fragmentos jerarquizados", len(datos2))
    tipos = {d.get("tipo_documento", "?") for d in datos2}
    clave("Tipo detectado", ", ".join(sorted(tipos)))

    rutas = [" > ".join(d.get("ruta_jerarquica", [])) for d in datos2
             if d.get("ruta_jerarquica")]
    if rutas:
        clave("Profundidad máxima", max(len(r.split(" > ")) for r in rutas))
        print(f"\n  {GRIS}Primeras rutas detectadas:{FIN}")
        for r in rutas[:6]:
            print(f"    {VERDE}{r}{FIN}")

    if datos2:
        d = datos2[0]
        caja("un fragmento jerarquizado",
             json.dumps({k: (v[:160] + "..." if isinstance(v, str) and len(v) > 160 else v)
                         for k, v in d.items()}, ensure_ascii=False, indent=2), AMBAR)
    nota("\nCompara con la etapa anterior: antes había líneas sueltas, ahora "
         "cada texto sabe bajo qué título y qué capítulo vive. Eso es lo que "
         "después permite citar la fuente exacta en una respuesta.")
    pausa(con_pausas)

    # ------------------------------------------------------------- etapa 3
    titulo("3", "Fragmentación y barrera de fidelidad")
    nota("Corta los fragmentos largos y los limpia con un modelo de lenguaje "
         "local. Después compara el texto limpio contra el original: si no se "
         "parece lo suficiente, descarta la limpieza y conserva el original. "
         "Esta etapa es la más lenta porque consulta al modelo.")
    print()
    print(f"  {GRIS}Procesando. Puede tardar varios minutos...{FIN}")
    segundos = correr("03_chunker_semantico.py", trabajo)
    carpeta3 = trabajo / "data" / "04_final_chunks"
    finales: List[Dict[str, Any]] = []
    for arch in carpeta3.rglob("*.json"):
        if arch.name.startswith("_"):
            continue
        d = leer_json(arch)
        finales.extend(d if isinstance(d, list) else [d])

    print(f"  {VERDE}Terminó en {segundos:.1f} segundos{FIN}")
    clave("Fragmentos finales", len(finales))

    marca = carpeta3 / "_fidelity_flags.jsonl"
    descartes = 0
    if marca.exists():
        lineas = [l for l in marca.read_text(encoding="utf-8").splitlines() if l.strip()]
        descartes = len(lineas)
    clave("Descartados por la barrera", descartes)
    if len(finales):
        pct = 100 * descartes / (len(finales) + descartes)
        color = VERDE if pct < 10 else AMBAR
        print(f"  {color}Eso es {pct:.1f}% del total.{FIN} "
              f"{'Normal.' if pct < 10 else 'Alto: revisa la calidad del escaneo.'}")

    if descartes and marca.exists():
        try:
            primero = json.loads([l for l in marca.read_text(encoding='utf-8').splitlines()
                                  if l.strip()][0])
            caja("un caso que la barrera rechazó",
                 json.dumps({k: (str(v)[:150] + "..." if len(str(v)) > 150 else v)
                             for k, v in primero.items()}, ensure_ascii=False, indent=2), ROJO)
            nota("\nEn este caso el modelo se alejó demasiado del original, así que "
                 "el sistema conservó el texto crudo. Eso es lo correcto: en un "
                 "reglamento es preferible una línea mal cortada que una norma "
                 "reescrita.")
        except Exception:                             # noqa: BLE001
            pass

    if finales:
        f = finales[0]
        caja("un fragmento listo para vectorizar",
             json.dumps({k: (v[:220] + "..." if isinstance(v, str) and len(v) > 220 else v)
                         for k, v in f.items()}, ensure_ascii=False, indent=2), VERDE)

    # ------------------------------------------------------------- resumen
    print(f"\n{AZUL}{'=' * ANCHO}{FIN}")
    print(f"{AZUL}  RESUMEN DE LA TRANSFORMACIÓN{FIN}")
    print(f"{AZUL}{'=' * ANCHO}{FIN}")
    clave("PDF de entrada", pdf.name)
    clave("1. Líneas extraídas", len(bloques))
    clave("2. Fragmentos con jerarquía", len(datos2))
    clave("3. Fragmentos finales", len(finales))
    clave("   Descartados por fidelidad", descartes)
    print()
    nota("Lo que sigue en el servicio real es la etapa 4: insertar estos "
         "fragmentos en PostgreSQL y calcular sus vectores. Esta demostración "
         "se detiene antes, para no tocar la base.")
    print()
    clave("Salidas intermedias en", trabajo / "data")

    if args.conservar:
        print(f"\n  {AMBAR}La carpeta de trabajo se conservó para que la revises.{FIN}")
    else:
        shutil.rmtree(trabajo, ignore_errors=True)
        print(f"\n  {GRIS}Carpeta temporal eliminada. Usa --conservar para mantenerla.{FIN}")


if __name__ == "__main__":
    main()
