# Examples

These files let you run and inspect the curation pipeline without the full corpus or a database.

## `estatuto_general/`

**Input.** `estatuto_general_extracto.pdf` holds pages 11 to 14 of the *Estatuto General* of the Universidad Autónoma de Guerrero, published in *Gaceta Universitaria* No. 119. It covers the end of Título Primero (Artículos 5 to 12) and Título Segundo, Capítulos I to IV (Artículos 13 to 26). The PDF has a native text layer with bold headings, so no OCR is needed.

**Output** (`output/`), produced by `despliegue/demo_pipeline.py` with `llama3.2:3b` as the cleanup model:

| File | Stage | Content |
|---|---|---|
| `1_extracted_blocks.json` | 1 | 148 line blocks with `font_size`, `is_bold` and `origen` |
| `2_semantic_tree.json` | 2 | 13 nodes with `ruta_jerarquica` (hierarchical path) |
| `3_final_chunks.json` | 3 | 13 final fragments with the text that gets vectorized |
| `3_fidelity_flags.jsonl` | 3 | 3 cleanups rejected by the fidelity gate, with original and discarded text |

A final fragment looks like this:

```json
{
  "documento_origen": "estatuto_general_extracto",
  "ruta_jerarquica": ["Título Segundo", "Capítulo I", "Artículo 15"],
  "texto_para_vectorizar": "DOCUMENTO: estatuto_general_extracto\nJERARQUÍA: Título Segundo > Capítulo I > Artículo 15\nCONTENIDO:\nEl desarrollo de las Unidades Académicas tendrá en la planeación integral su eje central, el cual será: ..."
}
```

The first entry in `3_fidelity_flags.jsonl` is the clearest case of what the gate is for. Given the heading "De la Estructura y Organización de la Universidad", the model returned it followed by two sentences about departments and department heads that the Estatuto does not contain. Similarity was 0.307, below the 0.55 threshold, so the invented text was dropped and the original heading was stored.

## Run it yourself

Requires Tesseract with Spanish, Poppler and Ollama with `llama3.2:3b` (see the main README). No database is needed.

```bash
python despliegue/demo_pipeline.py examples/estatuto_general/estatuto_general_extracto.pdf \
       --categoria 00_MARCO_CONSTITUCIONAL --sin-pausas --conservar
```

Results go to `data/demo_pipeline/`. Stages 1 and 2 are deterministic and should match `output/` exactly. Stage 3 calls a language model, so cleaned text and the number of rejected cleanups can vary slightly between runs and machines.

## What this example shows about current limits

The output is published as produced, without manual fixes, because it shows where the pipeline still falls short:

- **Some articles do not get their own level.** Stage 2 does not promote a heading that ends in a period, a rule meant to skip in-text references. This document writes most article headings as `Artículo 13.`, so Artículos 13–14, 18 and 22–24 stay inside their chapter fragment, and Artículos 10–12, 17 and 21 are merged into the fragment of the previous article. Citations for those articles resolve to the chapter, not the article.
- **The gate let an omission through.** The first half of Capítulo III (1,469 characters) came back from the cleanup as 645 characters. Artículo 23 and the opening line of Artículo 24 were dropped. Similarity (0.61) and length ratio (0.44) were both inside the limits, so the shortened text was accepted, and Artículo 23 does not appear in `3_final_chunks.json`. The gate was tuned to catch rewrites and additions; a length floor of 0.4 is too permissive to catch omissions.
- **The first fragment has an empty path.** The heading of Título Primero is on a page outside the extract.
- **Artículo 26 is cut** at the end of page 14.
- **One accepted cleanup added a header.** In the Artículo 26 fragment the model prepended the document name and the path as separate lines. The text of the article itself is intact, so similarity stayed above the threshold. The gate checks closeness to the source, not meaning.
- **Leading punctuation.** Some fragments start with `. ` because the period after the article number stays with the body (Artículo 19, Capítulo II).

Both problems, article-level segmentation for headings written as `Artículo N.` and omission detection in the gate, are on the roadmap.

## Source and license

The normative text belongs to the Universidad Autónoma de Guerrero and was taken from its official gazette. Under Mexican copyright law (Ley Federal del Derecho de Autor, art. 14, fr. VIII), legal and regulatory texts are not protected by copyright. The pipeline outputs in `output/` are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
