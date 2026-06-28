# Paquete reproducible — `.mini`

Material suplementario del artículo:

> **`.mini`: Notación bifurcable y eficiente en tokens para salidas estructuradas de modelos generativos.**
> A. E. J. Palma Obispo, E. J. Palomino Santa Cruz, J. L. Mayta Guillermo. Universidad Peruana de Ciencias Aplicadas.

Repositorio: <https://github.com/LiveLinDev/mini_articulo_reproducible_package>

Este paquete permite **reproducir los conteos de tokens** del artículo y **verificar
que el ahorro de tokens no se obtiene a costa de ambigüedad** (parseo + round-trip),
y contiene el **experimento de validación generativa** con LLM.

## Contenido

| Archivo | Función | Criterio de aceptación |
|---|---|---|
| `dataset_12.json` (`dataset_12_compact.json` / `dataset_12_pretty.json`) | Objeto canónico de 12 ítems (verdad de referencia) | Sirve como referencia para todas las serializaciones |
| `dataset_12.mini` | Corpus de 12 ítems en `.mini` | El parser reconoce todos los registros |
| `dataset_12.toon` / `dataset_12.yaml` | Equivalentes en TOON y YAML | Mismo contenido que el objeto canónico |
| `benchmark_mini.py` | Serializa el dataset a 7 formatos y cuenta tokens con `cl100k_base` y `o200k_base` | Reproduce las Tablas y Figuras de conteo |
| `verify_mini_parsers.py` | Round-trip de cada formato contra el objeto canónico | Todos los formatos hacen round-trip (exit 0) |
| `validate_generation_experiment.py` | Experimento de validación generativa con LLM (% parseable, round-trip, recuperación JSON, exactitud) | Mide la facilidad de **generar correctamente** cada formato |

## Requisitos

```bash
pip install tiktoken pyyaml matplotlib
# para el experimento generativo, además:
pip install anthropic openai   # openai sirve también para DeepSeek
```

## 1. Reproducir los conteos de tokens

```bash
python benchmark_mini.py
```

Imprime, para `cl100k_base` y `o200k_base`, los tokens de cada formato en tamaños
`5, 10, 12, 25, 50, 100`. Con el tokenizador principal `o200k_base` y **n = 12**:

| Formato | Tokens (o200k_base, n=12) |
|---|---|
| `.mini` | **817** |
| TOON | 911 |
| JSON compacto | 1316 |
| YAML | 1608 |
| XML | 1885 |
| JSON indentado | 2099 |
| QTI-XML ligero | 3411 |

Ahorro de `.mini`: **37,9 %** frente a JSON compacto y **10,3 %** frente a TOON.

## 2. Verificar parseo y round-trip

```bash
python verify_mini_parsers.py
```

Serializa el objeto canónico, lo vuelve a parsear y comprueba igualdad campo a
campo. Sale con código `0` solo si **todos** los formatos hacen round-trip.

## 3. Validación generativa con LLM (observación clave del revisor)

Mide no solo que `.mini` use menos tokens, sino que **sea más fácil de generar
correctamente** por un modelo:

```bash
# Anthropic (por defecto)
export ANTHROPIC_API_KEY=...
python validate_generation_experiment.py \
    --provider anthropic --model claude-opus-4-8 \
    --samples 30 --formats json yaml xml toon mini \
    --out resultados_generacion.csv

# OpenAI
export OPENAI_API_KEY=...
python validate_generation_experiment.py --provider openai --model gpt-4o

# DeepSeek (API compatible con OpenAI)
export DEEPSEEK_API_KEY=...
python validate_generation_experiment.py --provider deepseek --model deepseek-chat
```

Para cada formato genera `--samples` documentos y reporta:
**% parseable**, **% round-trip**, **% recuperación a JSON** y **exactitud de
campos**, además de los tipos de error más frecuentes. El script **no simula
resultados**: requiere una clave de API válida.

## Notas de reproducibilidad

- Todas las serializaciones se derivan del **mismo** objeto canónico.
- La línea base de comparación es **JSON compacto**.
- Se reportan tokenizador, tamaños, comparadores y temperatura del experimento.
