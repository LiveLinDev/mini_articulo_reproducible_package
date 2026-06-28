Paquete reproducible para el paper .mini corregido

Archivos:
- dataset_12.mini: dataset canónico en formato .mini.
- dataset_12.toon: dataset equivalente en TOON.
- dataset_12.yaml: dataset equivalente en YAML.
- dataset_12_compact.json: JSON compacto usado para el benchmark principal.
- dataset_12_pretty.json: JSON indentado mostrado en el apéndice del paper.
- review_mini_benchmark.py: script que serializa el dataset, cuenta tokens con cl100k_base y o200k_base, y prueba el parser.

Uso sugerido:
pip install tiktoken pyyaml
python review_mini_benchmark.py
