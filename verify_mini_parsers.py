#!/usr/bin/env python3
"""
verify_mini_parsers.py
======================
Verificación de ROUND-TRIP de los parsers del paquete .mini.

Toma el objeto canónico (dataset_12), lo serializa a cada formato con los
serializadores de benchmark_mini.py, lo vuelve a parsear con los parsers de
validate_generation_experiment.py y comprueba que el objeto reconstruido sea
idéntico a la referencia (igualdad estructural campo a campo).

Esto constituye el conjunto mínimo de fixtures exigido en el artículo: cada
variante .mini debe demostrar parseo + round-trip antes de considerarse válida.

Uso:
    pip install pyyaml tiktoken
    python verify_mini_parsers.py
Salida: PASS/FAIL por formato y código de salida 0 si todo hace round-trip.
"""
import importlib.util
import os
import sys


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    bm = _load(os.path.join(here, "benchmark_mini.py"), "benchmark_mini")
    ex = _load(os.path.join(here, "validate_generation_experiment.py"),
               "validate_generation_experiment")

    items = bm.BASE_ITEMS
    ref = ex.canonical_obj(items)
    serializers = {
        'json': bm.json_compact,
        'yaml': bm.yaml_ser,
        'xml': bm.xml_ser,
        'toon': bm.toon_ser,
        'mini': bm.mini,
    }

    all_ok = True
    print(f"{'formato':8} {'parseable':>10} {'round_trip':>11} {'exactitud':>10}")
    for fmt, ser in serializers.items():
        text = ser(items)
        ev = ex.evaluate(fmt, text, ref)
        ok = ev['parseable'] and ev['round_trip']
        all_ok = all_ok and ok
        flag = "PASS" if ok else "FAIL"
        print(f"{fmt:8} {str(ev['parseable']):>10} {str(ev['round_trip']):>11} "
              f"{ev['field_accuracy']:>10}  {flag} {ev['error']}")

    print("\nRESULTADO:", "TODOS LOS FORMATOS HACEN ROUND-TRIP" if all_ok
          else "HAY FALLOS DE ROUND-TRIP")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
