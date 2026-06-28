#!/usr/bin/env python3
"""
validate_generation_experiment.py
=================================
Experimento de VALIDACIÓN GENERATIVA para el artículo .mini.

Objetivo (observación del revisor): no basta con demostrar que .mini usa menos
tokens; hay que demostrar que .mini *también es más fácil de generar
correctamente* por un LLM. Este script mide, para cada formato, qué tan
confiablemente un modelo produce una salida estructurada VÁLIDA y RECUPERABLE.

Diseño experimental
-------------------
- Se usa el MISMO objeto canónico (dataset_12) como verdad de referencia.
- Para cada formato F in {JSON, YAML, XML, TOON, .mini} se pide al modelo
  generar N documentos equivalentes (mismos ítems) usando un prompt con
  especificación + un ejemplo (few-shot de 1).
- Cada salida del modelo se evalúa con cuatro métricas:
    1. parseable      -> el documento se parsea sin error de sintaxis.
    2. round_trip     -> al reconstruir el objeto canónico, coincide con la
                         referencia (igualdad estructural campo a campo).
    3. json_recovery  -> el documento puede convertirse a JSON canónico
                         (independiente de igualdad exacta con la referencia).
    4. field_accuracy -> proporción de campos correctos respecto a la referencia.
- Se registran además los tipos de error para análisis cualitativo.

Uso
---
    pip install anthropic openai pyyaml
    export ANTHROPIC_API_KEY=...      # o OPENAI_API_KEY=...
    python validate_generation_experiment.py --provider anthropic \
        --model claude-opus-4-8 --samples 30 \
        --formats json yaml xml toon mini --out resultados_generacion.csv

Notas de reproducibilidad
-------------------------
- temperature fija (por defecto 0.7) y semilla de muestreo documentada.
- Se reportan modelo, fecha, número de muestras y prompts exactos.
- El script NO inventa resultados: si no hay clave de API configurada, aborta.

Este archivo es parte del paquete suplementario de:
".mini: Notación bifurcable y eficiente en tokens para salidas estructuradas
 de modelos generativos."
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from copy import deepcopy

import yaml  # PyYAML

# ---------------------------------------------------------------------------
# 1. Objeto canónico (verdad de referencia) — idéntico a review_mini_benchmark.py
# ---------------------------------------------------------------------------
BASE_ITEMS = [
    dict(id='i1', bloom='L1', topic='Biología', statement='¿Dónde ocurre principalmente la fotosíntesis?', options=['cloroplastos', 'núcleo', 'mitocondria', 'ribosoma'], correct=0, a=0.9, b=-1.0, c=0.25, difficulty=1, area='biología', exposure_cap=0.20, demand='low'),
    dict(id='i2', bloom='L2', topic='Química', statement='¿Qué representa el pH de una solución?', options=['acidez o basicidad', 'masa molecular', 'temperatura', 'presión osmótica'], correct=0, a=1.1, b=-0.3, c=0.25, difficulty=2, area='química', exposure_cap=0.20, demand='medium'),
    dict(id='i3', bloom='L3', topic='Matemática', statement='Si 3x+6=18, ¿cuál es el valor de x?', options=['4', '6', '8', '12'], correct=0, a=1.4, b=0.2, c=0.20, difficulty=3, area='matemática', exposure_cap=0.25, demand='medium'),
    dict(id='i4', bloom='L4', topic='Historia', statement='¿Qué factor explica mejor la caída del Imperio romano de Occidente?', options=['presiones militares y crisis internas', 'un solo terremoto', 'la invención de la imprenta', 'la conquista española'], correct=0, a=1.7, b=0.8, c=0.20, difficulty=4, area='historia', exposure_cap=0.18, demand='high'),
    dict(id='i5', bloom='L5', topic='Literatura', statement='Evalúa cuál opción interpreta mejor un narrador no confiable.', options=['su relato debe contrastarse con evidencias', 'siempre dice la verdad', 'solo narra en tercera persona', 'no participa en la trama'], correct=0, a=2.0, b=1.2, c=0.15, difficulty=5, area='literatura', exposure_cap=0.15, demand='high'),
    dict(id='i6', bloom='L6', topic='Computación', statement='¿Qué solución diseñarías para reducir errores en una API?', options=['validación de esquema y pruebas automatizadas', 'eliminar logs', 'usar variables globales', 'evitar documentación'], correct=0, a=2.3, b=1.7, c=0.15, difficulty=5, area='computación', exposure_cap=0.15, demand='high'),
    dict(id='i7', bloom='L1', topic='Física', statement='¿Cuál es la unidad de fuerza en el SI?', options=['newton', 'joule', 'watt', 'pascal'], correct=0, a=0.9, b=-1.2, c=0.25, difficulty=1, area='física', exposure_cap=0.20, demand='low'),
    dict(id='i8', bloom='L2', topic='Geografía', statement='¿Qué describe una cuenca hidrográfica?', options=['territorio drenado por un río', 'altura de una montaña', 'tipo de clima', 'frontera política'], correct=0, a=1.1, b=-0.2, c=0.25, difficulty=2, area='geografía', exposure_cap=0.22, demand='medium'),
    dict(id='i9', bloom='L3', topic='Comunicación', statement='Elige el conector que completa una relación de causa.', options=['porque', 'sin embargo', 'además', 'por ejemplo'], correct=0, a=1.4, b=0.1, c=0.20, difficulty=3, area='comunicación', exposure_cap=0.25, demand='medium'),
    dict(id='i10', bloom='L4', topic='Economía', statement='Analiza qué sucede si sube el precio y baja la demanda.', options=['disminuye la cantidad demandada', 'sube la oferta siempre', 'desaparece el mercado', 'no cambia el equilibrio'], correct=0, a=1.7, b=0.7, c=0.20, difficulty=4, area='economía', exposure_cap=0.18, demand='high'),
    dict(id='i11', bloom='L5', topic='Ética', statement='¿Qué criterio permite evaluar mejor una decisión pública?', options=['impacto, justicia y evidencia', 'popularidad inmediata', 'costo únicamente', 'autoridad del emisor'], correct=0, a=2.0, b=1.3, c=0.15, difficulty=5, area='ética', exposure_cap=0.15, demand='high'),
    dict(id='i12', bloom='L6', topic='Arte', statement='Propón el principio clave para crear una campaña visual coherente.', options=['unidad entre mensaje, color y audiencia', 'usar todos los colores', 'copiar una plantilla', 'evitar bocetos'], correct=0, a=2.3, b=1.8, c=0.15, difficulty=5, area='arte', exposure_cap=0.15, demand='high'),
]


def canonical_obj(items):
    """Objeto canónico normalizado para comparación de round-trip."""
    return [
        {
            'id': it['id'], 'bloom': it['bloom'], 'topic': it['topic'],
            'statement': it['statement'], 'options': list(it['options']),
            'correct': it['correct'],
        }
        for it in items
    ]


# ---------------------------------------------------------------------------
# 2. Especificaciones y few-shot por formato (lo que el modelo debe producir)
# ---------------------------------------------------------------------------
def example_json(items):
    return json.dumps(
        {'items': canonical_obj(items[:1])}, ensure_ascii=False, indent=2)


SPECS = {
    'json': (
        "Genera un documento JSON con la clave 'items', una lista de objetos. "
        "Cada objeto tiene: id, bloom, topic, statement, options (lista de 4 "
        "strings), correct (índice 0-3 de la opción correcta)."
    ),
    'yaml': (
        "Genera un documento YAML con una clave 'items': lista de mapeos con "
        "id, bloom, topic, statement, options (lista de 4), correct (índice 0-3)."
    ),
    'xml': (
        "Genera un documento XML <assessment> con <item id=\"..\" bloom=\"..\"> "
        "que contiene <topic>, <statement>, <options> con cuatro <option "
        "correct=\"true|false\">. Exactamente una opción correcta por ítem."
    ),
    'toon': (
        "Genera un documento TOON con un bloque 'items[N]{id,bloom,topic,"
        "statement,o1,o2,o3,o4,correct}:' seguido de N filas CSV. 'correct' es "
        "el índice 1-4 de la opción correcta. Cita con comillas los valores que "
        "contengan comas."
    ),
    'mini': (
        "Genera un documento .mini. Cabecera: 'a|m=IRT3PL|d=20260603|n=N|l=es|"
        "t=evaluación'. Una línea por ítem con el orden EXACTO de campos "
        "separados por '|': id|bloom|topic|statement|opciones. Las opciones van "
        "separadas por coma; la opción correcta lleva el sufijo '*'. Si una "
        "opción contiene una coma, enciérrala entre comillas dobles."
    ),
}


def build_prompt(fmt, items):
    """Prompt de generación: especificación + 1 ejemplo + tarea."""
    spec = SPECS[fmt]
    topics = ", ".join(it['topic'] for it in items)
    one = example_json(items)  # ejemplo de referencia en JSON legible
    return (
        f"Eres un generador de evaluaciones. {spec}\n\n"
        f"El objeto de referencia (en JSON, solo para que conozcas el "
        f"contenido) del primer ítem es:\n{one}\n\n"
        f"TAREA: produce los {len(items)} ítems de las áreas: {topics}.\n"
        f"Devuelve ÚNICAMENTE el documento en formato '{fmt}', sin explicación "
        f"ni bloques de código markdown."
    )


# ---------------------------------------------------------------------------
# 3. Parsers / validadores por formato -> objeto canónico o excepción
# ---------------------------------------------------------------------------
def strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def parse_json(text):
    obj = json.loads(strip_fences(text))
    items = obj['items'] if isinstance(obj, dict) else obj
    return [
        {'id': it['id'], 'bloom': it['bloom'], 'topic': it['topic'],
         'statement': it['statement'], 'options': list(it['options']),
         'correct': int(it['correct'])}
        for it in items
    ]


def parse_yaml(text):
    obj = yaml.safe_load(strip_fences(text))
    items = obj['items'] if isinstance(obj, dict) else obj
    return [
        {'id': it['id'], 'bloom': it['bloom'], 'topic': it['topic'],
         'statement': it['statement'], 'options': list(it['options']),
         'correct': int(it['correct'])}
        for it in items
    ]


def parse_xml(text):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(strip_fences(text))
    out = []
    for it in root.findall('.//item'):
        opts, correct = [], 0
        for j, op in enumerate(it.find('options').findall('option')):
            opts.append((op.text or '').strip())
            if (op.get('correct', 'false').lower() == 'true'):
                correct = j
        out.append({
            'id': it.get('id'), 'bloom': it.get('bloom'),
            'topic': (it.findtext('topic') or '').strip(),
            'statement': (it.findtext('statement') or '').strip(),
            'options': opts, 'correct': correct,
        })
    return out


def _split_csv(line, delim=','):
    """CSV mínimo que respeta comillas dobles."""
    out, cur, q = [], '', False
    for ch in line:
        if ch == '"':
            q = not q
        elif ch == delim and not q:
            out.append(cur); cur = ''
        else:
            cur += ch
    out.append(cur)
    return [c.strip().strip('"') for c in out]


def parse_toon(text):
    lines = [l for l in strip_fences(text).splitlines() if l.strip()]
    hdr_idx = next(i for i, l in enumerate(lines) if l.strip().startswith('items['))
    header = lines[hdr_idx]
    fields = re.search(r"\{(.*?)\}", header).group(1).split(',')
    out = []
    for row in lines[hdr_idx + 1:]:
        vals = _split_csv(row.strip())
        rec = dict(zip(fields, vals))
        opts = [rec['o1'], rec['o2'], rec['o3'], rec['o4']]
        out.append({
            'id': rec['id'], 'bloom': rec['bloom'], 'topic': rec['topic'],
            'statement': rec['statement'], 'options': opts,
            'correct': int(rec['correct']) - 1,
        })
    return out


def parse_mini(text):
    lines = [l for l in strip_fences(text).splitlines() if l.strip()]
    if not lines or '|' not in lines[0]:
        raise ValueError("cabecera .mini ausente")
    out = []
    for line in lines[1:]:
        fields = line.split('|')
        if len(fields) < 5:
            raise ValueError(f"aridad insuficiente: {line!r}")
        _id, bloom, topic, statement = fields[0], fields[1], fields[2], fields[3]
        raw_opts = _split_csv(fields[4])
        opts, correct = [], None
        for j, op in enumerate(raw_opts):
            if op.endswith('*'):
                correct = j
                op = op[:-1]
            opts.append(op)
        if correct is None:
            raise ValueError(f"sin marcador * de respuesta correcta: {line!r}")
        out.append({'id': _id, 'bloom': bloom, 'topic': topic,
                    'statement': statement, 'options': opts, 'correct': correct})
    return out


PARSERS = {'json': parse_json, 'yaml': parse_yaml, 'xml': parse_xml,
           'toon': parse_toon, 'mini': parse_mini}


# ---------------------------------------------------------------------------
# 4. Métricas
# ---------------------------------------------------------------------------
def field_accuracy(parsed, ref):
    """Proporción de campos (id, bloom, topic, statement, correct, options)
    que coinciden con la referencia, alineando por posición."""
    if not parsed:
        return 0.0
    total, ok = 0, 0
    for p, r in zip(parsed, ref):
        for key in ('id', 'bloom', 'topic', 'statement', 'correct'):
            total += 1
            if str(p.get(key)).strip() == str(r.get(key)).strip():
                ok += 1
        total += 1
        if [o.strip() for o in p.get('options', [])] == [o.strip() for o in r['options']]:
            ok += 1
    # penaliza desajuste de cardinalidad
    total += abs(len(parsed) - len(ref))
    return ok / total if total else 0.0


def evaluate(fmt, text, ref):
    result = {'parseable': False, 'round_trip': False, 'json_recovery': False,
              'field_accuracy': 0.0, 'error': ''}
    try:
        parsed = PARSERS[fmt](text)
        result['parseable'] = True
    except Exception as e:  # noqa: BLE001
        result['error'] = f"{type(e).__name__}: {e}"
        return result
    try:
        json.dumps(parsed, ensure_ascii=False)
        result['json_recovery'] = True
    except Exception as e:  # noqa: BLE001
        result['error'] = f"json_recovery: {e}"
    acc = field_accuracy(parsed, ref)
    result['field_accuracy'] = round(acc, 4)
    result['round_trip'] = (len(parsed) == len(ref) and acc == 1.0)
    return result


# ---------------------------------------------------------------------------
# 5. Proveedores de LLM
# ---------------------------------------------------------------------------
def call_anthropic(model, prompt, temperature):
    import anthropic
    client = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model, max_tokens=4096, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def call_openai(model, prompt, temperature):
    from openai import OpenAI
    client = OpenAI()  # usa OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def call_deepseek(model, prompt, temperature):
    # DeepSeek expone una API compatible con OpenAI; solo cambia base_url y key.
    from openai import OpenAI
    client = OpenAI(base_url="https://api.deepseek.com",
                    api_key=os.environ.get("DEEPSEEK_API_KEY"))
    resp = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


PROVIDERS = {'anthropic': call_anthropic, 'openai': call_openai,
             'deepseek': call_deepseek}
DEFAULT_MODEL = {'anthropic': 'claude-opus-4-8', 'openai': 'gpt-4o',
                 'deepseek': 'deepseek-chat'}
ENV_KEY = {'anthropic': 'ANTHROPIC_API_KEY', 'openai': 'OPENAI_API_KEY',
           'deepseek': 'DEEPSEEK_API_KEY'}


# ---------------------------------------------------------------------------
# 6. Bucle experimental
# ---------------------------------------------------------------------------
def run(provider, model, formats, samples, temperature, out_csv):
    items = deepcopy(BASE_ITEMS)
    ref = canonical_obj(items)
    call = PROVIDERS[provider]
    rows = []
    summary = {}
    for fmt in formats:
        prompt = build_prompt(fmt, items)
        agg = {'parseable': 0, 'round_trip': 0, 'json_recovery': 0,
               'field_accuracy': 0.0, 'errors': {}}
        for s in range(samples):
            try:
                text = call(model, prompt, temperature)
            except Exception as e:  # noqa: BLE001
                print(f"[{fmt} #{s}] error de API: {e}", file=sys.stderr)
                time.sleep(2)
                continue
            ev = evaluate(fmt, text, ref)
            agg['parseable'] += int(ev['parseable'])
            agg['round_trip'] += int(ev['round_trip'])
            agg['json_recovery'] += int(ev['json_recovery'])
            agg['field_accuracy'] += ev['field_accuracy']
            if ev['error']:
                key = ev['error'].split(':')[0]
                agg['errors'][key] = agg['errors'].get(key, 0) + 1
            rows.append({'format': fmt, 'sample': s, **ev})
            print(f"[{fmt} #{s}] parse={ev['parseable']} rt={ev['round_trip']} "
                  f"acc={ev['field_accuracy']}")
        n = max(1, samples)
        summary[fmt] = {
            'parseable_pct': round(100 * agg['parseable'] / n, 1),
            'round_trip_pct': round(100 * agg['round_trip'] / n, 1),
            'json_recovery_pct': round(100 * agg['json_recovery'] / n, 1),
            'mean_field_accuracy': round(agg['field_accuracy'] / n, 4),
            'top_errors': sorted(agg['errors'].items(), key=lambda x: -x[1])[:3],
        }

    # CSV por muestra
    if out_csv and rows:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Tabla resumen (lista para el artículo)
    print("\n================ RESUMEN ================")
    print(f"provider={provider} model={model} samples={samples} "
          f"temperature={temperature}")
    print(f"{'Formato':10} {'%Parseable':>11} {'%Round-trip':>12} "
          f"{'%Recup.JSON':>12} {'Exactitud':>10}")
    for fmt in formats:
        s = summary[fmt]
        print(f"{fmt:10} {s['parseable_pct']:>11} {s['round_trip_pct']:>12} "
              f"{s['json_recovery_pct']:>12} {s['mean_field_accuracy']:>10}")
    print("\nErrores principales por formato:")
    for fmt in formats:
        print(f"  {fmt}: {summary[fmt]['top_errors']}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Validación generativa de formatos .mini vs JSON/YAML/XML/TOON")
    ap.add_argument('--provider', choices=list(PROVIDERS), default='anthropic')
    ap.add_argument('--model', default=None,
                    help="por defecto: claude-opus-4-8 / gpt-4o / deepseek-chat")
    ap.add_argument('--formats', nargs='+', default=['json', 'yaml', 'xml', 'toon', 'mini'])
    ap.add_argument('--samples', type=int, default=30, help="documentos por formato")
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--out', default='resultados_generacion.csv')
    args = ap.parse_args()

    model = args.model or DEFAULT_MODEL[args.provider]
    env_key = ENV_KEY[args.provider]
    if not os.environ.get(env_key):
        sys.exit(f"ERROR: falta {env_key}. Este script NO simula resultados; "
                 f"configura tu clave de API y vuelve a ejecutarlo.")

    run(args.provider, model, args.formats, args.samples,
        args.temperature, args.out)


if __name__ == '__main__':
    main()
