import json, csv, io, yaml, html, os, math
from copy import deepcopy
import tiktoken

# Canonical 12-item dataset for reproducibility.
BASE_ITEMS = [
    dict(id='i1', bloom='L1', topic='Biología', statement='¿Dónde ocurre principalmente la fotosíntesis?', options=['cloroplastos','núcleo','mitocondria','ribosoma'], correct=0, a=0.9, b=-1.0, c=0.25, difficulty=1, area='biología', exposure_cap=0.20, demand='low'),
    dict(id='i2', bloom='L2', topic='Química', statement='¿Qué representa el pH de una solución?', options=['acidez o basicidad','masa molecular','temperatura','presión osmótica'], correct=0, a=1.1, b=-0.3, c=0.25, difficulty=2, area='química', exposure_cap=0.20, demand='medium'),
    dict(id='i3', bloom='L3', topic='Matemática', statement='Si 3x+6=18, ¿cuál es el valor de x?', options=['4','6','8','12'], correct=0, a=1.4, b=0.2, c=0.20, difficulty=3, area='matemática', exposure_cap=0.25, demand='medium'),
    dict(id='i4', bloom='L4', topic='Historia', statement='¿Qué factor explica mejor la caída del Imperio romano de Occidente?', options=['presiones militares y crisis internas','un solo terremoto','la invención de la imprenta','la conquista española'], correct=0, a=1.7, b=0.8, c=0.20, difficulty=4, area='historia', exposure_cap=0.18, demand='high'),
    dict(id='i5', bloom='L5', topic='Literatura', statement='Evalúa cuál opción interpreta mejor un narrador no confiable.', options=['su relato debe contrastarse con evidencias','siempre dice la verdad','solo narra en tercera persona','no participa en la trama'], correct=0, a=2.0, b=1.2, c=0.15, difficulty=5, area='literatura', exposure_cap=0.15, demand='high'),
    dict(id='i6', bloom='L6', topic='Computación', statement='¿Qué solución diseñarías para reducir errores en una API?', options=['validación de esquema y pruebas automatizadas','eliminar logs','usar variables globales','evitar documentación'], correct=0, a=2.3, b=1.7, c=0.15, difficulty=5, area='computación', exposure_cap=0.15, demand='high'),
    dict(id='i7', bloom='L1', topic='Física', statement='¿Cuál es la unidad de fuerza en el SI?', options=['newton','joule','watt','pascal'], correct=0, a=0.9, b=-1.2, c=0.25, difficulty=1, area='física', exposure_cap=0.20, demand='low'),
    dict(id='i8', bloom='L2', topic='Geografía', statement='¿Qué describe una cuenca hidrográfica?', options=['territorio drenado por un río','altura de una montaña','tipo de clima','frontera política'], correct=0, a=1.1, b=-0.2, c=0.25, difficulty=2, area='geografía', exposure_cap=0.22, demand='medium'),
    dict(id='i9', bloom='L3', topic='Comunicación', statement='Elige el conector que completa una relación de causa.', options=['porque','sin embargo','además','por ejemplo'], correct=0, a=1.4, b=0.1, c=0.20, difficulty=3, area='comunicación', exposure_cap=0.25, demand='medium'),
    dict(id='i10', bloom='L4', topic='Economía', statement='Analiza qué sucede si sube el precio y baja la demanda.', options=['disminuye la cantidad demandada','sube la oferta siempre','desaparece el mercado','no cambia el equilibrio'], correct=0, a=1.7, b=0.7, c=0.20, difficulty=4, area='economía', exposure_cap=0.18, demand='high'),
    dict(id='i11', bloom='L5', topic='Ética', statement='¿Qué criterio permite evaluar mejor una decisión pública?', options=['impacto, justicia y evidencia','popularidad inmediata','costo únicamente','autoridad del emisor'], correct=0, a=2.0, b=1.3, c=0.15, difficulty=5, area='ética', exposure_cap=0.15, demand='high'),
    dict(id='i12', bloom='L6', topic='Arte', statement='Propón el principio clave para crear una campaña visual coherente.', options=['unidad entre mensaje, color y audiencia','usar todos los colores','copiar una plantilla','evitar bocetos'], correct=0, a=2.3, b=1.8, c=0.15, difficulty=5, area='arte', exposure_cap=0.15, demand='high'),
]

def expand_items(n):
    out=[]
    for idx in range(n):
        item=deepcopy(BASE_ITEMS[idx % len(BASE_ITEMS)])
        item['id']=f'i{idx+1}'
        out.append(item)
    return out

def assessment_obj(items):
    return {
        'meta': {'model':'IRT3PL','date':'20260603','n':len(items),'lang':'es','topic':'evaluación transversal','bloom_distribution':[1,1,1,1,1,1], 'cat': {'theta_init':0,'theta_min':-3,'theta_max':3,'se_stop':0.3,'max_items':len(items),'exposure_ctrl':'SH'}},
        'items': [
            {
                'id': it['id'], 'bloom': it['bloom'], 'topic': it['topic'], 'statement': it['statement'],
                'options': it['options'], 'correct': it['correct'],
                'irt': {'a': it['a'], 'b': it['b'], 'c': it['c']},
                'difficulty': it['difficulty'],
                'cat': {'area': it['area'], 'exposure_cap': it['exposure_cap'], 'demand': it['demand']}
            } for it in items
        ]
    }

def csv_join(vals):
    buf=io.StringIO()
    writer=csv.writer(buf, lineterminator='', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(vals)
    return buf.getvalue()

def mini(items):
    header=f"a|m=IRT3PL|d=20260603|n={len(items)}|l=es|t=evaluación transversal|bd=1,1,1,1,1,1|cat=0,-3,3,0.3,{len(items)},SH"
    lines=[header]
    for it in items:
        opts=it['options'][:]
        opts[it['correct']]=opts[it['correct']]+'*'
        line='|'.join([
            it['id'], it['bloom'], it['topic'], it['statement'], csv_join(opts),
            csv_join([it['a'],it['b'],it['c']]), str(it['difficulty']),
            csv_join([it['area'],it['exposure_cap'],it['demand']])
        ])
        lines.append(line)
    return '\n'.join(lines)

def flat_records(items):
    rows=[]
    for it in items:
        rows.append({
            'id': it['id'], 'bloom': it['bloom'], 'topic': it['topic'], 'statement': it['statement'],
            'o1': it['options'][0], 'o2': it['options'][1], 'o3': it['options'][2], 'o4': it['options'][3],
            'correct': it['correct']+1, 'a': it['a'], 'b': it['b'], 'c': it['c'], 'difficulty': it['difficulty'],
            'area': it['area'], 'exposure_cap': it['exposure_cap'], 'demand': it['demand']
        })
    return rows

def json_compact(items):
    return json.dumps(assessment_obj(items), ensure_ascii=False, separators=(',',':'))

def json_pretty(items):
    return json.dumps(assessment_obj(items), ensure_ascii=False, indent=2)

def yaml_ser(items):
    return yaml.safe_dump(assessment_obj(items), allow_unicode=True, sort_keys=False, default_flow_style=False)

def xml_ser(items):
    root=[f'<assessment model="IRT3PL" date="20260603" n="{len(items)}" lang="es" topic="evaluación transversal">']
    root.append(f'  <cat theta_init="0" theta_min="-3" theta_max="3" se_stop="0.3" max_items="{len(items)}" exposure_ctrl="SH"/>')
    for it in items:
        root.append(f'  <item id="{it["id"]}" bloom="{it["bloom"]}" difficulty="{it["difficulty"]}">')
        root.append(f'    <topic>{html.escape(it["topic"])}</topic>')
        root.append(f'    <statement>{html.escape(it["statement"])}</statement>')
        root.append('    <options>')
        for j,opt in enumerate(it['options']):
            root.append(f'      <option correct="{str(j==it["correct"]).lower()}">{html.escape(opt)}</option>')
        root.append('    </options>')
        root.append(f'    <irt a="{it["a"]}" b="{it["b"]}" c="{it["c"]}"/>')
        root.append(f'    <cat area="{html.escape(it["area"])}" exposure_cap="{it["exposure_cap"]}" demand="{it["demand"]}"/>')
        root.append('  </item>')
    root.append('</assessment>')
    return '\n'.join(root)

def toon_quote(val, delim=','):
    if isinstance(val, bool): return 'true' if val else 'false'
    if val is None: return 'null'
    if isinstance(val, (int,float)):
        return str(val).rstrip('0').rstrip('.') if isinstance(val,float) else str(val)
    s=str(val)
    needs = (s=='' or s.strip()!=s or s in ['true','false','null'] or s=='-' or s.startswith('-') or any(ch in s for ch in ['"','\\','[',']','{','}',':','\n','\r','\t']) or delim in s)
    # quote things that look numeric
    try:
        float(s); needs=True
    except Exception:
        pass
    if needs:
        s=s.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n').replace('\r','\\r').replace('\t','\\t')
        return f'"{s}"'
    return s

def toon_ser(items):
    rows=flat_records(items)
    fields=list(rows[0].keys())
    lines=[]
    lines.append('meta:')
    lines.append('  model: IRT3PL')
    lines.append('  date: 20260603')
    lines.append(f'  n: {len(items)}')
    lines.append('  lang: es')
    lines.append('  topic: evaluación transversal')
    lines.append('  bloom_distribution[6]: 1,1,1,1,1,1')
    lines.append('  cat:')
    lines.append('    theta_init: 0')
    lines.append('    theta_min: -3')
    lines.append('    theta_max: 3')
    lines.append('    se_stop: 0.3')
    lines.append(f'    max_items: {len(items)}')
    lines.append('    exposure_ctrl: SH')
    lines.append(f'items[{len(items)}]'+'{'+','.join(fields)+'}:')
    for row in rows:
        lines.append('  '+','.join(toon_quote(row[f]) for f in fields))
    return '\n'.join(lines)

def qti_xml_ser(items):
    # Lightweight QTI-like representation; enough to show verbosity, not a full package manifest.
    lines=['<qti-assessment-test xmlns="http://www.imsglobal.org/xsd/imsqtiasi_v3p0" identifier="mini_test" title="Evaluación transversal">']
    for it in items:
        lines.append(f'  <qti-assessment-item identifier="{it["id"]}" title="{html.escape(it["topic"])}" adaptive="false" time-dependent="false">')
        lines.append('    <qti-response-declaration identifier="RESPONSE" cardinality="single" base-type="identifier">')
        lines.append(f'      <qti-correct-response><qti-value>O{it["correct"]+1}</qti-value></qti-correct-response>')
        lines.append('    </qti-response-declaration>')
        lines.append('    <qti-item-body>')
        lines.append(f'      <p>{html.escape(it["statement"])}</p>')
        lines.append('      <qti-choice-interaction response-identifier="RESPONSE" max-choices="1">')
        for j,opt in enumerate(it['options']):
            lines.append(f'        <qti-simple-choice identifier="O{j+1}">{html.escape(opt)}</qti-simple-choice>')
        lines.append('      </qti-choice-interaction>')
        lines.append('    </qti-item-body>')
        lines.append(f'    <qti-metadata><qti-metadatafield><qti-fieldlabel>bloom</qti-fieldlabel><qti-fieldentry>{it["bloom"]}</qti-fieldentry></qti-metadatafield></qti-metadata>')
        lines.append('  </qti-assessment-item>')
    lines.append('</qti-assessment-test>')
    return '\n'.join(lines)

FORMATTERS = {'.mini': mini, '.toon': toon_ser, 'JSON-compact': json_compact, 'JSON-pretty': json_pretty, 'YAML': yaml_ser, 'XML': xml_ser, 'QTI-XML-lite': qti_xml_ser}

if __name__=='__main__':
    encs={'cl100k_base': tiktoken.get_encoding('cl100k_base'), 'o200k_base': tiktoken.get_encoding('o200k_base')}
    sizes=[5,10,12,25,50,100]
    for enc_name, enc in encs.items():
        print('\nENC', enc_name)
        print('size,'+','.join(FORMATTERS.keys()))
        for n in sizes:
            items=expand_items(n)
            counts=[len(enc.encode(fn(items))) for fn in FORMATTERS.values()]
            print(str(n)+','+','.join(map(str, counts)))
    # output 12 item appendix files (en el directorio actual)
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    items=expand_items(12)
    for name, fn in FORMATTERS.items():
        if name in ['.mini','.toon','JSON-compact','YAML']:
            ext = {'JSON-compact':'json','YAML':'yaml','.mini':'mini','.toon':'toon'}[name]
            with open(os.path.join(out_dir, f'dataset_12.{ext}'), 'w', encoding='utf-8') as f:
                f.write(fn(items))
