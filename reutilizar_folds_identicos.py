"""Evita repetir ajustes idénticos: verifica features, etiquetas y árboles antes de copiar.
No selecciona modelos, no calcula métricas nuevas y no modifica fuentes históricas.
"""
import importlib.util
import json
import hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fase2',ROOT/'baseline_datafest_2.0.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
out=ROOT/'resultados_fase2_validacion'
df=pd.read_csv(ROOT/'train.csv').sort_values(['mes','id_cliente']).reset_index(drop=True)
# Solo junio/julio: ambos historiales son exactamente iguales para todas las
# filas de entrenamiento y evaluación. Agosto NO se presume equivalente.
for month in [202606,202607]:
    prefix=df.loc[df.mes.le(month)].drop(columns='objetivo')
    pd.testing.assert_frame_equal(b.build_features(prefix,'completo'),b.build_features(prefix,'ultimos6'))
    full=[]; limited=[]
    for seed in [2026,2027,2028]:
        a=ROOT/'resultados_fase1/ajustes/cat_d3_l2_10'/str(seed)/str(month)
        c=ROOT/'resultados_fase2/ajustes/ultimos6_clase_1/cat_d3_l2_10'/str(seed)/str(month)
        ma=json.loads((a/'ajuste.json').read_text()); mc=json.loads((c/'ajuste.json').read_text())
        assert ma['iterations']==mc['iterations']
        pd.testing.assert_frame_equal(pd.read_csv(a/'predicciones.csv',float_precision='round_trip'),pd.read_csv(c/'predicciones.csv',float_precision='round_trip'))
        full.append(ma['iterations']); limited.append(mc['iterations'])
    assert full==limited
    for seed in [2026,2027,2028]:
        source=out/'ajustes/completo/mediana'/str(seed)/str(month)
        dest=out/'ajustes/ultimos6/mediana'/str(seed)/str(month)
        assert (source/'ajuste.json').exists()
        if dest.exists(): raise ValueError('No sobrescribir checkpoints existentes.')
        dest.mkdir(parents=True)
        content=(source/'predicciones.csv').read_bytes(); (dest/'predicciones.csv').write_bytes(content)
        meta=json.loads((source/'ajuste.json').read_text())
        meta.update(history='ultimos6',reused_equal_features_from=str(source.relative_to(ROOT)),feature_equivalence='Exact equality of all 32 features for ALL refit and evaluation rows; same original rounds and labels; no approximation',source_prediction_sha256=hashlib.sha256(content).hexdigest())
        (dest/'ajuste.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('Reutilizados seis checkpoints equivalentes; cuatro refits nuevos evitados.')
