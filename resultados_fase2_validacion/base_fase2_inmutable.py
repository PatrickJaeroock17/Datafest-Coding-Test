"""DATAFEST Fase 2: pesos de clase e historial temporal.
Standalone: Python >=3.10; numpy, pandas, scikit-learn, catboost.
python baseline_datafest_2.0.py --output resultados_fase2 --threads 4
Exploracion: exclusivamente d3/L2=10/semilla 2026, 4 pesos x 3 politicas.
Integracion automatica: d3/L2=10, d6/L2=10 y d4/L2=3, tres semillas cada uno.
Junio-octubre son desarrollo; noviembre y test no intervienen.
No se generan predicciones de diciembre en esta fase.

Politicas:
- completo: history original y todas las filas anteriores.
- ultimos6: history usa solo [mes-6, mes); conserva todas las filas de
  entrenamiento para aislar la construccion de variables.
- decay: history original y todas las filas; sample_weight exponencial con
  semivida --half-life (3 meses por defecto), referido al ultimo mes entrenado.

Peso de fila = peso temporal * multiplicador de clase, normalizado a media 1.
balanced significa N_negativos/N_positivos SIN ponderar, calculado sobre
cada ajuste (interno y refit por separado), no sobre el mes de evaluacion.
La normalizacion mantiene la escala media al comparar regularizacion.
No se usan class_weights ni auto_class_weights ademas de sample_weight.
Early stopping y evaluacion usan AUC sin ponderar.
--quick conserva los 12 experimentos y los finalistas, pero limita iteraciones;
sus resultados solo comprueban el funcionamiento.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

MONTHS = (202606, 202607, 202608, 202609, 202610)

CATS = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']

FLAGS = ['tiene_tarjeta_credito', 'activo_movil', 'es_nuevo_cliente', 'tiene_prestamo', 'tiene_seguro']

NUMS = ['edad', 'ingresos', 'ratio_deuda_ingresos', 'antiguedad_cuenta_meses',
        'numero_productos', 'saldo_promedio', 'dias_ultima_transaccion',
        'antiguedad_direccion_meses', 'visitas_web_ultimos_90_dias',
        'distancia_sucursal_km', 'dia_preferido_pago', 'dias_ultima_interaccion']

def history_features(frame):
    """Copia de las variables history originales; nunca recibe objetivo.

    Un -1 distingue la ausencia de una observacion anterior. La primera
    observacion es la primera disponible, no la fecha real de alta bancaria.
    """
    z = frame.drop(columns='objetivo', errors='ignore').copy()
    z['_order'] = np.arange(len(z))
    z = z.sort_values(['id_cliente', 'mes'])
    z['saldo_ingreso_mensual'] = z.saldo_promedio / (z.ingresos / 12).replace(0, np.nan)
    z['saldo_por_producto'] = z.saldo_promedio / z.numero_productos.replace(0, np.nan)
    z['visitas_por_producto'] = z.visitas_web_ultimos_90_dias / z.numero_productos.replace(0, np.nan)
    z['productos_flags'] = z[['tiene_tarjeta_credito', 'tiene_prestamo', 'tiene_seguro']].astype(int).sum(axis=1)
    z['movil_x_visitas'] = z.activo_movil.astype(int) * z.visitas_web_ultimos_90_dias
    for c in ['ingresos', 'saldo_promedio', 'dias_ultima_transaccion']:
        z['log_' + c] = np.log1p(z[c].clip(lower=0))
    z['_month_index'] = (z.mes // 100) * 12 + z.mes % 100
    g = z.groupby('id_cliente', sort=False)
    z['observaciones_previas'] = g.cumcount()
    z['meses_desde_observacion_previa'] = z._month_index - g._month_index.shift(1)
    z['meses_desde_primera_observacion'] = z._month_index - g._month_index.transform('first')
    z = z.sort_values('_order').drop(columns=['id_cliente', 'mes', '_order', '_month_index', 'dias_ultima_interaccion'])
    for c in CATS:
        z[c] = z[c].fillna('__MISSING__').astype(str)
    for c in FLAGS:
        z[c] = z[c].astype('int8')
    numeric = [c for c in z if c not in CATS]
    z[numeric] = z[numeric].replace([np.inf, -np.inf], np.nan).fillna(-1)
    return z.reset_index(drop=True)

def load_data(path):
    # Eliminar noviembre ANTES de validar etiquetas o construir caracteristicas.
    df = pd.read_csv(path)
    required = set(['id_cliente', 'mes', 'objetivo'] + CATS + FLAGS + NUMS)
    if set(df.columns) != required:
        raise ValueError(f'Esquema inesperado. Faltan {required-set(df.columns)}; sobran {set(df.columns)-required}.')
    df = df.loc[df.mes.between(202601, 202610)].sort_values(['mes', 'id_cliente']).reset_index(drop=True)
    if set(df.mes.unique()) != set(range(202601, 202611)):
        raise ValueError('Se requieren todos los meses de enero a octubre de 2026.')
    if df.isna().any().any() or df.duplicated(['id_cliente', 'mes']).any():
        raise ValueError('Hay faltantes o claves cliente-mes duplicadas.')
    if not df.objetivo.isin([0, 1]).all():
        raise ValueError('objetivo debe ser binario 0/1.')
    for c in FLAGS:
        if not df[c].isin([True, False, 0, 1]).all():
            raise ValueError(f'{c} debe contener booleanos o 0/1.')
    if not np.isfinite(df[NUMS].to_numpy(dtype=float)).all():
        raise ValueError('Hay numericos no finitos.')
    if (df.groupby('mes').objetivo.nunique() != 2).any():
        raise ValueError('Cada mes debe contener ambas clases.')
    ordered = df.sort_values(['id_cliente', 'mes'])
    prior_positive = ordered.groupby('id_cliente').objetivo.cumsum() - ordered.objetivo
    if prior_positive.gt(0).any():
        raise ValueError('Hay observaciones posteriores a la primera conversion.')
    return df

def fold_indices(df, month):
    previous = month - 1  # Solo junio-octubre de 2026; no cruza cambio de anio.
    inner = np.flatnonzero(df.mes.lt(previous))
    stop = np.flatnonzero(df.mes.eq(previous))
    refit = np.flatnonzero(df.mes.lt(month))
    outer = np.flatnonzero(df.mes.eq(month))
    if any(len(a) == 0 for a in [inner, stop, refit, outer]):
        raise ValueError(f'Particion vacia para {month}.')
    return inner, stop, refit, outer

def json_write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None

def prediction_frame(df, outer, p, refit):
    p = np.asarray(p)
    if p.shape != (len(outer),) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Predicciones invalidas.')
    result = df.iloc[outer][['id_cliente', 'mes', 'objetivo']].copy().reset_index(drop=True)
    result['segmento'] = np.where(result.id_cliente.isin(df.iloc[refit].id_cliente), 'conocido', 'nuevo')
    result['prediccion'] = p
    return result

def segment_metrics(pred, model_id, family, seed, metadata):
    from sklearn.metrics import log_loss
    rows = []
    for segment in ['todos', 'nuevo', 'conocido']:
        part = pred if segment == 'todos' else pred.loc[pred.segmento.eq(segment)]
        score = auc(part.objetivo, part.prediccion)
        rows.append(dict(modelo=model_id, familia=family, semilla=str(seed), mes=int(pred.mes.iloc[0]),
                         segmento=segment, n=len(part), positivos=int(part.objetivo.sum()),
                         tasa_positivos=float(part.objetivo.mean()) if len(part) else None,
                         auc=score, gini=2*score-1 if score is not None else None,
                         logloss=float(log_loss(part.objetivo, part.prediccion, labels=[0, 1])) if len(part) else None,
                         iteraciones=metadata.get('iterations'), C=metadata.get('C'),
                         auc_entrenamiento=metadata.get('train_auc')))
    return rows

def summarize(rows):
    frame = pd.DataFrame(rows)
    return (frame.groupby(['modelo', 'familia', 'semilla', 'segmento'], dropna=False)
            .agg(auc_media=('auc', 'mean'), auc_std_mensual=('auc', 'std'),
                 auc_min=('auc', 'min'), auc_max=('auc', 'max'),
                 gini_media=('gini', 'mean'), meses_validos=('auc', 'count'),
                 observaciones=('n', 'sum')).reset_index()
            .sort_values(['segmento', 'auc_media', 'modelo'], ascending=[True, False, True]))


FINALISTS = {'cat_d3_l2_10': {'depth': 3, 'l2': 10},
             'cat_d6_l2_10': {'depth': 6, 'l2': 10},
             'cat_d4_l2_3': {'depth': 4, 'l2': 3}}
SEEDS = (2026, 2027, 2028)
POLICIES = ('completo', 'ultimos6', 'decay')
CLASS_MODES = ('1', '2', '3', 'balanced')


def month_index(months):
    values = np.asarray(months, dtype=np.int64)
    return (values // 100) * 12 + values % 100


def build_features(frame, policy):
    if policy not in POLICIES:
        raise ValueError(f'Politica desconocida: {policy}')
    x = history_features(frame)
    if policy != 'ultimos6':
        return x
    # Mismo conjunto de columnas; solo cambia el horizonte de las tres
    # variables historicas. Una observacion de M nunca cuenta como pasado.
    ordered = frame[['id_cliente', 'mes']].copy()
    ordered['_row'] = np.arange(len(frame))
    ordered = ordered.sort_values(['id_cliente', 'mes']).reset_index(drop=True)
    counts = np.zeros(len(frame), dtype=np.int64)
    previous = np.full(len(frame), -1.0)
    first = np.zeros(len(frame), dtype=np.int64)
    times = month_index(ordered.mes)
    rows = ordered._row.to_numpy()
    for idx in ordered.groupby('id_cliente', sort=False).indices.values():
        t = times[idx]
        left = np.searchsorted(t, t - 6, side='left')
        j = np.arange(len(t))
        count = j - left
        dest = rows[idx]
        counts[dest] = count
        first[dest] = t - t[left]
        previous[dest] = np.where(count > 0, t - t[np.maximum(j-1, 0)], -1.0)
    x['observaciones_previas'] = counts
    x['meses_desde_observacion_previa'] = previous
    x['meses_desde_primera_observacion'] = first
    return x


def training_weights(y, months, class_mode, policy, half_life):
    """Solo etiquetas y meses del ajuste actual; nunca recibe validacion."""
    y = np.asarray(y, dtype=int)
    if class_mode not in CLASS_MODES or policy not in POLICIES:
        raise ValueError('Peso de clase o politica invalida.')
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError('half-life debe ser finito y positivo.')
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg or n_pos+n_neg != len(y):
        raise ValueError('El entrenamiento debe tener exactamente las clases 0 y 1.')
    factor = n_neg / n_pos if class_mode == 'balanced' else float(class_mode)
    t = month_index(months)
    if len(t) != len(y):
        raise ValueError('Etiquetas y meses no estan alineados.')
    age = t.max() - t
    temporal = np.exp2(-age / half_life) if policy == 'decay' else np.ones(len(y))
    raw = temporal * np.where(y == 1, factor, 1.0)
    normalizer = float(raw.mean())
    weights = raw / normalizer
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Pesos no finitos o no positivos.')
    meta = dict(class_mode=class_mode, positive_factor=float(factor), n_positive=n_pos,
                n_negative=n_neg, anchor_month=int(np.max(months)), half_life=half_life if policy=='decay' else None,
                temporal_min=float(temporal.min()), temporal_max=float(temporal.max()),
                normalizer=normalizer, weight_mean=float(weights.mean()),
                effective_sample_size=float(weights.sum()**2 / np.square(weights).sum()),
                positive_weight_sum=float(weights[y==1].sum()), negative_weight_sum=float(weights[y==0].sum()))
    # Omitir unos mantiene la llamada del control equivalente a Fase 1.
    return (None if policy != 'decay' and class_mode == '1' else weights), meta


def check_protocol(df, feature_sets):
    for month in MONTHS:
        inner, stop, refit, outer = fold_indices(df, month)
        if not (df.iloc[inner].mes.max() < df.iloc[stop].mes.min()
                and df.iloc[refit].mes.max() < df.iloc[outer].mes.min()):
            raise AssertionError('Particiones no causales.')
        # Se comprueban tambien las features del mes externo sin meses futuros.
        prefix_idx = np.flatnonzero(df.mes.le(month))
        prefix = df.iloc[prefix_idx].drop(columns='objetivo').reset_index(drop=True)
        for policy in ['completo', 'ultimos6']:
            pd.testing.assert_frame_equal(feature_sets[policy].iloc[prefix_idx].reset_index(drop=True),
                                          build_features(prefix, policy))
    for x in feature_sets.values():
        if {'id_cliente', 'mes', 'objetivo', 'dias_ultima_interaccion'} & set(x.columns):
            raise AssertionError('Hay una variable excluida en X.')


def cat_model(config, seed, rounds, args):
    from catboost import CatBoostClassifier
    # AUC no pondera por defecto. El eval_set tampoco recibe pesos.
    # No especificar use_weights=false: CatBoost lo rechaza en el control
    # cuando el entrenamiento no contiene pesos explicitos.
    return CatBoostClassifier(depth=config['depth'], l2_leaf_reg=config['l2'],
        iterations=rounds, learning_rate=0.03, loss_function='Logloss',
        eval_metric='AUC', random_seed=seed,
        thread_count=args.threads, allow_writing_files=False, verbose=False)


def run_fold(df, x, config, combo, seed, month, args):
    inner, stop, refit, outer = fold_indices(df, month)
    y = df.objetivo
    wi, info_i = training_weights(y.iloc[inner], df.mes.iloc[inner], combo['class_mode'], combo['policy'], args.half_life)
    wr, info_r = training_weights(y.iloc[refit], df.mes.iloc[refit], combo['class_mode'], combo['policy'], args.half_life)
    tuner = cat_model(config, seed, args.iterations, args)
    tuner.fit(x.iloc[inner], y.iloc[inner], cat_features=CATS, sample_weight=wi,
              eval_set=(x.iloc[stop], y.iloc[stop]),
              early_stopping_rounds=args.patience, use_best_model=True)
    rounds = int(tuner.tree_count_)
    curves = tuner.get_evals_result()
    evaluated = max(len(values) for values in curves['validation'].values())
    calibration_auc = auc(y.iloc[stop], tuner.predict_proba(x.iloc[stop])[:, 1])
    raw_auc = auc(y.iloc[stop], tuner.predict(x.iloc[stop], prediction_type='RawFormulaVal'))
    # CatBoost evalua scores raw. La conversion a probabilidades puede crear
    # empates por redondeo; guardar ambas metricas sin exigir igualdad exacta.
    internal_auc = float(curves['validation']['AUC'][rounds-1])
    del tuner
    model = cat_model(config, seed, rounds, args)
    model.fit(x.iloc[refit], y.iloc[refit], cat_features=CATS, sample_weight=wr)
    p = model.predict_proba(x.iloc[outer])[:, 1]
    meta = dict(iterations=rounds, evaluated_iterations=evaluated,
        iteration_limit_hit=evaluated >= args.iterations, config=config, combo=combo, seed=seed,
        calibration_month=month-1, calibration_auc_unweighted=calibration_auc,
        calibration_auc_raw_unweighted=raw_auc, calibration_auc_catboost=internal_auc,
        train_auc=auc(y.iloc[refit], model.predict_proba(x.iloc[refit])[:, 1]),
        weights_inner=info_i, weights_refit=info_r, curves=curves)
    # Nuevo/conocido se determina con todo el pasado disponible, aunque
    # ultimos6 no encuentre una observacion reciente de un cliente conocido.
    return prediction_frame(df, outer, p, refit), meta


def run_cached(df, x, config, combo, seed, month, args):
    folder = args.output / 'ajustes' / combo['id'] / config['id'] / str(seed) / str(month)
    folder.mkdir(parents=True, exist_ok=True)
    pred_path, meta_path = folder/'predicciones.csv', folder/'ajuste.json'
    # El manifiesto ya valido --resume al iniciar. Permite tambien reutilizar
    # d3/2026 de la exploracion al integrar, dentro de la misma ejecucion.
    if pred_path.exists() and meta_path.exists():
        pred = pd.read_csv(pred_path, dtype={'id_cliente': df.id_cliente.dtype}, float_precision='round_trip')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['config'] != config or meta['combo'] != combo or meta['seed'] != seed:
            raise ValueError('Checkpoint incompatible.')
        _, _, refit, outer = fold_indices(df, month)
        expected = prediction_frame(df, outer, pred.prediccion.to_numpy(), refit)
        pd.testing.assert_frame_equal(pred.drop(columns='prediccion'), expected.drop(columns='prediccion'))
        print(f'Reutilizado: {combo["id"]} {config["id"]} seed={seed} mes={month}', flush=True)
        return pred, meta
    start = time.perf_counter()
    print(f'Ajustando: {combo["id"]} {config["id"]} seed={seed} mes={month}', flush=True)
    pred, meta = run_fold(df, x, config, combo, seed, month, args)
    meta['seconds'] = time.perf_counter()-start
    tmp = pred_path.with_suffix('.csv.tmp')
    pred.to_csv(tmp, index=False)
    tmp.replace(pred_path)
    json_write(meta_path, meta)
    print(f'  AUC={auc(pred.objetivo, pred.prediccion):.6f}, iteraciones={meta["iterations"]}', flush=True)
    return pred, meta


def enrich_metrics(pred, model_id, family, seed, meta, combo, stage):
    rows = segment_metrics(pred, model_id, family, seed, meta)
    for row in rows:
        row.update(combinacion=combo['id'], politica_temporal=combo['policy'],
                   modo_peso_clase=combo['class_mode'], etapa=stage,
                   factor_positivo_interno=meta.get('weights_inner', {}).get('positive_factor'),
                   factor_positivo_refit=meta.get('weights_refit', {}).get('positive_factor'))
        row.pop('C', None)
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=ROOT/'resultados_fase2')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--iterations', type=int, default=2500)
    p.add_argument('--patience', type=int, default=150)
    p.add_argument('--half-life', type=float, default=3.0, help='Semivida del peso temporal, en meses')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--quick', action='store_true', help='12 combinaciones y 3 finalistas, pero solo 12 iteraciones; prueba funcional')
    args = p.parse_args()
    if min(args.threads, args.iterations, args.patience) < 1 or not np.isfinite(args.half_life) or args.half_life <= 0:
        p.error('threads, iterations, patience y half-life deben ser positivos y finitos.')
    if args.quick:
        args.iterations, args.patience = 12, 5
    args.output = args.output.resolve()
    return args


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def main():
    args = parse_args()
    df = load_data(args.data_dir/'train.csv')
    features = {p: build_features(df.drop(columns='objetivo'), p) for p in ['completo', 'ultimos6']}
    features['decay'] = features['completo']
    check_protocol(df, features)
    print(f'Datos verificados: {len(df)} filas; 32 variables; todas las filas conservadas en cada politica.', flush=True)
    if args.check_only:
        return
    try:
        versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'catboost']}
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit('Instalar: python -m pip install numpy pandas scikit-learn catboost') from exc
    combos = [dict(id=f'{policy}_clase_{c}', policy=policy, class_mode=c)
              for policy in POLICIES for c in CLASS_MODES]
    manifest_data = dict(protocol='fase2_v1', python=platform.python_version(), packages=versions,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_jan_oct_sha256=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
        months=list(MONTHS), combos=combos, finalists=FINALISTS, exploration_seed=2026,
        seeds=list(SEEDS), features=features['completo'].columns.tolist(),
        history_window_scope='features_only', history_window_months=6,
        threads=args.threads, iterations=args.iterations, patience=args.patience,
        half_life=args.half_life, quick=args.quick,
        weighting='sample_weight=temporal*class_multiplier, normalized to mean 1',
        balanced_definition='unweighted training negatives / training positives; recalculated per fit')
    manifest = args.output/'configuracion.json'
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume or not manifest.exists():
            raise SystemExit('Carpeta no vacia: usar otra carpeta o --resume.')
        if json.loads(manifest.read_text(encoding='utf-8')) != manifest_data:
            raise SystemExit('Codigo, parametros, versiones o datos cambiaron; usar otra carpeta.')
    args.output.mkdir(parents=True, exist_ok=True)
    json_write(manifest, manifest_data)
    partitions = []
    for month in MONTHS:
        inner, stop, refit, outer = fold_indices(df, month)
        partitions.append(dict(mes_evaluacion=month, mes_seleccion=month-1,
            n_interno=len(inner), n_seleccion=len(stop), n_refit=len(refit), n_evaluacion=len(outer),
            n_nuevos=int((~df.iloc[outer].id_cliente.isin(df.iloc[refit].id_cliente)).sum())))
    pd.DataFrame(partitions).to_csv(args.output/'particiones.csv', index=False)
    search_rows = []
    scout = dict(id='cat_d3_l2_10', **FINALISTS['cat_d3_l2_10'])
    for combo in combos:
        for month in MONTHS:
            pred, meta = run_cached(df, features[combo['policy']], scout, combo, 2026, month, args)
            search_rows.extend(enrich_metrics(pred, combo['id'], 'exploracion_d3', 2026, meta, combo, 'exploracion'))
        pd.DataFrame(search_rows).to_csv(args.output/'metricas_experimentos.csv', index=False)
    search_summary = summarize(search_rows)
    search_summary['gini_std_mensual'] = 2*search_summary.auc_std_mensual
    lookup = pd.DataFrame(combos).rename(columns={'id': 'modelo'})
    search_summary = search_summary.merge(lookup, on='modelo', validate='many_to_one')
    search_summary.to_csv(args.output/'resumen_experimentos.csv', index=False)
    ranking = search_summary.loc[search_summary.segmento.eq('todos')].copy()
    if not ranking.meses_validos.eq(5).all() or ranking.auc_media.isna().any():
        raise ValueError('No se puede elegir ganador sin los cinco AUC mensuales.')
    ranking['orden_control'] = ranking.modelo.map({c['id']: i for i,c in enumerate(combos)})
    ranking = ranking.sort_values(['auc_media', 'auc_std_mensual', 'orden_control'], ascending=[False, True, True])
    control_auc = float(ranking.loc[ranking.modelo.eq('completo_clase_1'), 'auc_media'].iloc[0])
    ranking['delta_auc_vs_control'] = ranking.auc_media-control_auc
    ranking.to_csv(args.output/'ranking_experimentos.csv', index=False)
    winner = next(c for c in combos if c['id'] == ranking.iloc[0].modelo)
    winner_report = dict(combinacion=winner, half_life=args.half_life if winner['policy']=='decay' else None,
        auc_media=float(ranking.iloc[0].auc_media), gini_media=float(ranking.iloc[0].gini_media),
        auc_std_mensual=float(ranking.iloc[0].auc_std_mensual),
        delta_auc_vs_control=float(ranking.iloc[0].delta_auc_vs_control),
        criterio='Mayor AUC no ponderado medio junio-octubre sobre d3/L2=10/2026; desempate por desviacion y orden del control.',
        advertencia='Ganador de desarrollo, no evidencia independiente de mejora. Peso 1 puede ganar.')
    json_write(args.output/'combinacion_ganadora.json', winner_report)
    print(f'Combinacion seleccionada: {winner["id"]}', flush=True)
    final_rows, final_predictions = [], []
    ensembles = {}
    for model_id, params in FINALISTS.items():
        config = dict(id=model_id, **params)
        for month in MONTHS:
            parts = []
            template = None
            for seed in SEEDS:
                pred, meta = run_cached(df, features[winner['policy']], config, winner, seed, month, args)
                if template is None:
                    template = pred
                else:
                    pd.testing.assert_frame_equal(template.drop(columns='prediccion'), pred.drop(columns='prediccion'))
                parts.append(pred.prediccion.to_numpy())
                final_rows.extend(enrich_metrics(pred, model_id, 'catboost', seed, meta, winner, 'integracion'))
                final_predictions.append(pred.assign(modelo=model_id, semilla=str(seed), combinacion=winner['id']))
            ens = template.copy()
            ens['prediccion'] = np.mean(parts, axis=0)
            ensembles[(model_id, month)] = ens
            ens_id = model_id+'_3semillas'
            final_rows.extend(enrich_metrics(ens, ens_id, 'ensemble_3semillas', 'promedio', {}, winner, 'integracion'))
            final_predictions.append(ens.assign(modelo=ens_id, semilla='promedio', combinacion=winner['id']))
        pd.DataFrame(final_rows).to_csv(args.output/'metricas_finalistas.csv', index=False)
    # Diagnostico adicional sin nuevos ajustes ni optimizacion de pesos:
    # promedio uniforme de los tres ensembles, equivalente a nueve modelos.
    for month in MONTHS:
        members = [ensembles[(model_id, month)] for model_id in FINALISTS]
        total = members[0].copy()
        for member in members[1:]:
            pd.testing.assert_frame_equal(total.drop(columns='prediccion'), member.drop(columns='prediccion'))
        total['prediccion'] = np.mean([m.prediccion.to_numpy() for m in members], axis=0)
        final_rows.extend(enrich_metrics(total, 'ensemble_uniforme_9modelos', 'ensemble_9modelos', 'promedio', {}, winner, 'integracion'))
        final_predictions.append(total.assign(modelo='ensemble_uniforme_9modelos', semilla='promedio', combinacion=winner['id']))
    final_summary = summarize(final_rows)
    final_summary['gini_std_mensual'] = 2*final_summary.auc_std_mensual
    final_summary['combinacion'] = winner['id']
    pd.DataFrame(final_rows).to_csv(args.output/'metricas_finalistas.csv', index=False)
    final_summary.to_csv(args.output/'resumen_finalistas.csv', index=False)
    pd.concat(final_predictions, ignore_index=True).to_csv(args.output/'predicciones_finalistas.csv', index=False)
    json_write(args.output/'metricas.json', dict(status='completo', quick=args.quick,
        combinacion_ganadora=winner_report, ranking_experimentos=records(ranking),
        resumen_finalistas=records(final_summary), notas=[
            'Historial seis meses cambia solo tres variables history; se conservan todas las filas.',
            'Decaimiento aplicado al peso de filas de entrenamiento, no a las variables history.',
            'Pesos de clase y temporales se aplican una sola vez mediante sample_weight normalizado.',
            'AUC de early stopping, ranking y evaluacion siempre sin ponderar.',
            'balanced usa conteos sin ponderar; con decay no implica masas ponderadas exactamente iguales.',
            'Ganador e integracion evaluados sobre los mismos meses de desarrollo; existe sesgo de seleccion.',
            'La configuracion ganadora en d3 puede no ser la optima en d6 o d4.',
            'Sin evaluacion en noviembre, sin test, sin submission.',
            'Quick solo verifica ejecucion; no sirve para elegir modelos.' if args.quick else 'Busqueda completa.']))
    print(final_summary.loc[final_summary.segmento.eq('todos')].to_string(index=False), flush=True)
    print(f'Resultados completos: {args.output}', flush=True)


if __name__ == '__main__':
    main()
