"""DATAFEST 2026 — Fase 1: protocolo temporal, CatBoost y logistica L2.

Python >= 3.10. Instalar: python -m pip install numpy pandas scikit-learn catboost
Ejecutar: python baseline_datafest_1.0.py --output resultados_fase1 --threads 4
Comprobar entradas: python baseline_datafest_1.0.py --check-only
Prueba funcional corta: python baseline_datafest_1.0.py --quick --output prueba_fase1
Continuar una ejecucion interrumpida: repetir el mismo comando con --resume.

Para evaluar mes M: entrenamiento interno < M-1; seleccion en M-1;
reentrenamiento < M; evaluacion exclusivamente en M. Se evalua junio-octubre.
Variables history IDENTICAS al baseline, sin dias_ultima_interaccion, sin pesos
de clase, sin ventanas recientes, sin nuevas variables. No se usa noviembre,
test.csv ni sample_submission.csv. No se produce una entrega en esta fase.

La seleccion global de hiperparametros/finalistas usa junio-octubre: sus scores
son de desarrollo, NO una estimacion independiente del ganador seleccionado.
La desviacion mensual NO es un intervalo de confianza (clientes repetidos).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from itertools import product
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


def check_protocol(df, x):
    """Invariancia causal real: quitar el mes evaluado y los futuros."""
    for month in MONTHS:
        inner, stop, refit, outer = fold_indices(df, month)
        if not (df.iloc[inner].mes.max() < df.iloc[stop].mes.min()
                and df.iloc[refit].mes.max() < df.iloc[outer].mes.min()):
            raise AssertionError('Particiones fuera de orden.')
        prefix = df.iloc[refit].drop(columns='objetivo').reset_index(drop=True)
        pd.testing.assert_frame_equal(x.iloc[refit].reset_index(drop=True), history_features(prefix))
    if set(['id_cliente', 'mes', 'objetivo', 'dias_ultima_interaccion']) & set(x.columns):
        raise AssertionError('Variable excluida presente en X.')


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


def cat_model(config, seed, iterations, args):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(depth=config['depth'], l2_leaf_reg=config['l2'],
                              iterations=iterations, learning_rate=0.03,
                              loss_function='Logloss', eval_metric='AUC',
                              random_seed=seed, thread_count=args.threads,
                              allow_writing_files=False, verbose=False)


def run_cat(df, x, config, seed, month, args):
    inner, stop, refit, outer = fold_indices(df, month)
    y = df.objetivo
    tuner = cat_model(config, seed, args.iterations, args)
    tuner.fit(x.iloc[inner], y.iloc[inner], cat_features=CATS,
              eval_set=(x.iloc[stop], y.iloc[stop]),
              early_stopping_rounds=args.patience, use_best_model=True)
    count = int(tuner.tree_count_)
    # Curvas internas: no se monitoriza ni se usa el mes externo al ajustar.
    curves = tuner.get_evals_result()
    calibration_auc = auc(y.iloc[stop], tuner.predict_proba(x.iloc[stop])[:, 1])
    del tuner
    model = cat_model(config, seed, count, args)
    model.fit(x.iloc[refit], y.iloc[refit], cat_features=CATS)
    p = model.predict_proba(x.iloc[outer])[:, 1]
    train_auc = auc(y.iloc[refit], model.predict_proba(x.iloc[refit])[:, 1])
    meta = dict(iterations=count, calibration_month=month-1, calibration_auc=calibration_auc,
                train_auc=train_auc, iteration_limit_hit=count == args.iterations,
                curves=curves, config=config, seed=seed)
    return prediction_frame(df, outer, p, refit), meta


def logistic_pipeline(x, strength):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    # Control lineal sobre las mismas variables. Preprocesamiento aprendido
    # exclusivamente con el entrenamiento de cada ajuste, nunca con validacion.
    prep = ColumnTransformer([
        ('categorias', OneHotEncoder(handle_unknown='ignore'), CATS),
        ('numericas', StandardScaler(), [c for c in x if c not in CATS]),
    ])
    return Pipeline([('preproceso', prep), ('modelo', LogisticRegression(
        C=strength, solver='lbfgs', max_iter=3000, tol=1e-6, class_weight=None))])


def fit_logistic(model, x, y, threads):
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    with warnings.catch_warnings(), threadpool_limits(limits=threads):
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            return model.fit(x, y)
        except ConvergenceWarning as exc:
            raise RuntimeError('La logistica no convergio; no se publican metricas de ese ajuste.') from exc


def run_logistic(df, x, month, args):
    inner, stop, refit, outer = fold_indices(df, month)
    candidates = []
    for strength in ([1.0] if args.quick else [0.01, 0.1, 1.0, 10.0]):
        model = fit_logistic(logistic_pipeline(x, strength), x.iloc[inner], df.objetivo.iloc[inner], args.threads)
        score = auc(df.objetivo.iloc[stop], model.predict_proba(x.iloc[stop])[:, 1])
        candidates.append(dict(C=strength, auc=score))
    best = sorted(candidates, key=lambda c: (-c['auc'], c['C']))[0]
    model = fit_logistic(logistic_pipeline(x, best['C']), x.iloc[refit], df.objetivo.iloc[refit], args.threads)
    p = model.predict_proba(x.iloc[outer])[:, 1]
    meta = dict(C=best['C'], calibration_month=month-1, calibration_auc=best['auc'],
                calibration_candidates=candidates,
                iterations=int(model.named_steps['modelo'].n_iter_[0]),
                train_auc=auc(df.objetivo.iloc[refit], model.predict_proba(x.iloc[refit])[:, 1]))
    return prediction_frame(df, outer, p, refit), meta


def run_cached(df, x, model_id, seed, month, config, args):
    folder = args.output / 'ajustes' / model_id / str(seed) / str(month)
    folder.mkdir(parents=True, exist_ok=True)
    pred_path, meta_path = folder / 'predicciones.csv', folder / 'ajuste.json'
    if args.resume and pred_path.exists() and meta_path.exists():
        pred = pd.read_csv(pred_path, dtype={'id_cliente': df.id_cliente.dtype})
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        _, _, refit, outer = fold_indices(df, month)
        expected = prediction_frame(df, outer, pred.prediccion.to_numpy(), refit)
        pd.testing.assert_frame_equal(pred.drop(columns='prediccion'), expected.drop(columns='prediccion'))
        print(f'Reutilizado: {model_id}, semilla {seed}, mes {month}', flush=True)
        return pred, meta
    started = time.perf_counter()
    print(f'Ajustando: {model_id}, semilla {seed}, mes {month}', flush=True)
    pred, meta = (run_logistic(df, x, month, args) if config is None
                  else run_cat(df, x, config, seed, month, args))
    meta['seconds'] = time.perf_counter() - started
    tmp = pred_path.with_suffix('.csv.tmp')
    pred.to_csv(tmp, index=False)
    tmp.replace(pred_path)
    json_write(meta_path, meta)
    print(f'  AUC={auc(pred.objetivo, pred.prediccion):.6f}; iteraciones={meta["iterations"]}; '
          f'{meta["seconds"]:.1f}s', flush=True)
    return pred, meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=ROOT / 'resultados_fase1')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--iterations', type=int, default=2500, help='Maximo de iteraciones internas de CatBoost')
    p.add_argument('--patience', type=int, default=150)
    p.add_argument('--top-k', type=int, default=3, help='Finalistas que se promedian sobre tres semillas')
    p.add_argument('--seeds', type=int, nargs=3, default=[2026, 2027, 2028])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Validar datos y causalidad; no entrenar')
    p.add_argument('--quick', action='store_true', help='Prueba funcional: rejilla reducida, 25 iteraciones; no comparar scores')
    args = p.parse_args()
    if min(args.threads, args.iterations, args.patience, args.top_k) < 1:
        p.error('threads, iterations, patience y top-k deben ser positivos.')
    if len(set(args.seeds)) != 3 or any(s < 0 or s >= 2**31 for s in args.seeds):
        p.error('Se requieren tres semillas distintas entre 0 y 2**31-1.')
    if args.quick:
        args.iterations, args.patience, args.top_k = 25, 10, 1
    args.output = args.output.resolve()
    return args


def main():
    args = parse_args()
    df = load_data(args.data_dir / 'train.csv')
    x = history_features(df.drop(columns='objetivo'))
    check_protocol(df, x)
    print(f'Datos verificados: {len(df):,} filas enero-octubre, {x.shape[1]} variables history.', flush=True)
    if args.check_only:
        print('OK: particiones temporales e invariancia de caracteristicas al retirar meses futuros.')
        return
    try:
        versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'catboost']}
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit('Faltan dependencias. Ejecutar: python -m pip install numpy pandas scikit-learn catboost') from exc
    configs = {f'cat_d{d}_l2_{l2}': dict(depth=d, l2=l2)
               for d, l2 in ([(3, 10)] if args.quick else product([3, 4, 5, 6], [3, 10, 30, 100]))}
    configs['cat_control_d5_l2_8'] = dict(depth=5, l2=8)
    signature = dict(protocol='fase1_v1', python=platform.python_version(), packages=versions,
                     script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     data_jan_oct_sha256=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
                     months=list(MONTHS), features=x.columns.tolist(), feature_mode='history',
                     categorical_columns=CATS, class_weights=None, configs=configs,
                     threads=args.threads, iterations=args.iterations, patience=args.patience,
                     top_k=args.top_k, seeds=args.seeds, quick=args.quick,
                     logistic_C=[1.0] if args.quick else [0.01, 0.1, 1.0, 10.0])
    manifest = args.output / 'configuracion.json'
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume or not manifest.exists():
            raise SystemExit('La carpeta de salida no esta vacia. Usar otra carpeta o --resume para continuar.')
        if json.loads(manifest.read_text(encoding='utf-8')) != signature:
            raise SystemExit('No se puede reanudar: cambiaron el codigo, datos, parametros o versiones. Usar otra carpeta.')
    args.output.mkdir(parents=True, exist_ok=True)
    json_write(manifest, signature)
    protocol = []
    for month in MONTHS:
        inner, stop, refit, outer = fold_indices(df, month)
        unseen = ~df.iloc[outer].id_cliente.isin(df.iloc[refit].id_cliente)
        protocol.append(dict(mes_evaluacion=month, mes_seleccion=month-1,
                             ultimo_mes_entrenamiento_interno=int(df.iloc[inner].mes.max()),
                             n_interno=len(inner), n_seleccion=len(stop), n_reentrenamiento=len(refit),
                             n_evaluacion=len(outer), n_nuevos=int(unseen.sum()), n_conocidos=int((~unseen).sum())))
    pd.DataFrame(protocol).to_csv(args.output / 'particiones.csv', index=False)
    rows, predictions = [], []
    cached_first_seed = {}
    for model_id, config in configs.items():
        for month in MONTHS:
            pred, meta = run_cached(df, x, model_id, args.seeds[0], month, config, args)
            cached_first_seed[(model_id, month)] = (pred, meta)
            rows.extend(segment_metrics(pred, model_id, 'catboost', args.seeds[0], meta))
            predictions.append(pred.assign(modelo=model_id, familia='catboost', semilla=str(args.seeds[0])))
        pd.DataFrame(rows).to_csv(args.output / 'metricas_mensuales.csv', index=False)
    search = summarize(rows)
    ranking = search.loc[search.segmento.eq('todos')].sort_values(
        ['auc_media', 'auc_std_mensual', 'modelo'], ascending=[False, True, True])
    ranking.to_csv(args.output / 'ranking_busqueda.csv', index=False)
    finalists = ranking.modelo.head(args.top_k).tolist()
    json_write(args.output / 'finalistas.json', dict(modelos=finalists, semillas=args.seeds,
        criterio='AUC mensual medio de junio-octubre con primera semilla; desempate por menor desviacion y nombre.',
        advertencia='Seleccion y comparacion de desarrollo; no es evaluacion independiente del ganador.'))
    print(f'Finalistas para tres semillas: {finalists}', flush=True)
    for model_id in finalists:
        for month in MONTHS:
            first, first_meta = cached_first_seed[(model_id, month)]
            parts = [first.prediccion.to_numpy()]
            rounds = [first_meta['iterations']]
            for seed in args.seeds[1:]:
                pred, meta = run_cached(df, x, model_id, seed, month, configs[model_id], args)
                pd.testing.assert_frame_equal(first.drop(columns='prediccion'), pred.drop(columns='prediccion'))
                parts.append(pred.prediccion.to_numpy())
                rounds.append(meta['iterations'])
                rows.extend(segment_metrics(pred, model_id, 'catboost', seed, meta))
                predictions.append(pred.assign(modelo=model_id, familia='catboost', semilla=str(seed)))
            ensemble = first.copy()
            ensemble['prediccion'] = np.mean(parts, axis=0)
            ensemble_id = model_id + '_3semillas'
            rows.extend(segment_metrics(ensemble, ensemble_id, 'catboost_ensemble', 'promedio', {}))
            predictions.append(ensemble.assign(modelo=ensemble_id, familia='catboost_ensemble', semilla='promedio'))
            print(f'Ensemble {model_id} {month}: AUC={auc(ensemble.objetivo, ensemble.prediccion):.6f}; arboles={rounds}', flush=True)
    for month in MONTHS:
        pred, meta = run_cached(df, x, 'logistica_l2', 'determinista', month, None, args)
        rows.extend(segment_metrics(pred, 'logistica_l2', 'logistica', 'determinista', meta))
        predictions.append(pred.assign(modelo='logistica_l2', familia='logistica', semilla='determinista'))
    pd.DataFrame(rows).to_csv(args.output / 'metricas_mensuales.csv', index=False)
    summary = summarize(rows)
    summary.to_csv(args.output / 'resumen_modelos.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output / 'predicciones_temporales.csv', index=False)
    json_write(args.output / 'metricas.json', dict(
        status='completo', quick=args.quick, months=list(MONTHS), finalistas=finalists,
        resumen=json.loads(summary.to_json(orient='records', double_precision=15)),
        notas=['AUC medio no ponderado de los cinco meses; no AUC agrupado.',
               'auc_std_mensual es desviacion muestral (ddof=1), no error estandar ni intervalo de confianza.',
               'AUC de segmento sin ambas clases se guarda nulo y no se promedia.',
               'Seleccion de iteraciones/C en el mes anterior y refit antes del mes evaluado.',
               'El ranking y la eleccion de finalistas usan estos meses: scores de desarrollo.',
               'Noviembre excluido antes de construir X; no se lee test ni se genera submission.',
               'Quick es solo una prueba de funcionamiento, sin valor para seleccionar modelos.' if args.quick else 'Ejecucion completa de Fase 1.']))
    print('\nResumen global (scores de desarrollo):', flush=True)
    print(summary.loc[summary.segmento.eq('todos')].to_string(index=False), flush=True)
    print(f'\nResultados completos en: {args.output}', flush=True)


if __name__ == '__main__':
    main()
