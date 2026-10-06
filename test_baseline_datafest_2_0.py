"""Pruebas de causalidad/pesos de Fase 2. Ejecutar con python este archivo.

Solo requieren numpy y pandas. La prueba de entrenamiento se ejecuta aparte
con --quick, pues requiere CatBoost y scikit-learn.
"""
import importlib.util
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fase2', ROOT/'baseline_datafest_2.0.py')
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = phase.load_data(ROOT/'train.csv')

    def test_history_window_calendar_and_gaps(self):
        # Julio incluye enero (M-6); agosto lo excluye. Solo hay tres
        # observaciones previas en agosto: no confundir filas con meses.
        base = self.df.iloc[[0]].drop(columns='objetivo')
        fixture = pd.concat([base]*5, ignore_index=True)
        fixture['mes'] = [202601, 202603, 202606, 202607, 202608]
        x = phase.build_features(fixture, 'ultimos6')
        self.assertEqual(x.observaciones_previas.tolist(), [0, 1, 2, 3, 3])
        self.assertEqual(x.meses_desde_primera_observacion.tolist(), [0, 2, 5, 6, 5])
        self.assertEqual(x.meses_desde_observacion_previa.tolist(), [-1, 2, 3, 1, 1])
        sparse = fixture.iloc[[0, 4]].reset_index(drop=True)
        last = phase.build_features(sparse, 'ultimos6').iloc[-1]
        self.assertEqual(last.observaciones_previas, 0)
        self.assertEqual(last.meses_desde_observacion_previa, -1)
        self.assertEqual(last.meses_desde_primera_observacion, 0)

    def test_only_three_columns_change(self):
        frame = self.df.drop(columns='objetivo')
        full, six = [phase.build_features(frame, p) for p in ['completo', 'ultimos6']]
        changed = {c for c in full if not full[c].equals(six[c])}
        self.assertTrue(changed <= {'observaciones_previas', 'meses_desde_observacion_previa',
                                    'meses_desde_primera_observacion'})
        pd.testing.assert_frame_equal(full, phase.build_features(frame, 'decay'))

    def test_future_features_do_not_change_past(self):
        frame = self.df.drop(columns='objetivo')
        cutoff = frame.mes.le(202608)
        altered = frame.copy()
        altered.loc[~cutoff, 'saldo_promedio'] *= 10
        for policy in ['completo', 'ultimos6']:
            expected = phase.build_features(frame.loc[cutoff].reset_index(drop=True), policy)
            actual = phase.build_features(altered, policy).loc[cutoff].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected)

    def test_balanced_recomputed_per_fit(self):
        y = np.array([0, 0, 0, 1])
        w, info = phase.training_weights(y, [202601]*4, 'balanced', 'completo', 3)
        self.assertEqual(info['positive_factor'], 3)
        self.assertAlmostEqual(w[y==1].sum(), w[y==0].sum())
        _, new = phase.training_weights([0, 0, 0, 1, 1], [202602]*5, 'balanced', 'completo', 3)
        self.assertEqual(new['positive_factor'], 1.5)
        self.assertEqual(info['positive_factor'], 3)

    def test_temporal_class_product_and_normalization(self):
        months = [202601, 202604, 202607, 202607]
        y = [0, 0, 0, 1]
        w, info = phase.training_weights(y, months, '2', 'decay', 3)
        raw = np.array([0.25, 0.5, 1, 2])
        np.testing.assert_allclose(w, raw/raw.mean())
        self.assertAlmostEqual(w.mean(), 1)
        self.assertEqual(info['anchor_month'], 202607)
        _, balanced = phase.training_weights(y, months, 'balanced', 'decay', 3)
        self.assertEqual(balanced['positive_factor'], 3)  # Conteos, no masas temporales.
        control, _ = phase.training_weights(y, months, '1', 'completo', 3)
        self.assertIsNone(control)

    def test_known_status_uses_all_history(self):
        base = self.df.iloc[[0]].copy()
        fixture = pd.concat([base, base], ignore_index=True)
        fixture['mes'] = [202601, 202610]
        result = phase.prediction_frame(fixture, np.array([1]), np.array([0.2]), np.array([0]))
        self.assertEqual(result.segmento.iloc[0], 'conocido')

    def test_no_november_and_temporal_splits(self):
        self.assertEqual(self.df.mes.max(), 202610)
        for month in phase.MONTHS:
            inner, stop, refit, outer = phase.fold_indices(self.df, month)
            self.assertTrue(self.df.iloc[inner].mes.lt(month-1).all())
            self.assertTrue(self.df.iloc[stop].mes.eq(month-1).all())
            self.assertTrue(self.df.iloc[refit].mes.lt(month).all())
            self.assertTrue(self.df.iloc[outer].mes.eq(month).all())


def smoke_test():
    """python test_baseline_datafest_2_0.py --smoke (requiere dependencias ML).

    Recorre los doce experimentos y toda la integracion sobre 800 clientes
    reales. Los resultados quedan en una carpeta temporal, nunca en Fase 1.
    """
    import contextlib
    import hashlib
    import io
    import json
    import runpy
    import tempfile
    from sklearn.metrics import roc_auc_score
    work = Path(tempfile.mkdtemp(prefix='datafest-phase2-smoke-'))
    train = pd.read_csv(ROOT/'train.csv')
    ids = np.random.default_rng(2026).choice(train.id_cliente.unique(), 800, replace=False)
    fixture = train.loc[train.id_cliente.isin(ids)]
    fixture.to_csv(work/'train.csv', index=False)
    out = work/'results'
    original = hashlib.sha256((ROOT/'baseline_datafest_1.0.py').read_bytes()).hexdigest()
    sys.argv = ['baseline_datafest_2.0.py', '--quick', '--data-dir', str(work),
                '--output', str(out), '--threads', '2']
    runpy.run_path(str(ROOT/'baseline_datafest_2.0.py'), run_name='__main__')
    exploration = pd.read_csv(out/'metricas_experimentos.csv')
    final = pd.read_csv(out/'metricas_finalistas.csv', dtype={'semilla': str})
    ranking = pd.read_csv(out/'ranking_experimentos.csv')
    pred = pd.read_csv(out/'predicciones_finalistas.csv', dtype={'semilla': str})
    winner = json.loads((out/'combinacion_ganadora.json').read_text(encoding='utf-8'))['combinacion']
    assert len(exploration) == 180 and len(final) == 195 and len(ranking) == 12
    assert set(exploration.semilla) == {2026} and set(pred.combinacion) == {winner['id']}
    for (model, seed, month), g in pred.groupby(['modelo', 'semilla', 'mes']):
        row = final.loc[(final.modelo == model) & (final.semilla == seed) &
                        (final.mes == month) & (final.segmento == 'todos')].iloc[0]
        assert abs(row.auc-roc_auc_score(g.objetivo, g.prediccion)) < 1e-12
    for model in phase.FINALISTS:
        for month in phase.MONTHS:
            p = pred.loc[(pred.modelo == model) & (pred.mes == month)].pivot(
                index='id_cliente', columns='semilla', values='prediccion')
            ens = pred.loc[(pred.modelo == model+'_3semillas') & (pred.mes == month)].set_index('id_cliente').prediccion
            assert set(p.columns) == {'2026', '2027', '2028'}
            np.testing.assert_allclose(ens.sort_index(), p.mean(axis=1).sort_index(), atol=1e-14)
    for month in phase.MONTHS:
        base = pred.loc[pred.modelo.isin([m+'_3semillas' for m in phase.FINALISTS]) & (pred.mes == month)]
        ensemble = pred.loc[(pred.modelo == 'ensemble_uniforme_9modelos') & (pred.mes == month)]
        np.testing.assert_allclose(ensemble.set_index('id_cliente').prediccion.sort_index(),
                                   base.groupby('id_cliente').prediccion.mean().sort_index(), atol=1e-14)
    for path in (out/'ajustes').rglob('ajuste.json'):
        meta = json.loads(path.read_text(encoding='utf-8'))
        month = int(path.parent.name)
        for key, end in [('weights_inner', month-1), ('weights_refit', month)]:
            y = fixture.loc[fixture.mes.lt(end), 'objetivo']
            info = meta[key]
            assert info['n_positive'] == int(y.sum()) and info['n_negative'] == int(len(y)-y.sum())
            if meta['combo']['class_mode'] == 'balanced':
                assert abs(info['positive_factor']-(len(y)-y.sum())/y.sum()) < 1e-12
            assert abs(info['weight_mean']-1) < 1e-12
    before = {str(p): p.stat().st_mtime_ns for p in (out/'ajustes').rglob('ajuste.json')}
    assert len(before) == 100
    sys.argv += ['--resume']
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runpy.run_path(str(ROOT/'baseline_datafest_2.0.py'), run_name='__main__')
    after = {str(p): p.stat().st_mtime_ns for p in (out/'ajustes').rglob('ajuste.json')}
    assert before == after and buffer.getvalue().count('Reutilizado:') == 105
    assert hashlib.sha256((ROOT/'baseline_datafest_1.0.py').read_bytes()).hexdigest() == original
    print('SMOKE_OK: 12 experimentos, pesos por particion, AUC recalculado, 3 finalistas x 3 semillas, '
          'promedios exactos, 100 checkpoints y reanudacion sin reentrenar. Fase 1 intacta.')
    print('TEST_OUTPUT', out)


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        smoke_test()
    else:
        unittest.main()
