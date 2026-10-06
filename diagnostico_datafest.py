"""Auditoria reproducible de los CSV originales; no modifica las fuentes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    all_x = pd.concat([train.drop(columns='objetivo'), test], ignore_index=True)
    positives = train.loc[train.objetivo.eq(1), ['id_cliente', 'mes']]
    after = all_x.merge(positives, on='id_cliente', suffixes=('', '_conversion'))
    monthly = train.groupby('mes').objetivo.agg(['size', 'sum', 'mean'])
    numeric = train.select_dtypes(include='number').drop(columns=['id_cliente', 'mes', 'objetivo'])
    drift = []
    nov = train.loc[train.mes.eq(202611)]
    for col in numeric:
        scale = nov[col].std()
        drift.append({'variable': col, 'nov_mean': float(nov[col].mean()),
                      'dec_mean': float(test[col].mean()),
                      'standardized_mean_difference': float((test[col].mean()-nov[col].mean())/scale) if scale else None})
    ordered = all_x.sort_values(['id_cliente', 'mes'])
    previous = ordered.groupby('id_cliente').shift()
    repeated = previous.mes.notna()
    changes = {c: float(ordered.loc[repeated, c].ne(previous.loc[repeated, c]).mean())
               for c in all_x.columns if c not in ['id_cliente', 'mes']}
    report = {
        'interaction_monthly': [dict(month=int(m), equal_fraction=float(g.dias_ultima_transaccion.eq(g.dias_ultima_interaccion).mean()), correlation=float(g.dias_ultima_transaccion.corr(g.dias_ultima_interaccion))) for m,g in all_x.groupby('mes')],
        'train_shape': list(train.shape), 'test_shape': list(test.shape),
        'sample_shape': list(sample.shape),
        'train_months': sorted(train.mes.unique().tolist()),
        'test_months': sorted(test.mes.unique().tolist()),
        'missing_train': int(train.isna().sum().sum()), 'missing_test': int(test.isna().sum().sum()),
        'duplicate_client_month_train': int(train.duplicated(['id_cliente', 'mes']).sum()),
        'duplicate_client_month_test': int(test.duplicated(['id_cliente', 'mes']).sum()),
        'target_counts': train.objetivo.value_counts().to_dict(),
        'unique_train_clients': int(train.id_cliente.nunique()),
        'test_clients_seen_in_train': int(test.id_cliente.isin(train.id_cliente).sum()),
        'rows_after_conversion': int(after.mes.gt(after.mes_conversion).sum()),
        'clients_with_multiple_conversions': int(positives.id_cliente.duplicated().sum()),
        'sample_ids_exact_order': bool(sample.id_cliente.equals(test.id_cliente)),
        'sample_columns': sample.columns.tolist(),
        'feature_schema_matches': train.drop(columns='objetivo').columns.tolist() == test.columns.tolist(),
        'monthly': monthly.reset_index().to_dict(orient='records'),
        'numeric_summary': numeric.describe().round(3).to_dict(),
        'categorical_counts': {c: train[c].value_counts().to_dict() for c in train.select_dtypes(include=['object','bool'])},
        'unseen_test_categories': {c: sorted(set(test[c])-set(train[c])) for c in train.select_dtypes(include='object')},
        'fraction_changed_between_observations': changes,
        'nov_to_dec_numeric_drift': drift,
    }
    (ROOT / 'diagnostico_datafest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['numeric_summary','categorical_counts']}, ensure_ascii=False, indent=2))
    print(numeric.describe().round(2).to_string())

if __name__ == '__main__':
    main()
