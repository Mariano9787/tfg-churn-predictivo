"""TP4 IBM Telco: CV model selection, sealed holdout, full-portfolio OOF and SHAP.

Usage: python -m src.complete_pipeline
Requires the existing P4 modules, split files and requirements.txt.
All files are local; the held-out test is read only AFTER configuration selection.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import shap
import xgboost
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES, build_model_pipeline
from src.validate_dataset import DATASET_PATH, EXPECTED_SHA256, calculate_sha256

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'data' / 'processed' / 'train.parquet'
TEST_PATH = ROOT / 'data' / 'processed' / 'test.parquet'
SPLIT_MANIFEST = ROOT / 'artifacts' / 'split_manifest.json'
OUT = ROOT / 'artifacts' / 'final'
SEED = 42
N_FOLDS = 5
SENSITIVE = ['gender', 'SeniorCitizen']


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def preflight() -> dict:
    if calculate_sha256(DATASET_PATH) != EXPECTED_SHA256:
        raise RuntimeError('Original CSV hash changed; stopping.')
    m = json.loads(SPLIT_MANIFEST.read_text(encoding='utf-8'))
    for p, key in [(TRAIN_PATH, 'train_sha256'), (TEST_PATH, 'test_sha256')]:
        if not p.is_file() or file_hash(p) != m[key]:
            raise RuntimeError(f'Split integrity error: {p}; stopping.')
    return m


def features(data: pd.DataFrame, include_sensitive: bool = True):
    cols = [c for c in FEATURE_COLUMNS if include_sensitive or c not in SENSITIVE]
    X = data[cols].copy()
    y = data['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    return X, y


def model_factory(name: str, prevalence: float):
    if name == 'logistic':
        return LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=SEED), True
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=120, max_depth=9, min_samples_leaf=8,
                                     class_weight='balanced_subsample', n_jobs=-1, random_state=SEED), False
    if name == 'xgboost':
        return XGBClassifier(n_estimators=160, learning_rate=0.06, max_depth=3,
                             subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
                             scale_pos_weight=(1 - prevalence) / prevalence,
                             objective='binary:logistic', eval_metric='logloss',
                             n_jobs=2, random_state=SEED), False
    if name == 'dummy':
        return DummyClassifier(strategy='prior'), False
    raise ValueError(name)


def pipeline(name: str, prevalence: float, include_sensitive: bool = True):
    model, scale = model_factory(name, prevalence)
    obj = build_model_pipeline(model, scale_numeric=scale)
    if not include_sensitive:
        prep = obj.named_steps['preprocessor']
        new_transforms = []
        for key, step, cols in prep.transformers:
            if key == 'categorical':
                cols = [c for c in cols if c not in SENSITIVE]
            new_transforms.append((key, step, cols))
        prep.set_params(transformers=new_transforms)
    return obj


def cv_predictions(data: pd.DataFrame, name: str, include_sensitive=True):
    X, y = features(data, include_sensitive)
    prevalence = float(y.mean())
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = np.full(len(data), np.nan)
    ap_folds = []
    roc_folds = []
    for i, (tr, va) in enumerate(cv.split(X, y), start=1):
        m = pipeline(name, prevalence, include_sensitive)
        m.fit(X.iloc[tr], y.iloc[tr])
        prob = m.predict_proba(X.iloc[va])[:, 1]
        scores[va] = prob
        ap = float(average_precision_score(y.iloc[va], prob))
        roc = float(roc_auc_score(y.iloc[va], prob))
        ap_folds.append(ap)
        roc_folds.append(roc)
        print(f'  {name} {"full" if include_sensitive else "reduced"}: fold {i}/{N_FOLDS} AP={ap:.4f}', flush=True)
    if not np.isfinite(scores).all():
        raise AssertionError('CV incomplete.')
    return {'scores': scores, 'ap_folds': ap_folds, 'roc_folds': roc_folds,
            'ap_mean': float(np.mean(ap_folds)), 'ap_se': float(np.std(ap_folds, ddof=1)/np.sqrt(N_FOLDS)),
            'roc_mean': float(np.mean(roc_folds))}


def decile_table(y, scores):
    tmp = pd.DataFrame({'y': np.asarray(y, dtype=int), 'score': np.asarray(scores, dtype=float)})
    tmp = tmp.sort_values('score', ascending=False, kind='stable').reset_index(drop=True)
    total = int(tmp.y.sum())
    rows = []
    for decile in range(1, 11):
        k = int(np.ceil(len(tmp) * decile / 10))
        count = int(tmp.y.iloc[:k].sum())
        rows.append({'decile': decile, 'top_share_percent': 10 * decile,
                     'clients': k, 'true_churn_captured': count,
                     'precision_cumulative': round(count/k, 6),
                     'recall_cumulative': round(count/total, 6) if total else None})
    return pd.DataFrame(rows)


def group_diagnostics(data, scores, target):
    records = []
    y = np.asarray(target, dtype=int)
    pred = np.asarray(scores, dtype=float)
    cutoff = np.quantile(pred, 0.9)
    for column in SENSITIVE:
        for group in sorted(data[column].dropna().unique(), key=str):
            mask = np.asarray(data[column] == group)
            yt, pr = y[mask], pred[mask]
            records.append({'variable': column, 'group': str(group), 'n': int(mask.sum()),
                            'churn_prevalence': float(yt.mean()),
                            'average_precision': float(average_precision_score(yt, pr)) if yt.sum() > 0 else None,
                            'roc_auc': float(roc_auc_score(yt, pr)) if len(np.unique(yt)) == 2 else None,
                            'brier': float(brier_score_loss(yt, pr)),
                            'top_decile_selection_rate': float((pr >= cutoff).mean())})
    return pd.DataFrame(records)


def shap_for_fold(fitted_pipeline, X_train, X_valid, selected_name):
    """SHAP for the fitted fold model in one consistent native output space.

    logistic: log-odds, xgboost: log-odds (raw margin), RF: class-1 probability.
    """
    pre = fitted_pipeline.named_steps['preprocessor']
    model = fitted_pipeline.named_steps['model']
    Z_train = np.asarray(pre.transform(X_train), dtype=float)
    Z_valid = np.asarray(pre.transform(X_valid), dtype=float)
    if selected_name == 'logistic':
        # Exact linear SHAP relative to the training-fold background mean.
        # This is equivalent to (x - E_train[x]) * fitted coefficient.
        # shap.LinearExplainer is used for reproducibility of the semantics.
        background = shap.sample(Z_train, min(100, len(Z_train)), random_state=SEED)
        explainer = shap.LinearExplainer(model, background)
        values = np.asarray(explainer.shap_values(Z_valid))
        space = 'log_odds'
    else:
        explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
        values = np.asarray(explainer.shap_values(Z_valid))
        if values.ndim == 3 and values.shape[-1] == 2:
            values = values[:, :, 1]
        space = 'probability' if selected_name == 'random_forest' else 'log_odds'
    if values.shape != Z_valid.shape:
        raise RuntimeError(f'Unexpected SHAP shape: {values.shape} vs {Z_valid.shape}')
    names = pre.get_feature_names_out()
    origins = []
    for name in names:
        if name.startswith('numeric__'):
            origins.append(name.removeprefix('numeric__'))
        elif name.startswith('categorical__'):
            raw = name.removeprefix('categorical__')
            match = [col for col in FEATURE_COLUMNS if raw.startswith(col + '_')]
            if len(match) != 1:
                raise RuntimeError(f'Cannot consolidate SHAP feature: {name}')
            origins.append(match[0])
        else:
            raise RuntimeError(f'Unexpected transformed feature: {name}')
    out = pd.DataFrame(values.T).groupby(origins).sum().T
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out[FEATURE_COLUMNS].to_numpy(), space


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    split_meta = preflight()
    if '--portfolio-only' in sys.argv:
        if not (OUT / 'holdout_seal.json').exists():
            raise RuntimeError('No sealed holdout; run the complete pipeline first.')
        lock = json.loads((OUT / 'selection_locked.json').read_text(encoding='utf-8'))
        seal = json.loads((OUT / 'holdout_seal.json').read_text(encoding='utf-8'))
        for label, filename in [('selection_locked_sha256','selection_locked.json'),
                                ('holdout_metrics_sha256','holdout_metrics.json'),
                                ('holdout_deciles_sha256','holdout_deciles.csv'),
                                ('holdout_scores_sha256','holdout_scores.parquet'),
                                ('holdout_calibration_sha256','holdout_calibration.csv')]:
            if file_hash(OUT / filename) != seal[label]:
                raise RuntimeError('The sealed holdout artifacts were altered; stopping.')
        run_portfolio(lock['model'], lock['include_sensitive'])
        return
    if (OUT / 'holdout_seal.json').exists():
        raise RuntimeError('Holdout already sealed. Use --portfolio-only to finish the OOF portfolio without repeating test evaluation.')
    train = pd.read_parquet(TRAIN_PATH)
    if len(train) != 5634 or train.customerID.duplicated().any():
        raise RuntimeError('Unexpected training split.')
    y_train = train.Churn.map({'No': 0, 'Yes': 1}).astype(int)
    print('\nPHASE 1: five-fold training-only CV, holdout remains sealed.', flush=True)
    families = {}
    for name in ['dummy', 'logistic', 'random_forest', 'xgboost']:
        families[name] = cv_predictions(train, name)
    rows = []
    for name, result in families.items():
        rows.append({'model': name, 'cv_ap_mean': result['ap_mean'], 'cv_ap_se': result['ap_se'],
                     'cv_roc_auc_mean': result['roc_mean'], 'cv_ap_folds': json.dumps(result['ap_folds'])})
    pd.DataFrame(rows).to_csv(OUT/'cv_models.csv', index=False)
    best = max(['logistic', 'random_forest', 'xgboost'], key=lambda n: families[n]['ap_mean'])
    chosen = 'logistic' if families['logistic']['ap_mean'] >= (families[best]['ap_mean'] - families[best]['ap_se']) else best
    print(f'CV best={best}; one-SE selected={chosen}', flush=True)

    # E1 and E2: training-only sensitivity, pre-declared conservative rule.
    print('\nPHASE 2: pre-holdout sensitive-feature sensitivity.', flush=True)
    reduced = cv_predictions(train, chosen, include_sensitive=False)
    full = families[chosen]
    paired = np.asarray(full['ap_folds']) - np.asarray(reduced['ap_folds'])
    diff_mean = float(paired.mean())
    diff_se = float(paired.std(ddof=1) / np.sqrt(N_FOLDS))
    include_sensitive = bool(diff_mean > diff_se)
    full_rank = full['scores']; reduced_rank = reduced['scores']
    n10 = int(np.ceil(len(train)*0.1))
    set_full = set(np.argsort(-full_rank, kind='stable')[:n10])
    set_reduced = set(np.argsort(-reduced_rank, kind='stable')[:n10])
    sensitivity = {'candidate_model': chosen, 'ap_cv_full': full['ap_mean'],
                   'ap_cv_reduced': reduced['ap_mean'], 'paired_difference_full_minus_reduced': diff_mean,
                   'paired_difference_se': diff_se, 'spearman_rho': float(spearmanr(full_rank, reduced_rank).statistic),
                   'top_10_percent_overlap': len(set_full & set_reduced)/n10,
                   'sensitive_features_retained': include_sensitive,
                   'decision_rule': 'Retain sensitive attributes only if 5-fold paired AP improvement exceeds 1 SE; otherwise prefer parsimonious exclusion. This is not a statistical fairness guarantee.',
                   'limitation': 'Five folds are not independent; SE is a heuristic. AP depends on prevalence.'}
    write_json(OUT/'sensitivity.json', sensitivity)
    selected_oof = full_rank if include_sensitive else reduced_rank
    pd.DataFrame({'customerID': train.customerID.to_numpy(), 'y': y_train.to_numpy(),
                  'cv_score': selected_oof}).to_parquet(OUT/'training_cv_scores.parquet', index=False)
    group_diagnostics(train, selected_oof, y_train).to_csv(OUT/'training_group_diagnostics.csv', index=False)

    selection = {'model': chosen, 'full_model_cv_best': best, 'include_sensitive': include_sensitive,
                 'preprocessing': 'numeric median imputation; logistic standard scaling; categorical mode imputation + one-hot',
                 'holdout_policy': 'single final performance evaluation AFTER all selection is locked',
                 'seed': SEED, 'n_folds': N_FOLDS,
                 'cv_model_results': rows,
                 'model_config': pipeline(chosen, float(y_train.mean()), include_sensitive).get_params(deep=False)}
    # Lock config to disk BEFORE first test read. No selected configuration changes afterward.
    write_json(OUT/'selection_locked.json', selection)
    print(f'Configuration LOCKED. sensitive features kept={include_sensitive}', flush=True)

    print('\nPHASE 3: one-time holdout evaluation (P9).', flush=True)
    test = pd.read_parquet(TEST_PATH)
    if len(test) != 1409 or set(train.customerID) & set(test.customerID):
        raise RuntimeError('Holdout integrity invalid.')
    Xtr, _ = features(train, include_sensitive)
    Xte, yte = features(test, include_sensitive)
    m = pipeline(chosen, float(y_train.mean()), include_sensitive)
    m.fit(Xtr, y_train)
    score = m.predict_proba(Xte)[:, 1]
    dummy = pipeline('dummy', float(y_train.mean()), include_sensitive)
    dummy.fit(Xtr, y_train)
    dummy_score = dummy.predict_proba(Xte)[:, 1]
    metrics = {'selected_family': chosen, 'include_sensitive': include_sensitive,
               'train_n': len(train), 'test_n': len(test), 'test_churn_n': int(yte.sum()),
               'average_precision': float(average_precision_score(yte, score)),
               'roc_auc': float(roc_auc_score(yte, score)),
               'brier': float(brier_score_loss(yte, score)),
               'dummy_average_precision': float(average_precision_score(yte, dummy_score)),
               'dummy_roc_auc': float(roc_auc_score(yte, dummy_score)),
               'dummy_brier': float(brier_score_loss(yte, dummy_score)),
               'note': 'This is the only final holdout evaluation; no re-selection or post-hoc tuning.'}
    write_json(OUT/'holdout_metrics.json', metrics)
    decile_table(yte, score).to_csv(OUT/'holdout_deciles.csv', index=False)
    pd.DataFrame({'customerID': test.customerID.to_numpy(), 'y': yte.to_numpy(),
                  'score': score}).to_parquet(OUT/'holdout_scores.parquet', index=False)
    frac_pos, mean_pred = calibration_curve(yte, score, n_bins=10, strategy='quantile')
    pd.DataFrame({'mean_predicted': mean_pred, 'fraction_positive': frac_pos}).to_csv(OUT/'holdout_calibration.csv', index=False)
    group_diagnostics(test, score, yte).to_csv(OUT/'holdout_group_diagnostics.csv', index=False)
    joblib.dump(m, OUT/'selected_model.joblib')
    # Seal the holdout evaluation BEFORE generating the full portfolio.
    holdout_seal = {'selection_locked_sha256': file_hash(OUT/'selection_locked.json'),
                    'holdout_metrics_sha256': file_hash(OUT/'holdout_metrics.json'),
                    'holdout_deciles_sha256': file_hash(OUT/'holdout_deciles.csv'),
                    'holdout_scores_sha256': file_hash(OUT/'holdout_scores.parquet'),
                    'holdout_calibration_sha256': file_hash(OUT/'holdout_calibration.csv')}
    write_json(OUT/'holdout_seal.json', holdout_seal)
    print(f"Holdout sealed. AP={metrics['average_precision']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}", flush=True)

    run_portfolio(chosen, include_sensitive)


def run_portfolio(chosen: str, include_sensitive: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    print('\nPHASE 4: five-fold full-portfolio OOF (post-evaluation, display ONLY).', flush=True)
    full_data = pd.read_csv(DATASET_PATH, dtype={'TotalCharges': 'string'})
    full_data['TotalCharges'] = pd.to_numeric(full_data['TotalCharges'].str.strip(), errors='coerce')
    if full_data.shape != (7043, 21) or full_data.customerID.duplicated().any():
        raise RuntimeError('Portfolio structure invalid.')
    X, y = features(full_data, include_sensitive)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    probs = np.full(len(full_data), np.nan)
    foldids = np.zeros(len(full_data), dtype=int)
    contributions = np.full((len(full_data), len(FEATURE_COLUMNS)), np.nan)
    fold_space = set()
    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        fold_model = pipeline(chosen, float(y.iloc[tr].mean()), include_sensitive)
        fold_model.fit(X.iloc[tr], y.iloc[tr])
        probs[va] = fold_model.predict_proba(X.iloc[va])[:, 1]
        foldids[va] = fold
        shap_vals, space = shap_for_fold(fold_model, X.iloc[tr], X.iloc[va], chosen)
        contributions[va] = shap_vals
        fold_space.add(space)
        print(f'  portfolio fold {fold}/{N_FOLDS}: {len(va)} out-of-fold predictions and explanations', flush=True)
    if not np.isfinite(probs).all() or not np.isfinite(contributions).all() or len(fold_space) != 1:
        raise RuntimeError('OOF scores/SHAP incomplete or inconsistent.')
    totals = np.abs(contributions).sum(axis=1)
    relative = np.divide(contributions, totals[:, None], out=np.zeros_like(contributions), where=totals[:, None]!=0)
    # Relative contribution is fold-relative, not a global causal effect.
    scores_df = full_data[['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges',
                           'InternetService', 'PaymentMethod', 'gender', 'SeniorCitizen']].copy()
    scores_df['score'] = probs
    scores_df['fold_id'] = foldids
    scores_df['model_criterion_id'] = chosen + ('_with_sensitive' if include_sensitive else '_without_sensitive')
    # Risk tiers use only portfolio OOF empirical quantiles. No fixed probability claims.
    q50, q80 = np.quantile(probs, [0.50, 0.80])
    scores_df['risk_band'] = np.select([probs >= q80, probs >= q50], ['Alto', 'Medio'], default='Bajo')
    scores_df = scores_df.sort_values('score', ascending=False, kind='stable')
    scores_df.to_parquet(OUT/'portfolio_oof.parquet', index=False)
    expl = pd.DataFrame({'customerID': np.repeat(full_data.customerID.to_numpy(), len(FEATURE_COLUMNS)),
                         'variable': np.tile(FEATURE_COLUMNS, len(full_data)),
                         'contribution': contributions.reshape(-1),
                         'relative_contribution': relative.reshape(-1),
                         'fold_id': np.repeat(foldids, len(FEATURE_COLUMNS))})
    expl.to_parquet(OUT/'explanations_oof.parquet', index=False)
    importance = (expl.groupby('variable')['relative_contribution']
                  .apply(lambda s: float(np.mean(np.abs(s))))
                  .reset_index(name='mean_abs_relative'))
    importance.sort_values('mean_abs_relative', ascending=False).to_csv(OUT/'global_factors.csv', index=False)
    write_json(OUT/'portfolio_metadata.json', {'source_sha256': EXPECTED_SHA256, 'rows': len(full_data),
               'folds': N_FOLDS, 'shap_space': next(iter(fold_space)), 'selected_family': chosen,
               'include_sensitive': include_sensitive, 'risk_quantiles': {'q50': float(q50), 'q80': float(q80)},
               'model_criterion_vs_fold_instance': 'model_criterion_id identifies chosen configuration; fold_id identifies concrete OOF-producing fit.',
               'interpretation': 'OOF for presentation only, NOT holdout assessment. Contributions are associative, not causal. Relative SHAP values are locally normalized, not comparable as absolute magnitudes across independently fit instances.'})
    write_json(OUT/'environment.json', {'python': sys.version, 'platform': platform.platform(),
               'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
               'xgboost': xgboost.__version__, 'shap': shap.__version__,
               'generated_at_utc': datetime.now(timezone.utc).isoformat()})
    write_json(OUT/'artifact_manifest.json', {p.name: file_hash(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print('\nCOMPLETE: all metrics, rankings, SHAP, model and manifests under artifacts/final/', flush=True)
    print('Next: python -m streamlit run app/main.py', flush=True)


if __name__ == '__main__':
    main()
