import os
import pickle
import numpy as np
import xgboost as xgb

from data_processor import load_and_combine_data, preprocess_data
from feature_engineer import create_all_features, generate_labels
import model_trainer_fixed as mt


def build_and_predict(data_dir='./data', target_col='label5'):
    df = load_and_combine_data(base_dir=data_dir)
    if df.empty:
        raise RuntimeError('No data loaded')
    df = preprocess_data(df)
    df = create_all_features(df)
    df = generate_labels(df)
    df = mt.classify_by_daily_trend(df)
    df = mt.preprocess_volume_and_features(df)
    df_train, df_test = mt.split_data_time_series(df, train_ratio=0.7)
    X_train, y_train, feature_cols = mt.prepare_X_y(df_train, target_col=target_col)
    X_test, y_test, _ = mt.prepare_X_y(df_test, target_col=target_col)
    X_train_s, X_test_s, scaler = mt.standardize_features(X_train, X_test)

    # Train two XGBoost models with softprob to get probabilities
    def train_softprob(seed):
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': seed,
            'nthread': 4
        }
        dtrain = xgb.DMatrix(X_train_s, label=y_train)
        model = xgb.train(params, dtrain, num_boost_round=100)
        return model

    print('Training ensemble model 1...')
    m1 = train_softprob(seed=42)
    print('Training ensemble model 2...')
    m2 = train_softprob(seed=7)

    dtest = xgb.DMatrix(X_test_s)
    proba1 = m1.predict(dtest)
    proba2 = m2.predict(dtest)

    proba_avg = (proba1 + proba2) / 2.0

    # Ensure model_artifacts directory exists
    out_dir = 'model_artifacts'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'voting_ensemble_results.pkl')

    payload = {
        'probas': [proba1, proba2],
        'proba_avg': proba_avg,
        'y_test': y_test.values,
        'feature_cols': feature_cols
    }

    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)

    print(f'✅ Saved voting ensemble results to: {out_path}')


if __name__ == '__main__':
    build_and_predict(data_dir='./data', target_col='label5')
