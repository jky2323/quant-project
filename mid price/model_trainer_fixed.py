import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def classify_by_daily_trend(df):
    """
    按每支股票的日线走势分类 - 使用close价格计算日内开收
    由于没有n_open，使用每日第一个和最后一个close价格近似
    """
    print("\n--- 阶段1: 按日线走势分类 ---")
    
    df = df.copy()
    
    # 计算每日的开收价（按sym+date分组，使用第一个和最后一个价格）
    daily_stats = df.groupby(['sym', 'date']).agg({
        'n_close': ['first', 'last']
    }).reset_index()
    daily_stats.columns = ['sym', 'date', 'close_open', 'close_last']
    
    daily_stats['daily_return'] = (daily_stats['close_last'] - daily_stats['close_open']) / daily_stats['close_open']
    
    # 定义分类规则: 下跌=-1, 持平=0, 上升=1
    daily_stats['daily_trend'] = pd.cut(daily_stats['daily_return'], 
                                        bins=[-np.inf, -0.001, 0.001, np.inf],
                                        labels=[-1, 0, 1])
    # 使用fillna处理NaN值，然后转换为int
    daily_stats['daily_trend'] = daily_stats['daily_trend'].fillna(0).astype(int)
    
    # 合并回原始数据
    df = df.merge(daily_stats[['sym', 'date', 'daily_trend']], on=['sym', 'date'], how='left')
    
    # 统计分布
    distribution = df['daily_trend'].value_counts().sort_index()
    print(f"✅ 日线分类结果:")
    print(f"   下跌 (-1): {distribution.get(-1, 0)} 天")
    print(f"   持平 (0):  {distribution.get(0, 0)} 天")
    print(f"   上升 (1):  {distribution.get(1, 0)} 天")
    
    return df


def preprocess_volume_and_features(df):
    """
    对成交量和衍生特征应用log变换。
    """
    print("\n--- 阶段2: 特征预处理 ---")
    
    df = df.copy()
    
    # 需要对数变换的列
    log_cols = ['amount_delta', 'volume_change_pct', 'volume_ma5_ratio',
                'volume_ma10_ratio', 'volume_ma20_ratio', 'bid_volume', 'ask_volume']
    
    applied_cols = []
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            applied_cols.append(col)
    
    print(f"✅ 对{len(applied_cols)}个成交量相关特征应用log(x+1)变换")
    
    return df


def split_data_time_series(df, train_ratio=0.7):
    """
    按时间序列划分训练集和测试集。
    """
    print("\n--- 阶段3: 时间序列划分 ---")
    
    # 按样本数量划分
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"✅ 数据已按时间序列划分")
    print(f"   训练集: {len(df_train):,} 样本")
    print(f"   测试集: {len(df_test):,} 样本")
    
    return df_train, df_test


def prepare_X_y(df, target_col='label_5'):
    """
    从DataFrame中分离特征 X 和标签 y。
    """
    print(f"\n--- 阶段4: 数据准备 ---")
    
    # 定义需要排除的列
    LABEL_COLS = [col for col in df.columns if col.startswith('label')]
    ID_COLS = ['date', 'time', 'sym', 'ampm', 'unique_id', 'daily_trend', 'n_close', 'n_midprice', 'amount_delta']
    
    # 丢弃标签为NaN的样本
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"✅ 丢弃NaN标签: {len(df) - len(df_clean)} 行 → {len(df_clean):,} 行")
    
    # 特征列
    feature_cols = [col for col in df_clean.columns 
                   if col not in LABEL_COLS and col not in ID_COLS]
    
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()
    
    # 处理NaN和无穷大
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # 确保标签是整数
    y = y.astype(int)
    
    print(f"✅ 特征准备完成: {len(feature_cols)} 个特征, {len(X):,} 个样本")
    
    return X, y, feature_cols


def standardize_features(X_train, X_test):
    """
    使用StandardScaler标准化特征。
    """
    print(f"\n--- 标准化特征 ---")
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    print(f"✅ 特征已标准化")
    print(f"   训练集: shape={X_train_scaled.shape}")
    print(f"   测试集: shape={X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_xgboost_model(X_train, y_train, num_rounds=500):
    """
    使用XGBoost训练多分类模型。
    """
    print(f"\n--- 阶段5: XGBoost模型训练 ---")
    
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'nthread': -1
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    print(f"⏳ 开始训练XGBoost (轮数={num_rounds})...")
    model = xgb.train(params, dtrain, num_rounds)
    print(f"✅ 训练完成")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能。
    """
    print(f"\n--- 阶段6: 模型评估 ---")
    
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ 测试集准确率: {acc:.4f}")
    print(f"\n📊 分类报告:")
    print(classification_report(y_test, y_pred, 
                              target_names=['下跌(-1)', '持平(0)', '上升(1)']))
    print(f"\n🔲 混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'accuracy': acc,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def get_feature_importance(model, feature_cols, top_n=20):
    """
    Robustly extract feature importance from either an XGBoost Booster or
    an sklearn-like estimator with `feature_importances_`.

    Returns a pandas DataFrame with columns ['feature','importance'] sorted
    by importance descending. If extraction fails, returns empty DataFrame.
    """
    import pandas as pd
    try:
        # Case 1: xgboost.Booster (returned by xgb.train)
        if hasattr(model, 'get_score'):
            importance_dict = model.get_score(importance_type='weight')
            rows = []
            for k, v in importance_dict.items():
                # keys might be 'f0','f1' or actual feature names
                if isinstance(k, str) and k.startswith('f'):
                    try:
                        idx = int(k[1:])
                        name = feature_cols[idx] if idx < len(feature_cols) else k
                    except Exception:
                        name = k
                else:
                    name = k
                rows.append({'feature': name, 'importance': v})
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows).sort_values('importance', ascending=False)
            return df

        # Case 2: sklearn-like estimator with feature_importances_
        if hasattr(model, 'feature_importances_'):
            import numpy as np
            imp = np.array(model.feature_importances_)
            rows = [{'feature': feature_cols[i], 'importance': float(imp[i])}
                    for i in range(min(len(feature_cols), len(imp)))]
            df = pd.DataFrame(rows).sort_values('importance', ascending=False)
            return df

    except Exception as e:
        # Don't raise — return empty DataFrame to keep pipeline robust
        print(f"Warning: failed to extract feature importance: {e}")

    return pd.DataFrame()
