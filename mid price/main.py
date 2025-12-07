import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from data_processor import load_and_combine_data, preprocess_data
from feature_engineer import create_all_features, generate_labels
import model_trainer_fixed as mt

from sklearn.metrics import f1_score, confusion_matrix, classification_report


def compute_per_class_accuracy(y_true, y_pred, classes=[0,1,2]):
    accs = {}
    for c in classes:
        mask = (y_true == c)
        if mask.sum() == 0:
            accs[c] = np.nan
        else:
            accs[c] = (y_pred[mask] == y_true[mask]).mean()
    return accs


def plot_class_accuracies(per_class_acc, overall_acc, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    # Use English labels for plots to avoid Chinese font issues
    labels = ['Down (0)', 'Flat (1)', 'Up (2)']
    vals = [per_class_acc.get(0, np.nan), per_class_acc.get(1, np.nan), per_class_acc.get(2, np.nan)]

    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=vals, palette='viridis')
    plt.ylim(0,1)
    plt.title(f'Per-class accuracy (Overall acc={overall_acc:.4f})')
    plt.ylabel('Accuracy')
    for i, v in enumerate(vals):
        plt.text(i, 0.01 + (v if not np.isnan(v) else 0), f"{v:.3f}" if not np.isnan(v) else "n/a", ha='center')
    fname = os.path.join(out_dir, 'per_class_accuracy.png')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"✅ 已保存每类准确率图像到: {fname}")


def plot_confusion_matrix(cm, class_names=None, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    fname = os.path.join(out_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"✅ 已保存混淆矩阵图像到: {fname}")


def main(data_dir='./data', target_col='label5', num_rounds=200):
    """主运行流程：加载数据 -> 预处理 -> 特征 -> 生成标签 -> 训练 -> 评估 -> 可视化"""
    print("🚀 开始主流程")

    # 1) 加载并合并数据
    df = load_and_combine_data(base_dir=data_dir)
    if df.empty:
        print("错误：未加载到数据，退出。")
        return

    # 2) 预处理
    df = preprocess_data(df)

    # 3) 特征工程
    df = create_all_features(df)

    # 4) 生成标签
    df = generate_labels(df)

    # 5) 可选地按日线分类（生成 daily_trend 列）
    df = mt.classify_by_daily_trend(df)

    # 6) 成交量与其他特征预处理
    df = mt.preprocess_volume_and_features(df)

    # 7) 按时间序列划分训练/测试
    df_train, df_test = mt.split_data_time_series(df, train_ratio=0.7)

    # 8) 准备 X 和 y（使用传入的 target_col，例如 'label5'）
    X_train, y_train, feature_cols = mt.prepare_X_y(df_train, target_col=target_col)
    X_test, y_test, _ = mt.prepare_X_y(df_test, target_col=target_col)

    # 9) 标准化
    X_train_s, X_test_s, scaler = mt.standardize_features(X_train, X_test)

    # 10) 训练模型
    model = mt.train_xgboost_model(X_train_s, y_train, num_rounds=num_rounds)

    # 11) 评估并获取预测
    eval_res = mt.evaluate_model(model, X_test_s, y_test)
    y_pred = eval_res['y_pred']
    overall_acc = eval_res['accuracy']
    cm = eval_res['confusion_matrix']

    # 12) 计算 per-class accuracy
    per_class_acc = compute_per_class_accuracy(y_test.values, y_pred, classes=[0,1,2])

    # 13) 打印并可视化
    print('\n--- 评估结果概要 ---')
    print(f"总体准确率: {overall_acc:.4f}")
    print('各类准确率:')
    for k,v in per_class_acc.items():
        print(f"  类 {k}: {v}")

    # F1 分数
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")

    # 更详细的分类报告
    print('\n分类报告:\n')
    print(classification_report(y_test, y_pred, target_names=['下跌(0)','持平(1)','上升(2)']))

    # 可视化并保存图片
    plot_class_accuracies(per_class_acc, overall_acc)
    plot_confusion_matrix(cm, class_names=['下跌(0)','持平(1)','上升(2)'])

    # Feature importance: extract and save if available (robust to model type)
    try:
        fi = mt.get_feature_importance(model, feature_cols, top_n=30)
        if fi is not None and not fi.empty:
            out_dir = 'outputs'
            os.makedirs(out_dir, exist_ok=True)
            plt.figure(figsize=(8,6))
            sns.barplot(x='importance', y='feature', data=fi.head(20))
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()
            fpath = os.path.join(out_dir, 'feature_importance.png')
            plt.savefig(fpath)
            plt.close()
            print(f"✅ Saved feature importance plot: {fpath}")
    except Exception as e:
        print(f"⚠️ Failed to generate feature importance: {e}")

    # Write a concise final report to outputs
    try:
        out_dir = 'outputs'
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, 'FINAL_REPORT.txt')
        with open(report_path, 'w') as rf:
            rf.write('Model Evaluation Report\n')
            rf.write('========================\n')
            rf.write(f'Overall accuracy: {overall_acc:.6f}\n')
            rf.write(f'F1 (macro): {f1_macro:.6f}\n')
            rf.write(f'F1 (weighted): {f1_weighted:.6f}\n')
            rf.write('\nPer-class accuracy:\n')
            for k, v in per_class_acc.items():
                rf.write(f' - Class {k}: {v}\n')
            rf.write('\nConfusion matrix:\n')
            rf.write(np.array2string(cm))
            rf.write('\n')
            voting_path = os.path.join('model_artifacts', 'voting_ensemble_results.pkl')
            if os.path.exists(voting_path):
                rf.write(f'Voting ensemble pickle: {voting_path}\n')
        print(f"✅ Saved final report to: {report_path}")
    except Exception as e:
        print(f"⚠️ Failed to write final report: {e}")

    print("🎯 主流程完成")


if __name__ == '__main__':
    # 你可以修改 target_col 或 num_rounds 来加速/调参
    main(data_dir='./data', target_col='label5', num_rounds=200)
