import pandas as pd
import numpy as np
import glob
import os

def load_and_combine_data(base_dir='./data', file_pattern='snapshot_sym*_date*_*.csv'):
    """
    加载指定目录下所有符合命名规则的CSV文件，并合并成一个大的DataFrame。

    Args:
        base_dir (str): 存储CSV文件的根目录。
        file_pattern (str): 匹配文件名的模式。

    Returns:
        pd.DataFrame: 合并后的所有数据。
    """
    # 使用glob匹配所有文件
    search_path = os.path.join(base_dir, file_pattern)
    all_files = glob.glob(search_path)

    if not all_files:
        print(f"⚠️ Error: No files found in directory {base_dir} matching {file_pattern}. Please check the path.")
        return pd.DataFrame()

    list_df = []
    
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            
            # --- 提取标识符 ---
            # 假设文件名为 snapshot_sym01_date20251201_am.csv
            parts = os.path.basename(filename).split('_')
            sym = parts[1].replace('sym', '')
            date = parts[2].replace('date', '')
            
            # 在某些文件命名格式中，'am/pm'可能在文件名最后
            ampm_part = parts[-1].replace('.csv', '')
            if ampm_part not in ['am', 'pm']:
                 # 尝试从倒数第二部分提取
                ampm_part = parts[-2].replace('.csv', '')

            df['sym'] = sym
            df['date'] = date
            df['ampm'] = ampm_part
            
            # 创建唯一的交易时段标识符
            df['unique_id'] = df['sym'].astype(str) + '_' + df['date'].astype(str) + '_' + df['ampm']
            
            list_df.append(df)
            
        except Exception as e:
            print(f"❌ Error reading file {filename}: {e}")
            continue

    if not list_df:
        return pd.DataFrame()
        
    full_df = pd.concat(list_df, ignore_index=True)
    print(f"✅ Successfully loaded and combined {len(list_df)} files. Total rows: {len(full_df)}")
    return full_df


def preprocess_data(df, threshold=0.1):
    """
    执行数据清洗，包括数据类型转换、缺失值填充，并删除过多缺失值的交易时段。

    Args:
        df (pd.DataFrame): 原始合并数据。
        threshold (float): 如果一个交易时段的缺失值比例超过此阈值，则删除该时段。

    Returns:
        pd.DataFrame: 清洗后的数据。
    """
    if df.empty:
        return df

    # --- 1. 数据类型转换 ---
    # 确定需要转换为数值型的特征列
    # 排除标识符列：date, time, sym, ampm, unique_id
    id_cols = ['date', 'time', 'sym', 'ampm', 'unique_id']
    feature_cols = [col for col in df.columns if col not in id_cols]

    for col in feature_cols:
        # 尝试将特征列转换为数值型。无法转换的错误值将变成 NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 2. 缺失值填充 (ffill) ---
    # 按照 'unique_id' 分组，只在同一个交易时段内用上一个tick的值填充
    # 即，如果 n_midprice[t] 是 NaN，用 n_midprice[t-1] 填充
    print("⏳ Applying Forward Fill (ffill) grouped by 'unique_id'...")
    df[feature_cols] = df.groupby('unique_id')[feature_cols].ffill()
    
    # --- 3. 删除过多缺失值的交易时段 ---
    # 如果一个 unique_id 下的某列（例如 n_midprice）仍然大量缺失（因为该列的第一个值就是缺失值），
    # 我们可以删除整个交易时段。
    
    # 使用 n_midprice 列作为判断标准
    missing_data_info = df.groupby('unique_id')['n_midprice'].apply(lambda x: x.isnull().sum() / len(x))
    
    # 找出缺失值比例超过阈值的 unique_id
    ids_to_drop = missing_data_info[missing_data_info > threshold].index.tolist()

    if ids_to_drop:
        original_count = len(df)
        df = df[~df['unique_id'].isin(ids_to_drop)]
        print(f"🗑️ Dropped {len(ids_to_drop)} sessions (out of {len(missing_data_info)} total) due to >{threshold*100}% missing n_midprice.")
        print(f"   Total rows remaining: {len(df)} (Dropped {original_count - len(df)} rows)")
    else:
        print("✅ No trading sessions dropped due to excessive missing data.")
        
    # 对于剩余的 NaN（通常是每个 session 的第一行，因为 ffill 无法填充），
    # 由于金融数据第一行缺失较难处理，我们可以选择直接删除这些行，或者用 0 填充。
    # 这里选择用 0 填充以保留尽可能多的数据。
    df[feature_cols] = df[feature_cols].fillna(0)

    print("✅ Data cleaning complete.")
    return df

