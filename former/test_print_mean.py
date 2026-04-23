import os
import pandas as pd

# 定义季节与月份的映射关系
season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

# 基础路径和海洋列表
BASE_DATA_DIR = '/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv/'
WEIGHTED_FILE = '/home/chenyiqi/251028_albedo_cot/processed_data/ocean_season_sza_weighted.csv'
OCEANS = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']

def load_weighted_angles(weighted_file_path):
    """加载加权角度数据，返回字典便于快速查询"""
    if not os.path.isfile(weighted_file_path):
        print(f"⚠️  加权角度文件不存在: {weighted_file_path}")
        return None
    
    try:
        df_weighted = pd.read_csv(weighted_file_path)
        # 检查必要列是否存在
        required_cols = ['ocean', 'season', 'weighted_angle_deg']
        if not all(col in df_weighted.columns for col in required_cols):
            print(f"❌ 加权角度文件缺少必要列，当前列：{df_weighted.columns.tolist()}")
            return None
        
        # 构建查询字典: {(ocean, season): weighted_angle_deg}
        weight_dict = {}
        for _, row in df_weighted.iterrows():
            ocean = row['ocean']
            season = row['season']
            weight_dict[(ocean, season)] = row['weighted_angle_deg']
        
        return weight_dict
    except Exception as e:
        print(f"❌ 加载加权角度文件出错: {str(e)}")
        return None

def calculate_seasonal_averages(ocean_list, base_dir, weight_dict):
    """计算季节平均COT/SZA，并整合加权角度一起打印"""
    all_results = {}
    
    for ocean in ocean_list:
        # 构建数据文件路径
        file_path = os.path.join(base_dir, f'{ocean}.csv')
        
        # 检查文件是否存在
        if not os.path.isfile(file_path):
            print(f"\n⚠️  文件不存在: {file_path}")
            all_results[ocean] = None
            continue
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 1. 从time列提取月份
            df['month'] = pd.to_datetime(df['time'], format='mixed').dt.month
            
            # 2. 分配季节标签
            df['season'] = None
            for season_name, months in season_dict.items():
                df.loc[df['month'].isin(months), 'season'] = season_name
            
            # 3. 按季节分组计算平均值（仅保留有有效季节标签的行）
            seasonal_avg = df.groupby('season')[['cot_mod08', 'sza']].mean()
            
            # 4. 存储当前海洋的结果
            all_results[ocean] = seasonal_avg
            
            # 5. 打印整合后的结果
            print(f"\n========== {ocean} 季节统计数据 ==========")
            # 格式化表头，保证对齐
            print(f"{'季节':<6} | {'平均COT':<10} | {'平均SZA(°)':<10} | {'加权SZA(°)':<10}")
            print("-" * 50)
            
            for season in season_dict.keys():
                # 初始化默认值
                avg_cot = "无数据"
                avg_sza = "无数据"
                weighted_angle = "无数据"
                
                # 获取平均COT和SZA
                if season in seasonal_avg.index:
                    avg_cot = f"{seasonal_avg.loc[season, 'cot_mod08']:.4f}"
                    avg_sza = f"{seasonal_avg.loc[season, 'sza']:.2f}"
                
                # 获取加权角度
                if weight_dict is not None and (ocean, season) in weight_dict:
                    weighted_angle = f"{weight_dict[(ocean, season)]:.2f}"
                
                # 打印一行数据
                print(f"{season:<6} | {avg_cot:<10} | {avg_sza:<10} | {weighted_angle:<10}")
        
        except Exception as e:
            print(f"\n❌ 处理 {ocean} 时出错: {str(e)}")
            all_results[ocean] = None
    
    return all_results

# 主执行逻辑
if __name__ == '__main__':
    # 第一步：加载加权角度数据
    weight_data = load_weighted_angles(WEIGHTED_FILE)
    
    # 第二步：计算并打印整合后的统计数据
    results = calculate_seasonal_averages(OCEANS, BASE_DATA_DIR, weight_data)