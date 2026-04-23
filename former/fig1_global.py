# plot_global.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# 导入函数脚本
import Ac_cot_fitting_utils as acfu

def plot_global():
    # 预处理数据
    _, global_processed_data = acfu.preprocess_ocean_data()
    
    # 创建单个子图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 绘制全球子图
    global_title = 'Global'
    global_results = acfu.plot_axes_content(global_processed_data, ax, title=global_title)
    
    # 存储拟合结果
    all_fit_results = []
    
    # 保存全球拟合结果
    global_result_row = {'Ocean': 'Global'}
    if global_results:
        for key in ['ret', 'cp', 'dcp', 'msk', 'LH74']: 
            g_slope, g_intercept, s_slopes, s_intercepts = global_results[key]
            # 全球年度斜率和截距
            global_result_row[f'Ann_Slope_{key}'] = g_slope
            global_result_row[f'Ann_Intercept_{key}'] = g_intercept
            # 季节斜率和截距
            for s_name in acfu.season_dict.keys():
                global_result_row[f'{s_name}_Slope_{key}'] = s_slopes.get(s_name, np.nan)
                global_result_row[f'{s_name}_Intercept_{key}'] = s_intercepts.get(s_name, np.nan)
    else:
        for key in ['ret', 'cp', 'dcp', 'msk', 'LH74']:
            global_result_row[f'Ann_Slope_{key}'] = np.nan
            global_result_row[f'Ann_Intercept_{key}'] = np.nan
            for s_name in acfu.season_dict.keys():
                global_result_row[f'{s_name}_Slope_{key}'] = np.nan
                global_result_row[f'{s_name}_Intercept_{key}'] = np.nan
    all_fit_results.append(global_result_row)
    
    # 设置轴标签
    ax.set_xlabel(r'ln(COT)', fontsize=14)
    ax.set_ylabel(r'$\ln\left[A_{\mathrm{c}}/(1-A_{\mathrm{c}})\right]$', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('figs', exist_ok=True)
    output_fig_path = 'figs/fittings_global.png'
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Global figure saved to: {output_fig_path}")
    
    # 保存拟合结果到CSV
    output_csv_path = '/home/chenyiqi/251028_albedo_cot/processed_data/slopes_intercepts_global.csv'
    output_df = pd.DataFrame(all_fit_results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Global slope and intercept results saved to: {output_csv_path}")

if __name__ == "__main__":
    plot_global()