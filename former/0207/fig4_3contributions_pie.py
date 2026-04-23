import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

oceans = ['NPO', 'NAO','TPO',  'TAO', 'TIO', 
          'SPO', 'SAO', 'SIO']
season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

# 五块饼图配色（新增一块深绿，保持色系渐变逻辑）
colors = [
    (1.0, 0.8, 0.8),      # delta1
    (1.0, 0.4, 0.4),      # delta2
    (0.8, 0.0, 0.0),    # delta3
    (0.7, 0.87, 0.98),      # delta4
    (0.0, 0.4, 0.8)      # k
]


def plot_pie_chart(ocean, delta1, delta2, delta3, delta4, slope):
    """绘制五块饼图（delta1-delta4 + k），仅Global显示图例"""
    pie_sizes = [delta1, delta2, delta3, delta4, slope]
    # 五块标签（delta1-delta4，k保持原标签）
    labels = ['$1-k_{\mathrm{dcp}}$', '$k_{\mathrm{dcp}}-k_{\mathrm{cp}}$', 
              '$k_{\mathrm{cp}}-k_{\mathrm{ret}}$', '$k_{\mathrm{ret}}-k_{\mathrm{msk}}$', 
              '$k_{\mathrm{msk}}$']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0.0)  # 背景透明
    
    # 绘制五块饼图，白色粗边框（linewidth=4）
    wedges, texts = ax.pie(
        pie_sizes,
        labels=[None]*5,  # 隐藏默认标签，自定义位置显示数值
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor='w', linewidth=4)
    )
    
    # 为每块饼图添加居中数值（字号32，粗体，黑色）
    for i, w in enumerate(wedges):
        ang = (w.theta2 + w.theta1) / 2  # 计算每块中心角度
        x = np.cos(np.deg2rad(ang)) * 0.63  # 数值位置x（半径0.63）
        y = np.sin(np.deg2rad(ang)) * 0.63  # 数值位置y（半径0.63）
        
        ax.text(
            x, y,
            f"{pie_sizes[i]:.2f}",
            ha='center', va='center',
            fontsize=32, weight='bold',
            color='k'
        )
    
    # 仅Global显示图例（右侧居中，字号15）
    if ocean == 'Global':
        ax.legend(
            wedges,
            labels,
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            fontsize=15,
            borderaxespad=0.
        )
    
    # 海洋名称标注（底部居中，字号45，粗体）
    plt.text(
        0, -1.25,
        f'{ocean}',
        ha='center', va='center',
        fontsize=45,
        weight='bold'
    )
    
    ax.axis('equal')  # 保证饼图为正圆
    
    # 保存透明背景饼图
    plt.savefig(
        f'figs/transparent_pie_chart_{ocean}.png',
        dpi=300,
        bbox_inches='tight',
        transparent=True
    )
    plt.close()


if __name__ == "__main__":
    # 读取斜率截距数据（需确保CSV中包含_Slope_msk/_ret/_cp/_dcp字段）
    df = pd.read_csv('/home/chenyiqi/251028_albedo_cot/processed_data/slopes_intercepts_8oceans_and_global.csv')

    # 初始化结果表格（8海洋+Global，4季节+Annual）
    k_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    delta2_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    delta3_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    delta4_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)  # 新增delta4表格
    lnb_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    
    # 核心修改：base_keys扩展为4个，对应msk/ret/cp/dcp
    base_keys = ['_msk', '_ret', '_cp', '_dcp']
    
    # 遍历8个海洋计算并绘图
    for i, ocean in enumerate(oceans):
        ocean_data = df[df['Ocean'] == ocean].iloc[0]
        
        # 遍历4个季节计算季节值
        for j, season_processed in enumerate(season_dict.keys()):
            # 拼接季节+Slope+base_key（如MAM_Slope_msk, MAM_Slope_ret...）
            keys = [f'{season_processed}_Slope{base_key}' for base_key in base_keys]
            # 4个斜率作差得到delta1-delta4（核心计算逻辑）
            delta1 = 1 - ocean_data[keys[3]]  # 1 - dcp_slope
            delta2 = ocean_data[keys[3]] - ocean_data[keys[2]]  # dcp_slope - cp_slope
            delta3 = ocean_data[keys[2]] - ocean_data[keys[1]]  # cp_slope - ret_slope
            delta4 = ocean_data[keys[1]] - ocean_data[keys[0]]  # ret_slope - msk_slope
            # 计算k值：1 - 所有delta之和
            slope = 1 - delta1 - delta2 - delta3 - delta4
            
            # 赋值到季节表格
            k_table[i, j] = slope
            delta2_table[i, j] = delta2
            delta3_table[i, j] = delta3
            delta4_table[i, j] = delta4  # 新增delta4赋值
            lnb_table[i, j] = ocean_data[f'{season_processed}_Intercept_ret']

        # 计算年平均值（Ann_Slope+base_key）
        keys = [f'Ann_Slope{base_key}' for base_key in base_keys]
        delta1 = 1 - ocean_data[keys[3]]
        delta2 = ocean_data[keys[3]] - ocean_data[keys[2]]
        delta3 = ocean_data[keys[2]] - ocean_data[keys[1]]
        delta4 = ocean_data[keys[1]] - ocean_data[keys[0]]
        slope = 1 - delta1 - delta2 - delta3 - delta4

        # 赋值到年度列（最后一列）
        k_table[i, -1] = slope
        delta2_table[i, -1] = delta2
        delta3_table[i, -1] = delta3
        delta4_table[i, -1] = delta4  # 新增delta4年度赋值
        lnb_table[i, -1] = ocean_data['Ann_Intercept_ret']

        # 绘制该海洋的年平均五块饼图
        plot_pie_chart(ocean, delta1, delta2, delta3, delta4, slope)
    
    # 处理Global数据（最后一行）
    ocean_data = df[df['Ocean'] == 'Global'].iloc[0]
    for j, season_processed in enumerate(season_dict.keys()):
        keys = [f'{season_processed}_Slope{base_key}' for base_key in base_keys]
        delta1 = 1 - ocean_data[keys[3]]
        delta2 = ocean_data[keys[3]] - ocean_data[keys[2]]
        delta3 = ocean_data[keys[2]] - ocean_data[keys[1]]
        delta4 = ocean_data[keys[1]] - ocean_data[keys[0]]
        slope = 1 - delta1 - delta2 - delta3 - delta4
        
        k_table[-1, j] = slope
        delta2_table[-1, j] = delta2
        delta3_table[-1, j] = delta3
        delta4_table[-1, j] = delta4
        lnb_table[-1, j] = ocean_data[f'{season_processed}_Intercept_ret']

    # Global年平均值计算
    keys = [f'Ann_Slope{base_key}' for base_key in base_keys]
    delta1 = 1 - ocean_data[keys[3]]
    delta2 = ocean_data[keys[3]] - ocean_data[keys[2]]
    delta3 = ocean_data[keys[2]] - ocean_data[keys[1]]
    delta4 = ocean_data[keys[1]] - ocean_data[keys[0]]
    slope = 1 - delta1 - delta2 - delta3 - delta4

    k_table[-1, -1] = slope
    delta2_table[-1, -1] = delta2
    delta3_table[-1, -1] = delta3
    delta4_table[-1, -1] = delta4
    lnb_table[-1, -1] = ocean_data['Ann_Intercept_ret']
    
    # 绘制Global的五块饼图（带图例）
    plot_pie_chart('Global', delta1, delta2, delta3, delta4, slope)

    # 构造表格索引和列名（8海洋+Global，4季节+Annual）
    complete_oceans = oceans + ['Global']
    complete_seasons = list(season_dict.keys()) + ['Annual']

    # 保存slope1表格（k值）
    slope1_df = pd.DataFrame(data=k_table, index=complete_oceans, columns=complete_seasons)
    slope1_df.reset_index(inplace=True)
    slope1_df.rename(columns={'index': 'Ocean'}, inplace=True)
    output_file_path = '/home/chenyiqi/251028_albedo_cot/processed_data/table_slope1.csv'
    slope1_df.to_csv(output_file_path, index=False, float_format='%.4f')

    # 保存slope2表格（k + delta3，保持原逻辑）
    slope2_df = pd.DataFrame(data=k_table + delta3_table, index=complete_oceans, columns=complete_seasons)
    slope2_df.reset_index(inplace=True)
    slope2_df.rename(columns={'index': 'Ocean'}, inplace=True)
    output_file_path = '/home/chenyiqi/251028_albedo_cot/processed_data/table_slope2.csv'
    slope2_df.to_csv(output_file_path, index=False, float_format='%.4f')

    # 保存截距表格
    intercept_df = pd.DataFrame(data=lnb_table, index=complete_oceans, columns=complete_seasons)
    intercept_df.reset_index(inplace=True)
    intercept_df.rename(columns={'index': 'Ocean'}, inplace=True)
    output_file_path = '/home/chenyiqi/251028_albedo_cot/processed_data/table_intercept2.csv'
    intercept_df.to_csv(output_file_path, index=False, float_format='%.4f')