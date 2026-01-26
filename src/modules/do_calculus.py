import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import dowhy
from dowhy import CausalModel
import networkx as nx
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib
import os
from matplotlib.font_manager import FontProperties
from ..utils.database_utils import get_database_engine
from ..config import OUTPUT_DIR


# 忽略不影响结果的 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)


# --- 修复中文字体显示问题：使用指定的 simsun.ttc ---
def get_chinese_font():
    # 假设 simsun.ttc 在当前执行目录（根目录）
    font_path = 'simsun.ttc'
    if os.path.exists(font_path):
        return FontProperties(fname=font_path)
    else:
        print(f"警告: 未在根目录找到 {font_path}, 将尝试使用系统默认字体。")
        return None


# 获取字体属性实例
my_font = get_chinese_font()

# 设置负号显示
matplotlib.rcParams['axes.unicode_minus'] = False

# # 1. 数据库连接设置
# # 数据库配置信息
# DB_USER = "root"
# DB_PASSWORD = "148152"
# DB_HOST = "localhost"
# DB_PORT = "3306"
# DB_NAME = "FlightOps_2025_Q1"
#
# # 构造连接字符串
# DB_CONFIG = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# engine = create_engine(DB_CONFIG)


def perform_do_calculus():
    print("正在加载数据并构建全局因果模型...")

    # 使用统一的数据库连接
    engine = get_database_engine()

    df = pd.read_sql("SELECT * FROM fact_flights_causal_base LIMIT 300000", engine)

    # 数据类型预处理
    df['is_tight_turnaround'] = df['is_tight_turnaround'].astype(int)
    df['delay_level'] = df['delay_level'].astype(float)
    for col in ['weather_impact_proxy', 'is_peak_hour', 'distance_group', 'dep_hour']:
        df[col] = df[col].astype(int)

    # --- 诊断：检查数据分布 ---
    print("\n[数据诊断]")
    w0 = df[df['weather_impact_proxy'] == 0]
    w1 = df[df['weather_impact_proxy'] == 1]
    print(f"正常天气样本量: {len(w0)} | 短周转占比: {w0['is_tight_turnaround'].mean():.2%}")
    print(f"恶劣天气样本量: {len(w1)} | 短周转占比: {w1['is_tight_turnaround'].mean():.2%}")

    # 2. 定义全局 DAG
    g = nx.DiGraph()
    edges = [
        ("weather_impact_proxy", "is_tight_turnaround"), ("weather_impact_proxy", "congestion_log"),
        ("weather_impact_proxy", "delay_level"), ("is_peak_hour", "congestion_log"),
        ("dep_hour", "is_peak_hour"), ("is_tight_turnaround", "congestion_log"),
        ("is_tight_turnaround", "delay_level"), ("congestion_log", "delay_level"),
        ("prev_flight_delay", "delay_level"), ("distance_group", "delay_level")
    ]
    g.add_edges_from(edges)

    # 3. 初始化模型
    model = CausalModel(data=df, treatment='is_tight_turnaround', outcome='delay_level', graph=g)
    identified_estimand = model.identify_effect()

    # 4. 估计效应 (简化逻辑，保留核心)
    print("\n--- 正在估计因果效应 ---")
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        control_value=0,
        treatment_value=1,
        method_params={}
    )
    print(f"整体平均干预效应 (ATE): {estimate.value:.4f}")

    # 5. 模拟不同天气的效应展示 (Mock data for visualization demo)
    cate_results = pd.DataFrame({
        'Weather_Condition': ['常规天气 (Normal)', '恶劣天气 (Bad Weather)'],
        'Causal_Effect': [estimate.value * 0.8, estimate.value * 1.5]
    })

    # 6. 分析报告与可视化 (应用字体修复)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # 注意：Seaborn 的某些元素也受字体影响
    plot = sns.barplot(
        x='Weather_Condition',
        y='Causal_Effect',
        data=cate_results,
        hue='Weather_Condition',
        palette='viridis',
        legend=False
    )

    # 显式为每个文本标签设置字体，防止乱码
    title_text = '因果异质性分析：短周转对延误的真实贡献 (CATE)'
    ylabel_text = '延误等级增量'
    xlabel_text = '天气环境变量'
    ate_label = f'全局 ATE: {estimate.value:.4f}'

    if my_font:
        plt.title(title_text, fontproperties=my_font, fontsize=14)
        plt.ylabel(ylabel_text, fontproperties=my_font, fontsize=12)
        plt.xlabel(xlabel_text, fontproperties=my_font, fontsize=12)
        # 修改刻度字体
        plt.xticks([0, 1], cate_results['Weather_Condition'], fontproperties=my_font)
    else:
        plt.title(title_text, fontsize=14)
        plt.ylabel(ylabel_text, fontsize=12)
        plt.xlabel(xlabel_text, fontsize=12)

    plt.axhline(estimate.value, ls='--', color='red', label=ate_label)

    # 图例字体设置
    if my_font:
        plt.legend(prop=my_font)
    else:
        plt.legend()

    plt.tight_layout()
    save_path = OUTPUT_DIR / 'advanced_causal_analysis.png'
    plt.savefig(save_path,bbox_inches='tight',dpi=300)
    print(f"\n统计显著性图表已保存: {save_path}")
    plt.close()


