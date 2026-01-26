import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import dowhy
from dowhy import CausalModel
import networkx as nx
import warnings
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from ..config import OUTPUT_DIR
from src.utils.database_utils import get_database_engine

# 忽略一些版本兼容性产生的警告
warnings.filterwarnings('ignore')

def build_scm_model():
    print("正在加载因果特征数据...")

    # 使用统一的数据库连接
    engine = get_database_engine()

    # 抽取样本进行建模
    query = "SELECT * FROM fact_flights_causal_base LIMIT 100000"
    df = pd.read_sql(query, engine)

    # 显式转换数据类型
    df['is_tight_turnaround'] = df['is_tight_turnaround'].astype(int)
    df['delay_level'] = df['delay_level'].astype(float)
    df['weather_impact_proxy'] = df['weather_impact_proxy'].astype(int)
    df['is_peak_hour'] = df['is_peak_hour'].astype(int)
    df['distance_group'] = df['distance_group'].astype(int)
    df['congestion_log'] = df['congestion_log'].astype(float)
    df['prev_flight_delay'] = df['prev_flight_delay'].astype(float)
    df['dep_hour'] = df['dep_hour'].astype(int)

    # 2. 重新定义因果图
    nodes = [
        "weather_impact_proxy", "is_tight_turnaround", "congestion_log",
        "delay_level", "is_peak_hour", "dep_hour", "prev_flight_delay", "distance_group"
    ]
    edges = [
        ("weather_impact_proxy", "is_tight_turnaround"),
        ("weather_impact_proxy", "congestion_log"),
        ("weather_impact_proxy", "delay_level"),
        ("is_peak_hour", "congestion_log"),
        ("dep_hour", "is_peak_hour"),
        ("is_tight_turnaround", "congestion_log"),
        ("is_tight_turnaround", "delay_level"),
        ("congestion_log", "delay_level"),
        ("prev_flight_delay", "delay_level"),
        ("distance_group", "delay_level")
    ]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # --- 新增：绘制并保存 DAG 图像 ---
    print("正在生成并保存 DAG 结构图...")
    try:
        # 配置中文字体
        font_path = os.path.join(os.getcwd(), "simsun.ttc")
        if os.path.exists(font_path):
            prop = fm.FontProperties(fname=font_path)
        else:
            print(f"警告：未在 {font_path} 找到字体文件，将使用默认字体。")
            prop = None

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(g, seed=42)  # 固定布局

        # 绘制节点
        nx.draw_networkx_nodes(g, pos, node_size=3000, node_color='skyblue', alpha=0.8)
        # 绘制边
        nx.draw_networkx_edges(g, pos, width=2, alpha=0.5, edge_color='gray', arrows=True, arrowsize=20)
        # 绘制标签
        nx.draw_networkx_labels(g, pos, font_size=10, font_family=prop.get_name() if prop else None)

        plt.title("结构因果模型 (SCM) 有向无环图 (DAG)", fontproperties=prop, fontsize=15)
        plt.axis('off')

        # 保存图片到根目录
        output_path = OUTPUT_DIR / "causal_dag.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"DAG 图像已成功保存至: {output_path.absolute()}")
    except Exception as img_e:
        print(f"生成图像失败: {img_e}")

    # 3. 初始化 DoWhy 模型
    print("正在初始化结构因果模型 (SCM)...")
    model = CausalModel(
        data=df,
        treatment='is_tight_turnaround',
        outcome='delay_level',
        graph=g,
        missing_value_treatment="drop"
    )

    # 4. 因果效应识别
    print("\n--- 正在执行因果识别 (Identification) ---")
    identified_estimand = model.identify_effect()

    # 5. 估计因果效应
    print("\n--- 正在估计 ATE (倾向评分加权法) ---")
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_weighting",
        target_units="ate",
        method_params={
            "propensity_score_model": LogisticRegression(solver='lbfgs', max_iter=1000)
        }
    )
    print(f"估计的平均干预效应 (ATE): {estimate.value}")

    # 6. 反驳验证
    print("\n--- 正在进行反驳验证 (Refutation) ---")
    refute_random = model.refute_estimate(
        identified_estimand, estimate, method_name="random_common_cause"
    )
    print(refute_random)

    return model, estimate

