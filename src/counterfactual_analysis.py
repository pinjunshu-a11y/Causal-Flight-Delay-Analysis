import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
import os
import warnings
from scipy import stats

# 忽略 Sklearn 的特征名称警告
warnings.filterwarnings('ignore', category=UserWarning)


# --- 解决中文字体乱码问题 ---
def set_chinese_font():
    font_path = './simsun.ttc'
    if os.path.exists(font_path):
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        print(f"已加载指定字体: {font_path}")
    else:
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
        for f in fonts:
            try:
                matplotlib.rcParams['font.family'] = f
                break
            except:
                continue
    matplotlib.rcParams['axes.unicode_minus'] = False


set_chinese_font()

# 1. 数据库连接设置
# 数据库配置信息
DB_USER = "root"
DB_PASSWORD = "000000"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "database"

# 构造连接字符串
DB_CONFIG = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_CONFIG)


def perform_counterfactual_analysis():
    print("正在加载数据用于因果响应建模...")
    df = pd.read_sql("SELECT * FROM fact_flights_causal_base LIMIT 300000", engine)

    # 2. 构建 T-Learner 基础特征
    features = [
        'weather_impact_proxy', 'congestion_log', 'prev_flight_delay',
        'distance_group', 'is_peak_hour', 'dep_hour'
    ]

    df_0 = df[df['is_tight_turnaround'] == 0]
    df_1 = df[df['is_tight_turnaround'] == 1]

    print(f"训练 T-Learner 基础模型: 控制组(T=0): {len(df_0)} | 处理组(T=1): {len(df_1)}")

    # 3. 引入 Bootstrap 以处理不确定性 (Addressing User Concerns)
    n_iterations = 15  # 轮数，生产环境建议 50+
    print(f"正在通过 Bootstrap ({n_iterations}轮采样) 估算置信区间...")

    # 预筛选用于个体分析的样本 (选取典型的短周转航班)
    sample_treated = df[df['is_tight_turnaround'] == 1].head(10).copy()
    individual_preds = np.zeros((n_iterations, len(sample_treated)))

    for i in range(n_iterations):
        # 对控制组进行有放回采样，捕捉模型不确定性
        df_0_boot = df_0.sample(frac=1.0, replace=True, random_state=i)
        m0_boot = RandomForestRegressor(n_estimators=30, max_depth=5, n_jobs=-1, random_state=i)
        m0_boot.fit(df_0_boot[features], df_0_boot['delay_level'])

        # 记录每轮对这些特定航班的反事实 Y(0) 预测
        individual_preds[i, :] = m0_boot.predict(sample_treated[features])

    # 4. 个体反事实推演与统计显著性判定
    print("\n--- 基于 T-Learner & Bootstrap 的个体反事实推演 ---")

    # 计算 ITE = 实际观测值 - 反事实均值预测
    obs_y_values = sample_treated['delay_level'].values
    ite_means = obs_y_values - individual_preds.mean(axis=0)
    ite_std = individual_preds.std(axis=0)

    # 计算 95% 置信区间
    ci_lower = ite_means - 1.96 * ite_std
    ci_upper = ite_means + 1.96 * ite_std

    results = []
    for idx, (_, row) in enumerate(sample_treated.iterrows()):
        mean_ite = ite_means[idx]

        # 统计显著性：如果 95% CI 不跨越 0，则该效应在统计上是显著的
        is_significant = "✅ 显著" if (ci_lower[idx] > 0 or ci_upper[idx] < 0) else "❌ 不显著"

        results.append({
            '实际延误': round(obs_y_values[idx], 2),
            'ITE均值': round(mean_ite, 2),
            '95% CI下界': round(ci_lower[idx], 2),
            '95% CI上界': round(ci_upper[idx], 2),
            '显著性': is_significant
        })

    print(pd.DataFrame(results).to_string(index=False))

    # 5. 全局政策模拟
    print("\n--- 全局政策模拟与逻辑校验 ---")
    m0_final = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    m0_final.fit(df_0[features], df_0['delay_level'])

    actual_mean = df['delay_level'].mean()
    df_policy = df.copy()
    mask_treated = df_policy['is_tight_turnaround'] == 1

    df_policy.loc[mask_treated, 'predicted_delay'] = m0_final.predict(df_policy.loc[mask_treated, features])
    df_policy.loc[~mask_treated, 'predicted_delay'] = df_policy.loc[~mask_treated, 'delay_level']

    policy_mean = df_policy['predicted_delay'].mean()
    improvement_pct = (actual_mean - policy_mean) / actual_mean * 100

    print(f"当前全网延误指数: {actual_mean:.4f}")
    print(f"预期政策收益: 延误水平降低 {improvement_pct:.2f}%")

    # 6. 可视化：ITE 置信区间森林图
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid", {"font.sans-serif": plt.rcParams['font.family']})

    y_axis = range(len(results))
    plt.errorbar(ite_means, y_axis, xerr=1.96 * ite_std, fmt='o', color='crimson',
                 ecolor='lightcoral', elinewidth=3, capsize=5, label='ITE 95% CI')

    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
    plt.yticks(y_axis, [f"航班样本 {i + 1}" for i in y_axis])
    plt.title('个体因果效应 (ITE) 统计显著性校验', fontsize=14)
    plt.xlabel('延误等级增量 (效应值 > 0 表示短周转显著增加了延误)', fontsize=12)
    plt.gca().invert_yaxis()  # 反转 Y 轴，使样本 1 在上方
    plt.legend()

    plt.tight_layout()
    plt.savefig('counterfactual_statistical_check.png')
    print("\n统计显著性图表已保存: counterfactual_statistical_check.png")


if __name__ == "__main__":
    try:
        perform_counterfactual_analysis()
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback

        traceback.print_exc()