import sys

import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from src.utils.database_utils import get_database_engine

def build_causal_feature_set():
    print("正在从数据库读取核心数据...")

    # 使用统一的数据库连接
    engine = get_database_engine()

    # 移除 DepTimeBlk，改为读取 CRSDepTime 后在 Python 中处理小时级别拥堵
    query = """
    SELECT 
        date_key, airline_code, Tail_Number, origin_airport, 
        CRSDepTime, DepTime, DepDelayMinutes, CarrierDelay, WeatherDelay, 
        NASDelay, LateAircraftDelay, ActualElapsedTime, Distance, Is_Cancelled
    FROM fact_flights
    WHERE Is_Cancelled = 0
    """
    df = pd.read_sql(query, engine)

    print("正在进行深度因果特征工程...")

    # --- 1. 修复时间顺序与物理时序 ---
    # 构建计划起飞的完整时间戳
    df['planned_timestamp'] = pd.to_datetime(df['date_key'].astype(str) + ' ' + df['CRSDepTime'])
    df = df.sort_values(['Tail_Number', 'planned_timestamp'])

    # 提取计划起飞小时，用于后续拥堵度计算和混杂变量
    df['dep_hour'] = df['planned_timestamp'].dt.hour

    # --- 2. 改进飞机周转因果项 ---
    # 计算前序航班的预计到达时间
    df['prev_planned_arrival'] = df.groupby('Tail_Number')['planned_timestamp'].shift(1) + \
                                 pd.to_timedelta(df.groupby('Tail_Number')['ActualElapsedTime'].shift(1), unit='m')

    # 计划周转时间 (分钟)
    df['planned_turnaround_mins'] = (df['planned_timestamp'] - df['prev_planned_arrival']).dt.total_seconds() / 60
    # 标记短周转压力
    df['is_tight_turnaround'] = (df['planned_turnaround_mins'] < 45).astype(int)

    # 获取前序航班的真实延误
    df['prev_flight_delay'] = df.groupby('Tail_Number')['DepDelayMinutes'].shift(1).fillna(0)

    # --- 3. 精细化机场拥堵度 (按小时) ---
    # 使用日期 + 机场 + 小时作为颗粒度
    df['airport_hourly_congestion'] = df.groupby(['date_key', 'origin_airport', 'dep_hour'])['Tail_Number'].transform(
        'count')
    # 对数平滑处理
    df['congestion_log'] = np.log1p(df['airport_hourly_congestion'])

    # --- 4. 优化离散化方案 (三分类) ---
    def categorize_delay(minutes):
        if minutes <= 15: return 0  # 无延误
        if minutes <= 60: return 1  # 中等延误
        return 2  # 严重延误

    df['delay_level'] = df['DepDelayMinutes'].apply(categorize_delay)

    # 辅助因果变量：是否受天气影响
    df['weather_impact_proxy'] = (df['WeatherDelay'] > 0).astype(int)

    # --- 5. 补充混杂变量 (Confounders) ---
    # 是否高峰时段 (07-09, 16-19)
    df['is_peak_hour'] = df['dep_hour'].apply(lambda x: 1 if x in [7, 8, 9, 16, 17, 18, 19] else 0).astype(int)
    # 距离分箱
    df['distance_group'] = pd.qcut(df['Distance'], 4, labels=[1, 2, 3, 4]).astype(int)

    # --- 6. 存储结果 ---
    # 移除临时的 timestamp 对象
    df_to_save = df.drop(columns=['planned_timestamp', 'prev_planned_arrival'])

    print("正在将优化后的因果特征表写回 MySQL (fact_flights_causal_base)...")
    try:
        df_to_save.to_sql(
            name='fact_flights_causal_base',
            con=engine,
            if_exists='replace',
            index=False,
            chunksize=10000
        )
        print("优化后的因果特征表存储成功！")
    except Exception as e:
        print(f"存储失败: {e}")

    print("\n特征工程改进完成。")
    print(df[['Tail_Number', 'planned_turnaround_mins', 'is_tight_turnaround', 'delay_level']].head())

