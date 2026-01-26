import pandas as pd
import glob
import os
from sqlalchemy import create_engine

# --- 解决 FutureWarning ---
# 显式开启新行为，以避免 fillna 时的向下转型警告
pd.set_option('future.no_silent_downcasting', True)


def clean_airline_data(file_path):
    print(f"正在处理文件: {file_path}...")

    # 定义我们需要保留的核心字段，减少内存占用并提高处理速度
    keep_cols = [
        'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
        'Reporting_Airline', 'Tail_Number', 'Flight_Number_Reporting_Airline',
        'Origin', 'OriginCityName', 'OriginState', 'Dest', 'DestCityName', 'DestState',
        'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes', 'DepDel15', 'DepTimeBlk',
        'ActualElapsedTime', 'AirTime', 'Distance',
        'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'
    ]

    # 1. 加载数据，只读取指定列
    df = pd.read_csv(file_path, low_memory=False, usecols=lambda x: x in keep_cols or x == 'Cancelled')

    # 2. 确保 FlightDate 是日期格式
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])

    # 3. 处理延误原因字段
    delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
    for col in delay_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).infer_objects()

    # 4. 优化后的时间格式转换函数
    def format_time(time_series):
        # 显式处理 fillna 并转换为 int，避免警告
        time_series = time_series.fillna(0).infer_objects().astype(int).astype(str).str.zfill(4)
        time_series = time_series.replace('2400', '0000')
        return time_series.str[:2] + ':' + time_series.str[2:]

    # 显式将时间列转换为 object 类型，防止将字符串赋值回 float64 列时报错
    df['CRSDepTime'] = df['CRSDepTime'].astype(object)
    df['DepTime'] = df['DepTime'].astype(object)

    # 转换计划起飞时间
    df['CRSDepTime'] = format_time(df['CRSDepTime'])

    # 转换实际起飞时间（仅针对非空值）
    mask_not_nan = df['DepTime'].notna()
    if mask_not_nan.any():
        df.loc[mask_not_nan, 'DepTime'] = format_time(df.loc[mask_not_nan, 'DepTime'])

    # 5. 处理缺失值与异常值
    df['DepDelay'] = df['DepDelay'].fillna(0).infer_objects()
    df['DepDelayMinutes'] = df['DepDelayMinutes'].fillna(0).infer_objects()
    df['DepDel15'] = df['DepDel15'].fillna(0).infer_objects()

    # 6. 标记取消航班
    if 'Cancelled' in df.columns:
        df['Is_Cancelled'] = df['Cancelled'].fillna(0).astype(int)
    else:
        df['Is_Cancelled'] = df['ActualElapsedTime'].isna().astype(int)

    return df[[c for c in df.columns if c != 'Cancelled']]


# --- 主程序：合并并存储至 MySQL ---

# 数据库配置信息
DB_USER = "root"
DB_PASSWORD = "000000"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "database"

# 构造连接字符串
db_config = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    engine = create_engine(db_config)
    print("数据库连接引擎创建成功！")
except Exception as e:
    print(f"数据库连接引擎创建失败: {e}")
    engine = None

# 文件路径设置
data_folder = './data'
all_files = sorted(glob.glob(os.path.join(data_folder, "2025_0[1-3].csv")))

if not all_files:
    all_files = sorted(glob.glob(os.path.join('../', "2025_0[1-3].csv")))

if not all_files:
    print("未找到匹配的 CSV 文件，请检查路径。")
else:
    for i, f in enumerate(all_files):
        try:
            df_cleaned = clean_airline_data(f)

            # 增加特征工程字段
            df_cleaned['Is_Weekend'] = df_cleaned['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)

            # 存入 MySQL
            if engine:
                print(f"正在将 {f} 的数据存入 MySQL (fact_flights_q1_raw)...")
                mode = 'replace' if i == 0 else 'append'

                df_cleaned.to_sql(
                    name='fact_flights_q1_raw',
                    con=engine,
                    if_exists=mode,
                    index=False,
                    chunksize=10000
                )
                print(f"文件 {f} 存储完成。")
            else:
                print("数据库引擎未就绪，无法存储。")
        except Exception as e:
            print(f"处理文件 {f} 时出错: {e}")

    print("\n--- 所有 Q1 数据已清洗并成功导入 MySQL 汇总表: fact_flights_q1_raw ---")