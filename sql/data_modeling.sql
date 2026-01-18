-- 1. 创建并生成 机场维度表 (dim_airports)
-- 使用 CAST 显式转换类型，彻底解决 TEXT 字段主键冲突
CREATE TABLE IF NOT EXISTS dim_airports (
    airport_code VARCHAR(10) PRIMARY KEY,
    city_name VARCHAR(100),
    state_code VARCHAR(10)
) AS
SELECT DISTINCT
    CAST(airport_code AS CHAR(10)) AS airport_code,
    CAST(city_name AS CHAR(100)) AS city_name,
    CAST(state_code AS CHAR(10)) AS state_code
FROM (
    SELECT Origin AS airport_code, OriginCityName AS city_name, OriginState AS state_code FROM fact_flights_q1_raw
    UNION
    SELECT Dest AS airport_code, DestCityName AS city_name, DestState AS state_code FROM fact_flights_q1_raw
) AS combined_airports;


-- 2. 创建并生成 航司维度表 (dim_airlines)
-- 显式定义字段类型以支持主键约束
CREATE TABLE IF NOT EXISTS dim_airlines (
    airline_code VARCHAR(10) PRIMARY KEY
) AS
SELECT DISTINCT
    CAST(Reporting_Airline AS CHAR(10)) AS airline_code
FROM fact_flights_q1_raw;


-- 3. 创建并生成 日历维度表 (dim_calendar)
CREATE TABLE IF NOT EXISTS dim_calendar (
    date_key DATE PRIMARY KEY,
    year INT,
    quarter INT,
    month INT,
    day_of_month INT,
    day_of_week INT,
    is_weekend INT
) AS
SELECT DISTINCT
    FlightDate AS date_key,
    Year AS year,
    Quarter AS quarter,
    Month AS month,
    DayofMonth AS day_of_month,
    DayOfWeek AS day_of_week,
    Is_Weekend AS is_weekend
FROM fact_flights_q1_raw
ORDER BY FlightDate;


-- 4. 创建并生成 核心事实表 (fact_flights)
CREATE TABLE IF NOT EXISTS fact_flights AS
SELECT
    FlightDate AS date_key,
    CAST(Reporting_Airline AS CHAR(10)) AS airline_code,
    CAST(Tail_Number AS CHAR(20)) AS Tail_Number,
    Flight_Number_Reporting_Airline AS flight_number,
    CAST(Origin AS CHAR(10)) AS origin_airport,
    CAST(Dest AS CHAR(10)) AS dest_airport,
    CRSDepTime,
    DepTime,
    DepDelay,
    DepDelayMinutes,
    DepDel15,
    ActualElapsedTime,
    AirTime,
    Distance,
    CarrierDelay,
    WeatherDelay,
    NASDelay,
    SecurityDelay,
    LateAircraftDelay,
    Is_Cancelled
FROM fact_flights_q1_raw;

-- 为事实表添加索引以优化后续关联查询
-- 字段现在已转换为定长字符类型，索引将正常工作
CREATE INDEX idx_flight_date ON fact_flights(date_key);
CREATE INDEX idx_airline ON fact_flights(airline_code);
CREATE INDEX idx_origin ON fact_flights(origin_airport);


-- 5. (可选) 验证建模结果
SELECT 'fact_flights_q1_raw' as table_name, COUNT(*) as row_count FROM fact_flights_q1_raw
UNION ALL
SELECT 'fact_flights' as table_name, COUNT(*) as row_count FROM fact_flights;