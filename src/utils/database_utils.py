#!/usr/bin/env python3
"""
数据库连接工具模块
"""
from sqlalchemy import create_engine, text  # <--- 1. 导入 text
from sqlalchemy.engine import Engine
from src.config import DB_CONFIG

def get_database_engine() -> Engine:
    """
    获取数据库引擎实例
    """
    connection_string = DB_CONFIG.get_connection_string()
    engine_kwargs = DB_CONFIG.get_engine_kwargs()

    try:
        engine = create_engine(connection_string, **engine_kwargs)
        # 测试连接
        with engine.connect() as conn:
            # <--- 2. 使用 text() 包装字符串
            conn.execute(text("SELECT 1"))
            conn.commit() # 建议加上 commit
        print(f"[成功] 数据库连接正常: {DB_CONFIG.host}:{DB_CONFIG.port}")
        return engine
    except Exception as e:
        print(f"[错误] 数据库连接失败: {e}")
        raise

def execute_query(query: str, params=None):
    """
    执行SQL查询
    """
    engine = get_database_engine()
    with engine.connect() as conn:
        # <--- 3. 这里的 query 也需要用 text() 包装
        if params:
            result = conn.execute(text(query), params)
        else:
            result = conn.execute(text(query))
        return result.fetchall()