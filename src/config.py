import os
from typing import Dict, Any
from pathlib import Path

# 获取项目根目录 (airline_data_analysis_2025_q1)
BASE_DIR = Path(__file__).resolve().parent.parent

# 统一指向根目录下的 result_png
OUTPUT_DIR = BASE_DIR / "result_png"


class DatabaseConfig:
    """数据库配置类"""

    def __init__(self):
        # 从环境变量获取配置
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', 3306))
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', '000000')
        self.db = os.getenv('DB_NAME', 'FlightOps_2025_Q1')

    def get_connection_string(self) -> str:
        """获取数据库连接字符串"""
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    def get_engine_kwargs(self) -> Dict[str, Any]:
        """获取引擎参数"""
        # 移除 auth_plugin 参数，使用更兼容的方式
        return {
            'connect_args': {}
        }


# 全局配置实例
DB_CONFIG = DatabaseConfig()