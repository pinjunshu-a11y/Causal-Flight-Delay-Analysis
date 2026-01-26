#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 2. 设置日志（路径调整到根目录下的 src/causal_analysis.log）
log_path = project_root / "src" / "causal_analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== 开始执行因果分析全流程 ===")

    # 步骤1: 数据准备与特征工程
    logger.info("\n=== 步骤1: 数据准备与特征工程 ===")
    from src.modules import build_causal_dataset
    build_causal_dataset.build_causal_feature_set()

    # 步骤2: 构建结构因果模型与DAG验证
    logger.info("\n=== 步骤2: 构建结构因果模型与DAG验证 ===")
    from src.modules import causal_modeling_dag
    causal_modeling_dag.build_scm_model()

    # 步骤3: 因果效应估计 (Do-Calculus)
    logger.info("\n=== 步骤3: 因果效应估计 (Do-Calculus) ===")
    from src.modules import do_calculus
    do_calculus.perform_do_calculus()

    # 步骤4: 反事实分析
    logger.info("\n=== 步骤4: 反事实分析 ===")
    from src.modules import counterfactual_analysis
    counterfactual_analysis.perform_counterfactual_analysis()

    logger.info("\n=== 所有分析步骤已完成 ===")
    logger.info("因果分析流程执行成功！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("主流程执行失败")
        sys.exit(1)