import os
import sys

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据目录配置
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
DATABASE_DIR = os.path.join(DATA_DIR, 'database')
INDEX_DIR = os.path.join(DATA_DIR, 'index')

# 数据库配置
DATABASE_PATH = os.path.join(DATABASE_DIR, 'advertisements.db')

# 模型配置
CLIP_MODEL_NAME = "ViT-B-16"  # 可选: "ViT-L-14", "ViT-H-14", "RN50"
EMBEDDING_DIM = 512  # ViT-B-16的向量维度

# Faiss索引配置
IMAGE_INDEX_PATH = os.path.join(INDEX_DIR, 'image_embeddings.index')
TEXT_INDEX_PATH = os.path.join(INDEX_DIR, 'text_embeddings.index')

# 图片下载配置
MAX_DOWNLOAD_WORKERS = 10
DOWNLOAD_TIMEOUT = 30
MAX_RETRIES = 3
IMAGE_SIZE = (224, 224)  # 统一图片尺寸

# 搜索配置
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# Streamlit配置
PAGE_TITLE = "创意广告图文搜索系统"
PAGE_ICON = "🔍"
LAYOUT = "wide"

# 日志配置 (使用loguru)
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"

def setup_logging():
    """配置loguru日志系统"""
    from loguru import logger
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出处理器（不保存文件）
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        colorize=True
    )
    
    return logger

# 创建必要的目录
def create_directories():
    """创建项目所需的目录结构"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
        IMAGE_DIR, DATABASE_DIR, INDEX_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
if __name__ == "__main__":
    create_directories()
    logger = setup_logging()
    logger.info("项目目录结构创建完成！")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"数据目录: {DATA_DIR}")
    logger.info(f"图片目录: {IMAGE_DIR}")
    logger.info(f"索引目录: {INDEX_DIR}") 