import os
import sys

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_search.downloader import Downloader
from config import (
    DATABASE_PATH, 
    IMAGE_DIR, 
    MAX_DOWNLOAD_WORKERS, 
    DOWNLOAD_TIMEOUT, 
    MAX_RETRIES,
    setup_logging
)

def main():
    """
    执行图片下载流程：
    1. 配置日志。
    2. 初始化Downloader。
    3. 运行下载流程。
    """
    setup_logging()
    
    downloader = Downloader(
        db_path=DATABASE_PATH,
        image_dir=IMAGE_DIR,
        max_workers=MAX_DOWNLOAD_WORKERS,
        timeout=DOWNLOAD_TIMEOUT,
        max_retries=MAX_RETRIES
    )
    
    downloader.run()

if __name__ == "__main__":
    main()
