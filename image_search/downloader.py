import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import requests
from PIL import Image
from loguru import logger
from tqdm import tqdm

class Downloader:
    """
    负责从数据库中读取图片URL，下载图片，并更新数据库记录。
    """
    def __init__(self, db_path: str, image_dir: str, max_workers: int = 10, timeout: int = 30, max_retries: int = 3):
        """
        初始化Downloader。

        Args:
            db_path (str): SQLite数据库路径。
            image_dir (str): 图片存储目录。
            max_workers (int): 下载线程池的最大线程数。
            timeout (int): 下载请求超时时间。
            max_retries (int): 下载失败最大重试次数。
        """
        self.db_path = db_path
        self.image_dir = image_dir
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        os.makedirs(self.image_dir, exist_ok=True)

    def get_pending_images(self) -> List[Tuple]:
        """从数据库获取所有待下载的图片记录。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, ad_id, image_url FROM images WHERE download_status = 'pending'")
            return cursor.fetchall()

    def download_image_task(self, image_data: Tuple):
        """
        单个图片的下载任务，包含重试逻辑。
        
        Args:
            image_data (Tuple): 包含 (id, ad_id, image_url) 的元组。
        """
        image_id, ad_id, image_url = image_data
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(image_url, timeout=self.timeout)
                response.raise_for_status()
                
                # 生成本地文件路径
                file_extension = os.path.splitext(image_url)[1] or '.jpg'
                local_path = os.path.join(self.image_dir, f"{ad_id}_{image_id}{file_extension}")
                
                # 保存图片文件
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                # 获取图片信息
                with Image.open(local_path) as img:
                    width, height = img.size
                file_size = os.path.getsize(local_path)
                
                # 更新数据库记录
                self.update_image_record(image_id, 'completed', local_path, width, height, file_size)
                return
                
            except requests.RequestException as e:
                logger.warning(f"下载失败 (第 {attempt + 1} 次): {image_url}, 错误: {e}")
                time.sleep(2 ** attempt) # 指数退避
        
        # 所有重试失败后
        logger.error(f"下载失败，已达最大重试次数: {image_url}")
        self.update_image_record(image_id, 'failed')

    def update_image_record(self, image_id: int, status: str, local_path: str = None, width: int = None, height: int = None, file_size: int = None):
        """更新数据库中的图片记录。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE images SET download_status=?, local_path=?, width=?, height=?, file_size=? WHERE id=?",
                (status, local_path, width, height, file_size, image_id)
            )
            conn.commit()

    def run(self):
        """
        执行图片下载的主流程。
        """
        pending_images = self.get_pending_images()
        if not pending_images:
            logger.info("没有待下载的图片。")
            return
            
        logger.info(f"发现 {len(pending_images)} 张待下载的图片，开始下载...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.download_image_task, pending_images), total=len(pending_images), desc="下载图片"))

        logger.info("图片下载流程完成。") 