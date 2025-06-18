import os
import sys
import sqlite3
import numpy as np
from tqdm import tqdm
from loguru import logger

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_search.embedding_generator import EmbeddingGenerator
from image_search.search_engine import SearchEngine
from config import (
    DATABASE_PATH,
    IMAGE_INDEX_PATH,
    TEXT_INDEX_PATH,
    EMBEDDING_DIM,
    CLIP_MODEL_NAME,
    setup_logging
)

def fetch_data_from_db(db_path):
    """从数据库获取图片和文本数据。"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # 获取已下载的图片
        cursor.execute("SELECT id, local_path FROM images WHERE download_status = 'completed' AND local_path IS NOT NULL")
        images = cursor.fetchall()
        # 获取广告文本
        cursor.execute("SELECT id, title, background, insight, creative FROM advertisements")
        texts = cursor.fetchall()
    return images, texts

def update_embedding_id_in_db(db_path, image_id, embedding_id):
    """更新数据库中图片的embedding_id。"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE images SET embedding_id = ? WHERE id = ?", (embedding_id, image_id))
        conn.commit()

def main():
    """
    执行索引构建流程：
    1. 初始化日志、向量生成器和搜索引擎。
    2. 从数据库获取数据。
    3. 生成并构建图片索引。
    4. 生成并构建文本索引。
    5. 保存索引。
    """
    setup_logging()

    # 1. 初始化
    embedder = EmbeddingGenerator(model_name=CLIP_MODEL_NAME)
    search_engine = SearchEngine(EMBEDDING_DIM, IMAGE_INDEX_PATH, TEXT_INDEX_PATH)

    # 2. 获取数据
    logger.info("正在从数据库获取数据...")
    images, texts = fetch_data_from_db(DATABASE_PATH)
    logger.info(f"获取到 {len(images)} 张图片和 {len(texts)} 条文本。")

    # 3. 构建图片索引
    logger.info("开始构建图片索引...")
    image_embeddings = []
    current_embedding_id = 0
    for image_id, image_path in tqdm(images, desc="生成图片向量"):
        embedding = embedder.encode_image(image_path)
        if embedding is not None:
            image_embeddings.append(embedding.cpu().numpy().flatten())
            # 更新数据库
            update_embedding_id_in_db(DATABASE_PATH, image_id, current_embedding_id)
            current_embedding_id += 1
    
    if image_embeddings:
        search_engine.build_index(np.array(image_embeddings), 'image')

    # 4. 构建文本索引
    logger.info("开始构建文本索引...")
    text_embeddings = []
    for ad_id, title, background, insight, creative in tqdm(texts, desc="生成文本向量"):
        # 将多个文本字段合并为一个长文本
        full_text = ' '.join(filter(None, [title, background, insight, creative]))
        if full_text:
            embedding = embedder.encode_text(full_text)
            if embedding is not None:
                text_embeddings.append(embedding.cpu().numpy().flatten())

    if text_embeddings:
        search_engine.build_index(np.array(text_embeddings), 'text')

    # 5. 保存索引
    search_engine.save_indexes()
    
    logger.info("索引构建流程完成。")

if __name__ == "__main__":
    main() 