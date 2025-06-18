import os
import sys
import pprint

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_search.embedding_generator import EmbeddingGenerator
from image_search.search_engine import SearchEngine
from config import (
    DATABASE_PATH,
    IMAGE_INDEX_PATH,
    TEXT_INDEX_PATH,
    CLIP_MODEL_NAME,
    EMBEDDING_DIM,
    setup_logging
)

def main():
    """
    执行搜索功能测试：
    1. 初始化模块。
    2. 执行一次文搜图搜索。
    3. 打印结果。
    """
    setup_logging()

    # 1. 初始化
    embedder = EmbeddingGenerator(model_name=CLIP_MODEL_NAME)
    search_engine = SearchEngine(
        embedding_generator=embedder,
        embedding_dim=EMBEDDING_DIM,
        db_path=DATABASE_PATH,
        image_index_path=IMAGE_INDEX_PATH,
        text_index_path=TEXT_INDEX_PATH
    )

    # 2. 执行文搜图
    query_text = "一个男人在看手机"
    print(f"执行文搜图搜索，查询: '{query_text}'")
    
    results = search_engine.text_to_image_search(query_text, top_k=5)

    # 3. 打印结果
    print("\n搜索结果:")
    if results:
        pprint.pprint(results)
    else:
        print("没有找到匹配的结果。")

if __name__ == "__main__":
    main() 