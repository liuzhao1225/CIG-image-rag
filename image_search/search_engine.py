import faiss
import numpy as np
import os
import sqlite3
from loguru import logger
from typing import List, Dict, Any

from .embedding_generator import EmbeddingGenerator

class SearchEngine:
    """
    负责管理Faiss向量索引，包括构建、保存、加载和搜索。
    """
    def __init__(self, embedding_generator: EmbeddingGenerator, embedding_dim: int, db_path: str, image_index_path: str, text_index_path: str):
        """
        初始化SearchEngine。

        Args:
            embedding_generator (EmbeddingGenerator): 向量生成器实例。
            embedding_dim (int): 向量维度。
            db_path (str): SQLite数据库路径。
            image_index_path (str): 图片Faiss索引文件路径。
            text_index_path (str): 文本Faiss索引文件路径。
        """
        self.embedder = embedding_generator
        self.db_path = db_path
        self.image_index_path = image_index_path
        self.text_index_path = text_index_path
        self.embedding_dim = embedding_dim

        # 初始化空的索引
        self.image_index = None
        self.text_index = None
        
        # 加载索引
        self.load_indexes()

    def build_index(self, embeddings: np.ndarray, index_type: str):
        """
        使用给定的向量构建或更新一个Faiss索引。

        Args:
            embeddings (np.ndarray): 用于构建索引的向量数组。
            index_type (str): 'image' 或 'text'，指定要构建的索引类型。
        """
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings)

        if index_type == 'image':
            self.image_index = index
            logger.info(f"图片索引构建完成，共添加 {self.image_index.ntotal} 个向量。")
        elif index_type == 'text':
            self.text_index = index
            logger.info(f"文本索引构建完成，共添加 {self.text_index.ntotal} 个向量。")
        else:
            raise ValueError("index_type必须是 'image' 或 'text'")

    def save_indexes(self):
        """将当前的图片和文本索引保存到文件。"""
        os.makedirs(os.path.dirname(self.image_index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.text_index_path), exist_ok=True)
        
        logger.info(f"正在保存图片索引到: {self.image_index_path}")
        faiss.write_index(self.image_index, self.image_index_path)
        
        logger.info(f"正在保存文本索引到: {self.text_index_path}")
        faiss.write_index(self.text_index, self.text_index_path)
        logger.info("索引保存完成。")

    def load_indexes(self):
        """从文件加载图片和文本索引。"""
        if os.path.exists(self.image_index_path):
            logger.info(f"正在从 {self.image_index_path} 加载图片索引...")
            self.image_index = faiss.read_index(self.image_index_path)
            logger.info(f"图片索引加载完成，包含 {self.image_index.ntotal} 个向量。")
        else:
            logger.warning(f"图片索引文件未找到: {self.image_index_path}")
            
        if os.path.exists(self.text_index_path):
            logger.info(f"正在从 {self.text_index_path} 加载文本索引...")
            self.text_index = faiss.read_index(self.text_index_path)
            logger.info(f"文本索引加载完成，包含 {self.text_index.ntotal} 个向量。")
        else:
            logger.warning(f"文本索引文件未找到: {self.text_index_path}")
    
    def _search(self, index, query_embedding, top_k):
        """通用搜索函数"""
        if index is None:
            logger.error("索引未加载，无法执行搜索。")
            return np.array([]), np.array([])
        return index.search(query_embedding, top_k)

    def _get_image_details_for_ads(self, ad_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        if not ad_ids: return {}
        image_details_map = {ad_id: [] for ad_id in ad_ids}
        placeholders = ','.join('?' for _ in ad_ids)
        query = f"SELECT ad_id, local_path, embedding_id FROM images WHERE ad_id IN ({placeholders}) AND download_status = 'completed' AND embedding_id IS NOT NULL"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(ad_ids))
            for ad_id, path, emb_id in cursor.fetchall():
                image_details_map[ad_id].append({"path": path, "embedding_id": emb_id})
        return image_details_map

    def _fetch_text_ad_results(self, ad_ids: List[int], scores_map: Dict[int, float]) -> List[Dict[str, Any]]:
        if not ad_ids: return []
        placeholders = ','.join('?' for _ in ad_ids)
        ads_query = f"SELECT id, title, creative FROM advertisements WHERE id IN ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(ads_query, tuple(ad_ids))
            ad_data_map = {row[0]: {'title': row[1], 'text': row[2]} for row in cursor.fetchall()}
        return [{
            "ad_id": ad_id, "score": scores_map.get(ad_id, 0),
            "title": ad_data_map[ad_id]['title'], "text": ad_data_map[ad_id]['text']
        } for ad_id in ad_ids if ad_id in ad_data_map]

    def _finalize_results(self, ad_ids, ad_scores, query_embedding):
        results_with_text = self._fetch_text_ad_results(ad_ids, ad_scores)
        image_details_map = self._get_image_details_for_ads(ad_ids)
        for result in results_with_text:
            ad_id = result["ad_id"]
            ad_images = image_details_map.get(ad_id, [])
            if not ad_images:
                result["representative_image"] = None
                result["other_images"] = []
                continue
            emb_ids = [img['embedding_id'] for img in ad_images]
            image_embeddings = np.array([self.image_index.reconstruct(int(eid)) for eid in emb_ids])
            similarities = np.dot(image_embeddings, query_embedding.flatten())
            best_image_idx = np.argmax(similarities)
            result["representative_image"] = ad_images[best_image_idx]["path"]
            result["other_images"] = [img["path"] for i, img in enumerate(ad_images) if i != best_image_idx]
        return results_with_text

    def _search_image_index_and_process(self, query_embedding, top_k):
        distances, ids = self._search(self.image_index, query_embedding, top_k * 5)
        if not ids.size: return [], {}
        embedding_ids = [int(i) for i in ids[0]]
        placeholders = ','.join('?' for _ in embedding_ids)
        query = f"SELECT embedding_id, ad_id FROM images WHERE embedding_id IN ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(embedding_ids))
            image_to_ad_map = dict(cursor.fetchall())
        ad_scores = {}
        for emb_id, score in zip(embedding_ids, distances[0]):
            ad_id = image_to_ad_map.get(emb_id)
            if ad_id and (ad_id not in ad_scores or score > ad_scores[ad_id]):
                ad_scores[ad_id] = score
        sorted_ad_ids = sorted(ad_scores, key=ad_scores.get, reverse=True)
        return sorted_ad_ids[:top_k], ad_scores

    def _search_text_index_and_process(self, query_embedding, top_k):
        distances, ids = self._search(self.text_index, query_embedding, top_k)
        if not ids.size: return [], {}
        ad_ids = [int(i) + 1 for i in ids[0]]
        ad_scores = {int(i) + 1: s for i, s in zip(ids[0], distances[0])}
        return ad_ids, ad_scores

    def image_to_image_search(self, image_path: str, top_k: int = 10):
        query_embedding = self.embedder.encode_image(image_path)
        if query_embedding is None: return []
        query_embedding_np = query_embedding.cpu().numpy()
        top_ad_ids, ad_scores = self._search_image_index_and_process(query_embedding_np, top_k)
        return self._finalize_results(top_ad_ids, ad_scores, query_embedding_np)

    def text_to_image_search(self, text: str, top_k: int = 10):
        query_embedding = self.embedder.encode_text(text)
        if query_embedding is None: return []
        query_embedding_np = query_embedding.cpu().numpy()
        top_ad_ids, ad_scores = self._search_image_index_and_process(query_embedding_np, top_k)
        return self._finalize_results(top_ad_ids, ad_scores, query_embedding_np)

    def image_to_text_search(self, image_path: str, top_k: int = 10):
        query_embedding = self.embedder.encode_image(image_path)
        if query_embedding is None: return []
        query_embedding_np = query_embedding.cpu().numpy()
        top_ad_ids, ad_scores = self._search_text_index_and_process(query_embedding_np, top_k)
        return self._finalize_results(top_ad_ids, ad_scores, query_embedding_np)

    def text_to_text_search(self, text: str, top_k: int = 10):
        query_embedding = self.embedder.encode_text(text)
        if query_embedding is None: return []
        query_embedding_np = query_embedding.cpu().numpy()
        top_ad_ids, ad_scores = self._search_text_index_and_process(query_embedding_np, top_k)
        return self._finalize_results(top_ad_ids, ad_scores, query_embedding_np) 