import pandas as pd
from typing import List, Dict
import json
import re
import sqlite3

class DataProcessor:
    """
    数据处理器，负责从原始数据源提取、清洗和转换数据。
    """
    def __init__(self, db_path: str):
        """
        初始化DataProcessor。
        
        Args:
            db_path (str): SQLite数据库文件路径。
        """
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        """
        创建数据库表结构 (advertisements 和 images)。
        如果表已存在，则不会重复创建。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS advertisements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            url TEXT,
            title TEXT,
            background TEXT,
            insight TEXT,
            creative TEXT,
            result TEXT,
            score REAL,
            favorites INTEGER,
            comments INTEGER,
            publish_time TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ad_id INTEGER,
            image_url TEXT,
            local_path TEXT,
            embedding_id INTEGER,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            download_status TEXT,
            FOREIGN KEY (ad_id) REFERENCES advertisements (id)
        )
        ''')
        
        conn.commit()
        conn.close()

    def parse_csv(self, csv_path: str) -> List[Dict]:
        """
        解析CSV文件，提取并清洗广告数据。

        Args:
            csv_path (str): CSV文件路径。

        Returns:
            List[Dict]: 清洗后的数据列表，每个字典代表一条广告。
        """
        df = pd.read_csv(csv_path)
        
        # 定义字段映射关系
        column_mapping = {
            '创意广告名字': 'name',
            '广告URL': 'url',
            '参赛说明标题': 'title',
            '背景与目标内容': 'background',
            '洞察与策略内容': 'insight',
            '创意阐述内容': 'creative',
            '结果与影响内容': 'result',
            '应数评分': 'score',
            '收藏': 'favorites',
            '评论': 'comments',
            '发布时间': 'publish_time',
            '参赛类别': 'category',
            '创意图片地址': 'images_str'
        }
        
        df = df[list(column_mapping.keys())].rename(columns=column_mapping)
        
        # 数据清洗
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        df['favorites'] = pd.to_numeric(df['favorites'], errors='coerce').fillna(0).astype(int)
        df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0).astype(int)
        
        # 提取图片URL
        df['image_urls'] = df['images_str'].apply(self.extract_image_urls)
        
        # 转换为字典列表
        clean_data = df.drop(columns=['images_str']).to_dict('records')
        
        return clean_data

    def extract_image_urls(self, url_string: str) -> List[str]:
        """
        从包含多个URL的字符串中提取所有图片URL。
        URL可能由换行符、逗号或分号分隔。

        Args:
            url_string (str): 包含图片URL的原始字符串。

        Returns:
            List[str]: 提取出的URL列表。
        """
        if not isinstance(url_string, str):
            return []
        
        # 使用正则表达式匹配URL，更健壮
        urls = re.findall(r'https?://[^\s,;]+', url_string)
        return urls

    def save_to_database(self, data: List[Dict]):
        """
        将数据批量保存到SQLite数据库中。

        Args:
            data (List[Dict]): 从Excel中解析出的数据列表。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for record in data:
            ad_data = {k: v for k, v in record.items() if k != 'image_urls'}
            
            # 插入广告数据
            cursor.execute('''
            INSERT INTO advertisements (name, url, title, background, insight, creative, result, score, favorites, comments, publish_time, category)
            VALUES (:name, :url, :title, :background, :insight, :creative, :result, :score, :favorites, :comments, :publish_time, :category)
            ''', ad_data)
            
            ad_id = cursor.lastrowid
            
            # 插入图片URL
            if 'image_urls' in record and record['image_urls']:
                for img_url in record['image_urls']:
                    cursor.execute('''
                    INSERT INTO images (ad_id, image_url, download_status)
                    VALUES (?, ?, ?)
                    ''', (ad_id, img_url, 'pending'))

        conn.commit()
        conn.close() 