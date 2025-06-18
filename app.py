import streamlit as st
import os
from PIL import Image
from typing import List, Dict, Any

# 将KMP_DUPLICATE_LIB_OK设置为TRUE，以避免OMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from image_search.embedding_generator import EmbeddingGenerator
from image_search.search_engine import SearchEngine
from config import (
    DATABASE_PATH,
    IMAGE_INDEX_PATH,
    TEXT_INDEX_PATH,
    CLIP_MODEL_NAME,
    EMBEDDING_DIM,
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    setup_logging
)

# --- 页面配置 (必须是第一个Streamlit命令) ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# --- 初始化 ---
setup_logging()

@st.cache_resource
def get_embedder():
    """缓存EmbeddingGenerator实例"""
    return EmbeddingGenerator(model_name=CLIP_MODEL_NAME)

@st.cache_resource
def get_search_engine(_embedder):
    """缓存SearchEngine实例"""
    return SearchEngine(
        embedding_generator=_embedder,
        embedding_dim=EMBEDDING_DIM,
        db_path=DATABASE_PATH,
        image_index_path=IMAGE_INDEX_PATH,
        text_index_path=TEXT_INDEX_PATH
    )

embedder = get_embedder()
search_engine = get_search_engine(embedder)

# --- 页面布局 ---
st.title("🎨 " + PAGE_TITLE)

tab1, tab2, tab3, tab4 = st.tabs(["文搜图", "图搜图", "图搜文", "文搜文"])

def display_results(results: List[Dict[str, Any]]):
    """以卡片形式展示广告案例结果"""
    if not results:
        st.info("未找到匹配的结果。")
        return
    
    st.subheader(f"为你找到 {len(results)} 个相关的广告案例")
    st.markdown("---")

    for result in results:
        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                if result.get("representative_image"):
                    st.image(result["representative_image"], use_container_width=True, caption="最佳匹配图片")
                else:
                    st.info("该案例无图片")
            with col2:
                st.markdown(f"#### {result.get('title', '无标题')}")
                st.markdown(f"**广告ID**: `{result['ad_id']}` | **最高匹配分**: `{result['score']:.2f}`")
                st.caption(result.get('text', '无详细描述')[:300] + "...")
            
            # 为其他图片创建可展开区域
            other_images = result.get("other_images", [])
            if other_images:
                with st.expander(f"查看其余 {len(other_images)} 张图片"):
                    # 定义网格布局来展示其他图片
                    expander_cols = st.columns(4)
                    for i, img_path in enumerate(other_images):
                        if os.path.exists(img_path):
                            expander_cols[i % 4].image(img_path, use_container_width=True)

        st.write("") # 添加垂直间距


# --- 各搜索模式的实现 ---

with tab1: # 文搜图
    st.header("🖼️ 文搜图")
    text_query_img = st.text_input("输入文本描述", key="text_to_image_input")
    if st.button("搜索", key="text_to_image_button"):
        if text_query_img:
            with st.spinner("正在搜索..."):
                search_results = search_engine.text_to_image_search(text_query_img)
                display_results(search_results)
        else:
            st.warning("请输入文本描述。")

with tab2: # 图搜图
    st.header("🖼️ 图搜图")
    uploaded_file_img = st.file_uploader("上传图片进行搜索", type=['jpg', 'jpeg', 'png', 'gif'], key="image_to_image_uploader")
    if uploaded_file_img is not None:
        st.image(uploaded_file_img, caption="您上传的图片", width=200)
        if st.button("开始搜索", key="image_to_image_button"):
            with st.spinner("正在搜索..."):
                search_results = search_engine.image_to_image_search(uploaded_file_img)
                display_results(search_results)

with tab3: # 图搜文
    st.header("📝 图搜文")
    uploaded_file_text = st.file_uploader("上传图片进行搜索", type=['jpg', 'jpeg', 'png', 'gif'], key="image_to_text_uploader")
    if uploaded_file_text is not None:
        st.image(uploaded_file_text, caption="您上传的图片", width=200)
        if st.button("开始搜索", key="image_to_text_button"):
            with st.spinner("正在搜索..."):
                search_results = search_engine.image_to_text_search(uploaded_file_text)
                display_results(search_results)

with tab4: # 文搜文
    st.header("📝 文搜文")
    text_query_text = st.text_input("输入关键词", key="text_to_text_input")
    if st.button("搜索", key="text_to_text_button"):
        if text_query_text:
            with st.spinner("正在搜索..."):
                search_results = search_engine.text_to_text_search(text_query_text)
                display_results(search_results)
        else:
            st.warning("请输入关键词。") 