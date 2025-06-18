# 创意广告图文搜索系统

基于Chinese-CLIP的多模态创意广告搜索引擎，支持图搜图、文搜图、图搜文、文搜文四种搜索模式。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 创建项目目录
```bash
python config.py
```

### 3. 启动应用
```bash
streamlit run app.py
```

## 📁 项目结构

```
./                                  # 项目根目录
├── app.py                          # Streamlit应用入口
├── requirements.txt                # 依赖包
├── config.py                       # 配置文件
├── prd.md                         # 产品需求文档
├── README.md                       # 本文件
├── image_search/                   # 核心代码包
│   ├── __init__.py                 # Python包初始化
│   ├── data_processor.py           # 数据处理
│   ├── embedding_generator.py      # 向量生成
│   ├── search_engine.py            # 搜索引擎
│   └── utils.py                    # 工具函数
├── ui/                             # Web界面
│   └── streamlit_app.py            # Streamlit应用
├── scripts/                        # 工具脚本
│   ├── setup_database.py           # 数据库初始化
│   ├── download_images.py          # 图片下载脚本
│   └── build_index.py              # 索引构建脚本
└── data/                           # 数据存储目录
    ├── raw/                        # 原始Excel数据
    ├── processed/                  # 处理后的数据
    ├── images/                     # 下载的图片
    ├── database/                   # SQLite数据库
    └── index/                      # Faiss向量索引
```

## 🎯 核心功能

- **图搜图**：上传图片，找到视觉相似的创意广告
- **文搜图**：输入文本描述，找到匹配的创意广告图片  
- **图搜文**：上传图片，找到相关的文案内容
- **文搜文**：输入关键词，找到相似的文案内容

## 📊 技术栈

- **🤖 Chinese-CLIP**: 多模态预训练模型
- **🔍 Faiss**: 向量相似性搜索引擎
- **🗄️ SQLite**: 轻量级数据库
- **🌐 Streamlit**: Web应用框架
- **📝 Loguru**: 日志系统（控制台输出）

## 📖 详细文档

详细的产品需求文档和技术说明请查看：[prd.md](prd.md)

## 🔧 开发说明

1. **日志系统**：使用loguru，所有日志输出到控制台，不保存文件
2. **数据存储**：所有数据文件统一存放在`./data/`目录下
3. **应用入口**：通过`streamlit run app.py`启动应用
4. **标准结构**：采用标准Python项目结构，重要文件在根目录

## 🚨 注意事项

- 首次运行需要下载Chinese-CLIP模型，请确保网络连接稳定
- 建议至少8GB RAM用于处理大批量图片数据
- 图片下载可能需要较长时间，取决于数据量和网络状况

## 📞 技术支持

如有问题请查看：
1. [产品需求文档](prd.md)
2. 控制台日志输出
3. 项目issue跟踪# CIG-image-rag
