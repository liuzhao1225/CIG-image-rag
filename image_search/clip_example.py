import torch 
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
import numpy as np
from loguru import logger
import time

# 加载模型
start_time = time.time()
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./models')
model.eval()
logger.info(f"Model loading time: {time.time() - start_time:.2f}s")

text_labels = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]    
# 加载图片和文本
start_time = time.time()
image = preprocess(Image.open("./examples/pokemon.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(text_labels).to(device)
logger.info(f"Data preprocessing time: {time.time() - start_time:.2f}s")

# 计算相似度
start_time = time.time()
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)    
    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
logger.info(f"Feature extraction and similarity computation time: {time.time() - start_time:.2f}s")

logger.info(f"Label probs: {probs}")

index = np.argmax(probs)
logger.info(f"最匹配的标签: {text_labels[index]}")