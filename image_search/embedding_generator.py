import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from PIL import Image
from typing import List
from loguru import logger

class EmbeddingGenerator:
    """
    负责加载Chinese-CLIP模型并生成图片和文本的向量。
    """
    def __init__(self, model_name: str = "ViT-B-16"):
        """
        初始化EmbeddingGenerator。

        Args:
            model_name (str): 要使用的Chinese-CLIP模型名称。
            device (str): 运行模型的设备 ('cuda', 'mps' or 'cpu')。
        """
        # if device == "cuda" and not torch.cuda.is_available():
        #     if torch.backends.mps.is_available():
        #         logger.warning("CUDA不可用，自动切换到MPS。")
        #         device = "mps"
        #     else:
        #         logger.warning("CUDA和MPS都不可用，自动切换到CPU。")
        #         device = "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"正在加载模型 '{model_name}' 到设备 '{self.device}'...")
        
        # 加载模型和预处理器
        self.model, self.preprocess = load_from_name(
            model_name, 
            device=self.device, 
            download_root='./models'
        )
        self.model.eval()
        logger.info("模型加载完成。")

    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        为单个图片文件生成向量。

        Args:
            image_path (str): 图片文件路径。

        Returns:
            torch.Tensor: 生成的图片向量。
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input) # 224x224x3 -> 512/786 vector (FAISS)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features
        except Exception as e:
            logger.error(f"处理图片失败: {image_path}, 错误: {e}")
            return None

    def encode_text(self, text: str) -> torch.Tensor:
        """
        为单个文本生成向量。

        Args:
            text (str): 输入文本。

        Returns:
            torch.Tensor: 生成的文本向量。
        """
        try:
            text_input = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features
        except Exception as e:
            logger.error(f"处理文本失败: {text}, 错误: {e}")
            return None 