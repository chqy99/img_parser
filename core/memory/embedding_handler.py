import torch
import numpy as np
from PIL import Image
from core.modules.clip_module import ClipModule
from core.modules.module_factory import ModuleFactory

class EmbeddingHandler:
    """
    Handles extracting image embeddings using CLIP.
    """

    def __init__(self, clip_module: ClipModule = None):
        self.clip_module = clip_module or ModuleFactory.get_module("clip")

    def _load_default_clip(self):
        # You may want to customize this to load your own config
        from core.modules.model_config import ModelLoader

        model_bundle = ModelLoader().get_model("clip")
        return ClipModule(
            model=model_bundle["model"],
            processor=model_bundle["processor"],
            label_texts=model_bundle["label_texts"],
        )

    def get_embedding(self, image: np.ndarray):
        # Use CLIP to get image embedding
        img = Image.fromarray(image.astype("uint8"))
        inputs = self.clip_module.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.clip_module.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.clip_module.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]
