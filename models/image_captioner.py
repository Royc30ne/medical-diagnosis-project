from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch

class ImageCaptioner:
    def __init__(self, model_name="nickmuchi/vit-finetuned-chest-xray-pneumonia"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

    def describe(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted = logits.argmax(-1)
        label = self.model.config.id2label[predicted.item()]
        return f"X-ray finding: {label}"