import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

class VisionAnalyzer:
    def __init__(self, model_id: str = "Salesforce/blip-vqa-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device)
        
    def analyze_frame(self, image_path: str, prompt: str) -> str:
        """
        Loads an image, processes it along with the prompt, and returns the VQA output.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading image from {image_path}: {e}")
            
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        
        # Generate the answer
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50)
            
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)
        return decoded_output
