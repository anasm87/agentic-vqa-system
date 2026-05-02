import os
import sys
import requests

# Add the parent directory to the python path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vqa_engine import VisionAnalyzer

from PIL import Image, ImageDraw

def download_sample_image(image_path: str):
    # Creating a placeholder image of a street scene
    img = Image.new('RGB', (800, 600), color=(135, 206, 235)) # Sky blue
    d = ImageDraw.Draw(img)
    # Draw road
    d.rectangle([0, 300, 800, 600], fill=(100, 100, 100))
    # Draw road lines
    d.rectangle([390, 300, 410, 600], fill=(255, 255, 0))
    # Draw a red traffic light
    d.rectangle([700, 50, 750, 200], fill=(30, 30, 30))
    d.ellipse([710, 60, 740, 90], fill=(255, 0, 0)) # Red light
    # Draw a blue car
    d.rectangle([300, 350, 450, 450], fill=(0, 0, 255))
    d.rectangle([320, 320, 430, 350], fill=(0, 0, 200))
    img.save(image_path)

def main():
    print("Downloading sample image...")
    image_path = "sample_dashcam.jpg"
    if not os.path.exists(image_path):
        download_sample_image(image_path)
        
    print("Initializing VisionAnalyzer...")
    analyzer = VisionAnalyzer()
    
    prompt = "Describe the state of the road, vehicles, and traffic lights."
    print(f"Running inference with prompt: '{prompt}'")
    
    result = analyzer.analyze_frame(image_path, prompt)
    
    print("\n--- VLM Output ---")
    print(result)
    print("------------------")

if __name__ == "__main__":
    main()
