import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def generate_synthetic_images(objects, output_dir, num_images=5):
    """Generate synthetic images using Stable Diffusion"""
    try:
        # Initialize Stable Diffusion pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate images for each object
        for obj in objects:
            obj_dir = os.path.join(output_dir, obj)
            os.makedirs(obj_dir, exist_ok=True)
            prompt = f"A high-quality image of a {obj} in a construction site, realistic lighting, detailed background"
            for i in range(num_images):
                image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                image_path = os.path.join(obj_dir, f"{obj}_{i+1}.jpg")
                image.save(image_path)
                print(f"Saved {image_path}")
    except Exception as e:
        print(f"Error generating images: {e}")

if __name__ == '__main__':
    output_dir = 'images/train'
    generate_synthetic_images(objects=['Person', 'sewing machine'], output_dir=output_dir, num_images=5)