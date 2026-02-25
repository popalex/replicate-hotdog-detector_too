from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionXLPipeline

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading Stable Diffusion XL model into memory...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Prompt for image generation"),
        num_steps: int = Input(
            description="Number of denoising steps",
            default=30,
            ge=1,
            le=100
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        print(f"Generating image with prompt: {prompt}")
        print(f"Using {num_steps} inference steps")
        
        image = self.pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=7.5
        ).images[0]
        
        out_path = "/tmp/out.png"
        image.save(out_path)
        print(f"Image saved to {out_path}")
        
        return Path(out_path)
