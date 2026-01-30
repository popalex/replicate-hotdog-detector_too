from cog import BasePredictor, Input
import torch
from diffusers import StableDiffusionXLPipeline

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        num_steps: int = Input(default=30),
    ) -> str:
        image = self.pipe(prompt, num_inference_steps=num_steps).images[0]
        out_path = "/tmp/out.png"
        image.save(out_path)
        return out_path
