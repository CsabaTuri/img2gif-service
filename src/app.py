import os
import requests
import torch
import base64
from io import BytesIO
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

class GIFGeneratorService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llava")

    def enhance_prompt_with_ollama(self, image_b64: str, user_instruction: str) -> str:
        prompt_text = (
            f"Analyze this image and the request: '{user_instruction}'. "
            f"Write a concise image generation prompt describing the scene and intended movement."
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt_text,
            "images": [image_b64],
            "stream": False
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception:
            pass
        return user_instruction

    def load_diffusers_model(self):
        if self.pipe is None:
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2", 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipe = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                motion_adapter=adapter,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config,
                clip_sample=False,
                timestep_spacing="linspace",
                steps_offset=1
            )
            if self.device == "cuda":
                self.pipe.enable_vae_slicing()
                self.pipe.to("cuda")

    def generate_gif_from_image(self, image_b64: str, instruction: str, num_frames: int = 16) -> str:
        final_prompt = self.enhance_prompt_with_ollama(image_b64, instruction)
        self.load_diffusers_model()
        
        output = self.pipe(
            prompt=final_prompt,
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=20
        )
        frames = output.frames[0]
        
        buffer = BytesIO()
        export_to_gif(frames, buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

service = GIFGeneratorService()