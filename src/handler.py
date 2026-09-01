import runpod
from app import service

def handler(event):
    job_input = event.get("input", {})
    
    image_b64 = job_input.get("image_base64")
    instruction = job_input.get("instruction", "Animate this image smoothly")
    num_frames = job_input.get("num_frames", 16)

    if not image_b64:
        return {"status": "error", "message": "Hiányzik az 'image_base64' mező!"}

    try:
        gif_b64 = service.generate_gif_from_image(
            image_b64=image_b64,
            instruction=instruction,
            num_frames=num_frames
        )
        return {
            "status": "success",
            "format": "gif",
            "gif_base64": gif_b64
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})