import runpod
import os
from vllm import LLM, SamplingParams

# Modell inicializálása indításkor (cold start alatt egyszer fut le)
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    enforce_eager=False
)

def handler(job):
    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "")
    
    if not prompt:
        return {"status": "error", "message": "Hiányzik a 'prompt' mező!"}

    try:
        sampling_params = SamplingParams(
            temperature=job_input.get("temperature", 0.7),
            max_tokens=job_input.get("max_tokens", 2048),
            repetition_penalty=1.05
        )
        
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return {
            "status": "success",
            "generated_text": generated_text
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})