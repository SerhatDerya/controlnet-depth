# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import torch
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    # Midas
    global dpt_model
    global feature_extractor

    feature_extractor  = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)


if __name__ == "__main__":
    download_model()