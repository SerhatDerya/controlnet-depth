# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import torch
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    
    path = {"midas": ["ckpt/dpt_hybrid-midas-501f0c75.pt","https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"]}
    torch.hub.download_url_to_file(path["midas"][1], path["midas"][0])


if __name__ == "__main__":
    download_model()