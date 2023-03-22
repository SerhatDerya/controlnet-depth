import cv2
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from midas import apply_midas
from midas.util import HWC3, resize_image
import time

from midas.api import MiDaSInference

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global controlnet
    
    timestart = time.time()
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    print("Time to load controlnet: ", time.time() - timestart)

    # Midas
    timestart = time.time()
    global midas_model
    midas_model = MiDaSInference(model_type="dpt_hybrid").cuda()
    print("Time to load Midas: ", time.time() - timestart)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global controlnet

    global midas_model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', None)
    num_inference_steps = model_inputs.get('num_inference_steps', 20)
    image_data = model_inputs.get('image_data', None)
    detect_resolution = model_inputs.get('detect_resolution', 384)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")  #cambiar

    timestart = time.time()
    # Run MiDAS
    with torch.no_grad():
        input_image = HWC3(np.array(image))
        detected_map, _ = apply_midas(resize_image(input_image, detect_resolution), model=midas_model)
        detected_map = HWC3(detected_map)
    depth_image = Image.fromarray(detected_map)
    
    print("Time to run MiDAS: ", time.time() - timestart)

    buffered = BytesIO()

    depth_image.save(buffered,format="JPEG")
    depth_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    timestart = time.time()
    
    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    model.enable_model_cpu_offload()
    model.enable_xformers_memory_efficient_attention()
    output = model(
        prompt,
        depth_image,
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_inference_steps)
    )

    image = output.images[0]
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print("Time to run ControlNet: ", time.time() - timestart)
    # Return the results as a dictionary
    return {
        'depth_base64': depth_base64,
        'image_base64': image_base64
    }