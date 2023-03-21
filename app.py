import cv2
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import time

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global controlnet
    
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )

    # Midas
    global dpt_model
    global feature_extractor

    feature_extractor  = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global controlnet

    global dpt_model
    global feature_extractor
    
    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', None)
    num_inference_steps = model_inputs.get('num_inference_steps', 20)
    image_data = model_inputs.get('image_data', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB") 
    # Run DPT
    inputs = feature_extractor(images = image, return_tensors="pt")


    timestart = time.time()
    with torch.no_grad():
        outputs = dpt_model(**inputs)
        predicted_depth = outputs.predicted_depth
    print("Time taken for depth map: ", time.time() - timestart)

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    depth = prediction.squeeze().cpu().numpy()
    depth_image = Image.fromarray((depth * 255 / np.max(depth)).astype(np.uint8)).convert("RGB")

    buffered = BytesIO()
    depth_image.save(buffered,format="JPEG")
    depth_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    model.enable_model_cpu_offload()
    model.enable_xformers_memory_efficient_attention()
    output = model(
        prompt,
        depth_image,
        negative_prompt=negative_prompt,
        num_inference_steps=20
    )

    image = output.images[0]
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {
        'depth_base64': depth_base64,
        'image_base64': image_base64
    }

if __name__ == '__main__':
    init()
    with open("input/room.jpg", "rb") as f:
        img_bytes = f.read()

    inference({"prompt": "tropical room"}, img_bytes, debug=True)