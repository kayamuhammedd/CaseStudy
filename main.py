import PIL
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")

photo = PIL.Image.open("perfume.png").convert("RGB")
newsize = (512, 512)
photo = photo.resize(newsize)

image_array = np.array(photo)
white_pixels = np.where(np.all(image_array == 255, axis=-1))
mask = np.zeros_like(image_array)
mask[white_pixels] = 255
mask_image = PIL.Image.fromarray(mask)

prompt = "Create a background for my original photo."
#background = pipe(prompt).images[0]

#background.save("back.png")

torch.manual_seed(2023)

inp_img = photo  # loaded with PIL.Image
mask = mask_image      # also PIL.Image
inner_image = inp_img.convert("RGBA")

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    custom_pipeline="img2img_inpainting",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

negative_prompt = "do not change the mask photo content"

result = pipe(prompt=prompt, image=inp_img, inner_image=inner_image,
    mask_image=mask,
    num_inference_steps = 50, guidance_scale = 10).images
result[0].save("generated.png")  # this is the generated image
#PIL.Image.fromarray(result[0]).save("combined_image.jpg")
