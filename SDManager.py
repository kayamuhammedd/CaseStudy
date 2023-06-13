from PIL import Image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import u2net
import io
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionUpscalePipeline, ControlNetModel, \
    StableDiffusionControlNetPipeline, PNDMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

class StableDiffusionManager():
        def __init__(self):
            self.prompt="",


        def ImageGenerate(self, prompt, image, isRemoveBackground):
            if isRemoveBackground:
                inpainting_repo_id = "stabilityai/stable-diffusion-2-inpainting"
                upscaler_repo_id = "stabilityai/stable-diffusion-x4-upscaler"

                # init_image_full_size = Image.open(image)  # .convert("RGB")
                """new_image = Image.new("RGBA", (1920,1080), "WHITE")  # Create a white rgba background
                new_image.paste(image, (0, 0), image)"""
                init_image = image.resize((512, 512))
                mask_image = np.array(init_image)[:, :,
                             3]  # assume image has alpha mask (use .mode to check for "RGBA")
                mask_image = Image.fromarray(255 - mask_image).convert("RGB")
                init_image = init_image.convert("RGB")

                pipe = DiffusionPipeline.from_pretrained(inpainting_repo_id, torch_dtype=torch.float16, revision="fp16")
                pipe.set_use_memory_efficient_attention_xformers(True)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.enable_attention_slicing()
                pipe = pipe.to("cuda")
                max_length_pipe = pipe.tokenizer.model_max_length

                def dummy(images, **kwargs):
                    return images, False

                pipe.safety_checker = dummy

                uppipe = StableDiffusionUpscalePipeline.from_pretrained(upscaler_repo_id, torch_dtype=torch.float16)
                uppipe.set_use_memory_efficient_attention_xformers(True)
                uppipe = uppipe.to("cuda")
                uppipe.enable_attention_slicing()

                print("Prompt = ", prompt)
                orig_prompt = prompt
                orig_negative_prompt = "text, widen, extend, bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"

                num_images = 1
                prompt = [orig_prompt] * num_images
                negative_prompt = [orig_negative_prompt] * num_images

                input_ids = pipe.tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length_pipe,
                                           return_tensors="pt").input_ids
                input_ids = input_ids.to("cuda")

                negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                              max_length=input_ids.shape[-1],
                                              return_tensors="pt").input_ids
                negative_ids = negative_ids.to("cuda")

                concat_embeds = []
                neg_embeds = []
                for i in range(0, input_ids.shape[-1], max_length_pipe):
                    concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length_pipe])[0])
                    neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length_pipe])[0])

                prompt_embeds = torch.cat(concat_embeds, dim=1)
                negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

                mask_image.save(f"static//mask_image.png")
                init_image.save(f"static//init_image.png")
                images = \
                pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=init_image,
                     mask_image=mask_image,
                     num_inference_steps=100)[0]

                images[0].save("static//first_image_remove.png")
                hi_res_images = []
                for i in range(num_images):
                    low_res_image = images[i]
                    prompt = [orig_prompt] * 1
                    negative_prompt = [orig_negative_prompt] * 1
                    hi_res_image = uppipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                          image=low_res_image.resize((256, 256))).images[0]
                    hi_res_image = hi_res_image.convert("RGB")
                    hi_res_images.append(hi_res_image)

                x, y = image.size
                if isRemoveBackground:
                    final_image = hi_res_images[0]
                    final_image.paste(image.convert("RGBA"), mask=image.convert("RGBA"))
                    final_image.save(f"static//BGRemoved_Inpainted_Upscaled.png")
                else:
                    final_image = hi_res_images[0].resize((x, y))
                    final_image.paste(image.convert("RGBA"), mask=image.convert("RGBA"))
                    final_image.save(f"static//Inpainted_OutPainted.png")
            else:
                inpainting_repo_id = "stabilityai/stable-diffusion-2-inpainting"
                upscaler_repo_id = "stabilityai/stable-diffusion-x4-upscaler"

                # init_image_full_size = Image.open(image)  # .convert("RGB")
                init_image = image.resize((512, 512))
                mask_image = np.array(init_image)[:, :,
                             3]  # assume image has alpha mask (use .mode to check for "RGBA")
                mask_image = np.array(Image.fromarray(mask_image.astype(np.uint8)).resize((512, 512))).astype('float32')
                init_image = np.array(image.resize((512, 512))).astype(np.uint8)

                print("init_image = ", init_image.shape[0])
                print("init_image 1= ", init_image.shape[1])
                print("image= ", image.size)
                y, x = image.size
                # Insert image2 into image1

                """mask_image = Image.fromarray(255 - mask_image).convert("RGB")

                image1 = np.ones((512, 512, 3), dtype=np.uint8) *255
                image1[start_row:start_row + x, start_col:start_col + y] = mask_image
                image1 = Image.fromarray(image1.astype(np.uint8)).resize((512, 512))
                init_image = init_image.convert("RGB")"""

                # Create a new 512x512 array filled with 255 values
                expanded_array = np.ones((x, y, 3), dtype=np.uint8) * 255

                # Copy the original array into the larger array
                # Calculate the coordinates to place the original image in the middle
                start_row = (expanded_array.shape[0] - mask_image.shape[0]) // 2
                end_row = start_row + mask_image.shape[0]
                start_col = (expanded_array.shape[1] - mask_image.shape[1]) // 2
                end_col = start_col + mask_image.shape[1]
                mask_image = Image.fromarray(255 - mask_image).convert("RGB")

                # Insert the original image into the middle of the expanded image
                expanded_array[start_row:end_row, start_col:end_col:, ] = mask_image
                image1 = (Image.fromarray(expanded_array.astype(np.uint8)).resize((512, 512)))

                expanded_array_init_image = np.zeros((x, y, 3), dtype=np.uint8)
                # Copy the original array into the larger array
                # Calculate the coordinates to place the original image in the middle
                start_row = (expanded_array_init_image.shape[0] - init_image.shape[1]) // 2
                end_row = start_row + init_image.shape[1]
                start_col = (expanded_array_init_image.shape[1] - init_image.shape[0]) // 2
                end_col = start_col + init_image.shape[0]

                init_image = Image.fromarray(init_image).convert("RGB")
                print("muh = ", image.size)
                # Insert the original image into the middle of the expanded image
                expanded_array_init_image[start_row:end_row, start_col:end_col:, ] = init_image
                init_image = (Image.fromarray(expanded_array_init_image.astype(np.uint8)).resize((512, 512)))

                init_image.save("init_image.png")
                image1.save("mask.png")

                pipe = DiffusionPipeline.from_pretrained(inpainting_repo_id, torch_dtype=torch.float16, revision="fp16")
                pipe.set_use_memory_efficient_attention_xformers(True)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.enable_attention_slicing()
                pipe = pipe.to("cuda")
                max_length_pipe = pipe.tokenizer.model_max_length

                def dummy(images, **kwargs):
                    return images, False

                pipe.safety_checker = dummy

                uppipe = StableDiffusionUpscalePipeline.from_pretrained(upscaler_repo_id, torch_dtype=torch.float16)
                uppipe.set_use_memory_efficient_attention_xformers(True)
                uppipe = uppipe.to("cuda")
                uppipe.enable_attention_slicing()
                max_length_uppipe = uppipe.tokenizer.model_max_length
                print("Prompt = ", prompt)
                orig_prompt = prompt
                orig_negative_prompt = "widen, extend, bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"

                num_images = 1
                prompt = [orig_prompt] * num_images
                negative_prompt = [orig_negative_prompt] * num_images

                input_ids = pipe.tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length_pipe,
                                           return_tensors="pt").input_ids
                input_ids = input_ids.to("cuda")

                negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                              max_length=input_ids.shape[-1],
                                              return_tensors="pt").input_ids
                negative_ids = negative_ids.to("cuda")

                concat_embeds = []
                neg_embeds = []
                for i in range(0, input_ids.shape[-1], max_length_pipe):
                    concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length_pipe])[0])
                    neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length_pipe])[0])

                prompt_embeds = torch.cat(concat_embeds, dim=1)
                negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

                images = \
                pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=init_image,
                     mask_image=mask_image if isRemoveBackground else image1,
                     num_inference_steps=100)[0]

                print(images[0].size)
                images[0].save("MyFrst.png")
                # images[1].save("MySecond.png")
                hi_res_images = []
                for i in range(num_images):
                    low_res_image = images[i]
                    prompt = [orig_prompt] * 1
                    negative_prompt = [orig_negative_prompt] * 1
                    hi_res_image = \
                        uppipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                               image=low_res_image.resize((256, 256))).images[0]
                    hi_res_image = hi_res_image.convert("RGB")
                    hi_res_images.append(hi_res_image)

                x, y = image.size
                if isRemoveBackground:
                    final_image = hi_res_images[0]
                    final_image.paste(image.convert("RGBA"), mask=image.convert("RGBA"))
                    final_image.save(f"static//BGRemoved_Inpainted_Upscaled.png")
                else:
                    final_image = hi_res_images[0].resize((x, y))
                    print(final_image.size)
                    print(image.size)
                    print(image1.size)
                    # final_image.paste(init_image.convert("RGBA"), mask=init_image.convert("RGBA"))
                    final_image.save(f"static//Inpainted_OutPainted.png")


        def RemoveBackground(self, prompt, image):

            img = image

            if img.size[0] > 1024 or img.size[1] > 1024:
                img.thumbnail((1024, 1024))

            res = u2net.run(np.array(img))

            mask = res.convert('L').resize((img.width, img.height))

            empty = Image.new("RGBA", img.size, 0)
            img = Image.composite(img, empty, mask)

            buff = io.BytesIO()
            img.save("static//backGroundRemoved.png")
            buff.seek(0)

            self.ImageGenerate(prompt, img, True)


        def ImageGenerateMy(self, prompt, image, isRemoveBackground):
            pndm = PNDMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")
            if isRemoveBackground:
                inpainting_repo_id = "stabilityai/stable-diffusion-2-inpainting"
                upscaler_repo_id = "stabilityai/stable-diffusion-x4-upscaler"

                # init_image_full_size = Image.open(image)  # .convert("RGB")
                """new_image = Image.new("RGBA", (1920,1080), "WHITE")  # Create a white rgba background
                new_image.paste(image, (0, 0), image)"""
                init_image = image.resize((512, 512))
                mask_image = np.array(init_image)[:, :, 3]  # assume image has alpha mask (use .mode to check for "RGBA")
                mask_image = Image.fromarray(255 - mask_image).convert("RGB")
                init_image = init_image.convert("RGB")

                pipe = DiffusionPipeline.from_pretrained(inpainting_repo_id, torch_dtype=torch.float16, revision="fp16")
                pipe.set_use_memory_efficient_attention_xformers(True)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.enable_attention_slicing()
                pipe = pipe.to("cuda")
                max_length_pipe = pipe.tokenizer.model_max_length
                def dummy(images, **kwargs):
                    return images, False

                pipe.safety_checker = dummy

                uppipe = StableDiffusionUpscalePipeline.from_pretrained(upscaler_repo_id, torch_dtype=torch.float16)
                uppipe.set_use_memory_efficient_attention_xformers(True)
                uppipe = uppipe.to("cuda")
                uppipe.enable_attention_slicing()

                print("Prompt = ", prompt)
                orig_prompt = prompt
                orig_negative_prompt = "text, widen, extend, bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"

                num_images = 1
                prompt = [orig_prompt] * num_images
                negative_prompt = [orig_negative_prompt] * num_images


                input_ids = pipe.tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length_pipe,
                                           return_tensors="pt").input_ids
                input_ids = input_ids.to("cuda")

                negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                              max_length=input_ids.shape[-1],
                                              return_tensors="pt").input_ids
                negative_ids = negative_ids.to("cuda")

                concat_embeds = []
                neg_embeds = []
                for i in range(0, input_ids.shape[-1], max_length_pipe):
                    concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length_pipe])[0])
                    neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length_pipe])[0])

                prompt_embeds = torch.cat(concat_embeds, dim=1)
                negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

                mask_image.save(f"static//mask_image.png")
                init_image.save(f"static//init_image.png")
                images = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=init_image, mask_image=mask_image,
                              num_inference_steps=100)[0]

                images[0].save("static//first_image_remove.png")
                hi_res_images = []
                for i in range(num_images):
                    low_res_image = images[i]
                    prompt = [orig_prompt] * 1
                    negative_prompt = [orig_negative_prompt] * 1
                    hi_res_image = uppipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=low_res_image.resize((256, 256))).images[0]
                    hi_res_image = hi_res_image.convert("RGB")
                    hi_res_images.append(hi_res_image)

                x, y = image.size
                if isRemoveBackground:
                    final_image = hi_res_images[0]
                    final_image.paste(image.convert("RGBA"), mask=image.convert("RGBA"))
                    final_image.save(f"static//BGRemoved_Inpainted_UpscaledMY.png")
                else:
                    final_image = hi_res_images[0].resize((x, y))
                    final_image.paste(image.convert("RGBA"), mask=image.convert("RGBA"))
                    final_image.save(f"static//Inpainted_OutPaintedMY.png")
            else:
                inpainting_repo_id = "stabilityai/stable-diffusion-2-inpainting"
                upscaler_repo_id = "stabilityai/stable-diffusion-x4-upscaler"

                #init_image_full_size = Image.open(image)  # .convert("RGB")
                init_image = image.resize((512, 512))
                mask_image = np.array(init_image)[:, :, 3]  # assume image has alpha mask (use .mode to check for "RGBA")
                mask_image = np.array(Image.fromarray(mask_image.astype(np.uint8)).resize((512, 512))).astype('float32')
                init_image = np.array(image.resize((512, 512))).astype(np.uint8)

                y, x = image.size
                # Insert image2 into image1

                """mask_image = Image.fromarray(255 - mask_image).convert("RGB")
    
                image1 = np.ones((512, 512, 3), dtype=np.uint8) *255
                image1[start_row:start_row + x, start_col:start_col + y] = mask_image
                image1 = Image.fromarray(image1.astype(np.uint8)).resize((512, 512))
                init_image = init_image.convert("RGB")"""

                # Create a new 512x512 array filled with 255 values
                expanded_array = np.ones((x, y,3 ), dtype=np.uint8) * 255


                # Copy the original array into the larger array
                # Calculate the coordinates to place the original image in the middle
                start_row = (expanded_array.shape[0] - mask_image.shape[0]) // 2
                end_row = start_row + mask_image.shape[0]
                start_col = (expanded_array.shape[1] - mask_image.shape[1]) // 2
                end_col = start_col + mask_image.shape[1]
                mask_image = Image.fromarray(255 - mask_image).convert("RGB")

                # Insert the original image into the middle of the expanded image
                expanded_array[start_row:end_row, start_col:end_col:,] = mask_image
                image1 =(Image.fromarray(expanded_array.astype(np.uint8)).resize((512, 512)))



                expanded_array_init_image = np.zeros((x, y, 3), dtype=np.uint8)
                # Copy the original array into the larger array
                # Calculate the coordinates to place the original image in the middle
                start_row = (expanded_array_init_image.shape[0] -  init_image.shape[1]) // 2
                end_row = start_row +   init_image.shape[1]
                start_col = (expanded_array_init_image.shape[1] -  init_image.shape[0]) // 2
                end_col = start_col +   init_image.shape[0]

                init_image =Image.fromarray(init_image).convert("RGB")
                print("muh = ", image.size)
                # Insert the original image into the middle of the expanded image
                expanded_array_init_image[start_row:end_row, start_col:end_col:, ] = init_image
                init_image = (Image.fromarray(expanded_array_init_image.astype(np.uint8)).resize((512, 512)))


                init_image.save("init_image.png")
                image1.save("mask.png")

                pipe = DiffusionPipeline.from_pretrained(inpainting_repo_id, torch_dtype=torch.float16, revision="fp16")
                pipe.set_use_memory_efficient_attention_xformers(True)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.enable_attention_slicing()
                pipe = pipe.to("cuda")
                max_length_pipe = pipe.tokenizer.model_max_length
                def dummy(images, **kwargs):
                    return images, False

                pipe.safety_checker = dummy

                uppipe = StableDiffusionUpscalePipeline.from_pretrained(upscaler_repo_id, torch_dtype=torch.float16)
                uppipe.set_use_memory_efficient_attention_xformers(True)
                uppipe = uppipe.to("cuda")
                uppipe.enable_attention_slicing()
                max_length_uppipe = uppipe.tokenizer.model_max_length
                print("Prompt = ", prompt)
                orig_prompt = prompt
                orig_negative_prompt = "text, widen, extend, bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"

                num_images =1
                prompt = [orig_prompt] * num_images
                negative_prompt = [orig_negative_prompt] * num_images

                input_ids = pipe.tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length_pipe,return_tensors="pt").input_ids
                input_ids = input_ids.to("cuda")

                negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1],
                                              return_tensors="pt").input_ids
                negative_ids = negative_ids.to("cuda")

                concat_embeds = []
                neg_embeds = []
                for i in range(0, input_ids.shape[-1], max_length_pipe):
                    concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length_pipe])[0])
                    neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length_pipe])[0])

                prompt_embeds = torch.cat(concat_embeds, dim=1)
                negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

                images = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=init_image, mask_image=mask_image if isRemoveBackground else image1,
                              num_inference_steps=100)[0]

                print(images[0].size)
                images[0].save("MyFrst.png")
                #images[1].save("MySecond.png")
                hi_res_images = []
                for i in range(num_images):
                    low_res_image = images[i]
                    hi_res_image = \
                    uppipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=low_res_image.resize((256, 256))).images[0]
                    hi_res_image = hi_res_image.convert("RGB")
                    hi_res_images.append(hi_res_image)

                x, y = image.size
                if isRemoveBackground:
                    final_image = hi_res_images[0]
                    final_image.paste(image.convert("RGBA"), mask=image.convert("RGBA"))
                    final_image.save(f"static//BGRemoved_Inpainted_UpscaledMy.png")
                else:
                    final_image = hi_res_images[0].resize((x, y))
                    #final_image2 = hi_res_images[1].resize((x, y))
                    final_image.paste(image.convert("RGBA"), mask=image.convert("RGBA"))
                    final_image.save(f"static//Inpainted_OutPaintedMY.png")
                    #final_image2.save(f"static//Inpainted_OutPainted2MY.png")


        def RemoveBackgroundMy(self, prompt, image):

            img = image

            if img.size[0] > 1024 or img.size[1] > 1024:
                img.thumbnail((1024, 1024))

            res = u2net.run(np.array(img))

            mask = res.convert('L').resize((img.width, img.height))

            empty = Image.new("RGBA", img.size, 0)
            img = Image.composite(img, empty, mask)

            buff = io.BytesIO()
            img.save("u2net.png")
            buff.seek(0)

            self.ImageGenerateMy(prompt, img, True)

            """#init_image_full_size = Image.open(img)  # .convert("RGB")
            init_image = img.resize((512, 512))
            mask_image = np.array(init_image)[:, :, ]  # assume image has alpha mask (use .mode to check for "RGBA")
            mask_image = Image.fromarray(255 - mask_image).convert("RGB")
            init_image = init_image.convert("RGB")

            mask_image.save("mask_remove_my.png")

            controlnet = ControlNetModel.from_pretrained(remove_repo_id, torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            )

            pipe.set_use_memory_efficient_attention_xformers(True)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to("cuda")

            def dummy(images, **kwargs):
                return images, False

            pipe.safety_checker = dummy


            #orig_prompt = "A " + description + ", epic, exciting, wow, cinematic, moody, exciting, stop motion, highly detailed, octane render, soft lighting, professional, 35mm, Zeiss, Hasselblad, Fujifilm, Arriflex, IMAX, 4k, 8k"
            orig_negative_prompt = "bright, oversaturated, ugly, 3d, render, cartoon, grain, low-res, kitsch, blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry, bad art, blurred, text, watermark, disfigured, deformed, mangled"

            num_images = 1
            prompt = [prompt] * num_images
            negative_prompt = [orig_negative_prompt] * num_images

            images = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, num_inference_steps=25)[0]

            images[0].save("first_image_remove.png")"""