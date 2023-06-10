from PIL import Image

from torchvision import models
from SDManager import StableDiffusionManager
model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=1).eval()

# GENERATE

stable = StableDiffusionManager()
pilImage = Image.open("temp.png")
prompt="Create a visually stunning image featuring a tranquil bamboo background. The bamboo forest should exude a sense of serenity, with tall, slender bamboo stalks gently swaying in the breeze. The setting should evoke a feeling of peace and harmony.  Specifics:  The bamboo forest should be dense, with bamboo stalks filling the frame from top to bottom. The stalks should be various shades of green, showcasing the natural beauty of bamboo. The sunlight filters through the canopy, casting dappled shadows on the ground, creating an enchanting play of light and shadow. The composition should capture the height of the bamboo, highlighting their verticality and elegance. Some stalks can be closer to the viewer, allowing for depth and a sense of immersion. Surrounding the bamboo forest, there could be hints of other elements like rocks, moss, or small streams, adding a touch of organic diversity to the scene. The overall mood of the image should convey tranquility and calmness, inviting viewers to step into the serene world of bamboo."
stable.ImageGenerateMy(prompt, pilImage, False)

# REMOVE

"""input_path = 'temp_remove.png'
output_path = 'content//deneme_out_remove.png'
input = Image.open(input_path)
stable = StableDiffusionManager()
prompt="Create a visually stunning image featuring a tranquil bamboo background. The bamboo forest should exude a sense of serenity, with tall, slender bamboo stalks gently swaying in the breeze. The setting should evoke a feeling of peace and harmony.  Specifics:  The bamboo forest should be dense, with bamboo stalks filling the frame from top to bottom. The stalks should be various shades of green, showcasing the natural beauty of bamboo. The sunlight filters through the canopy, casting dappled shadows on the ground, creating an enchanting play of light and shadow. The composition should capture the height of the bamboo, highlighting their verticality and elegance. Some stalks can be closer to the viewer, allowing for depth and a sense of immersion. Surrounding the bamboo forest, there could be hints of other elements like rocks, moss, or small streams, adding a touch of organic diversity to the scene. The overall mood of the image should convey tranquility and calmness, inviting viewers to step into the serene world of bamboo."
stable.RemoveBackground(prompt, 'temp_remove.png')"""