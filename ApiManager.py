import glob
import io
import threading
from base64 import encodebytes
from threading import Thread

from PIL import Image
from flask import Flask, render_template, redirect, request, send_file, jsonify
from SDManager import StableDiffusionManager


import traceback
from werkzeug.wsgi import ClosingIterator

class AfterResponse:
    def __init__(self, app=None):
        self.callbacks = []
        if app:
            self.init_app(app)

    def __call__(self, callback):
        self.callbacks.append(callback)
        return callback

    def init_app(self, app):
        # install extension
        app.after_response = self

        # install middleware
        app.wsgi_app = AfterResponseMiddleware(app.wsgi_app, self)

    def flush(self):
        for fn in self.callbacks:
            try:
                fn()
            except Exception:
                traceback.print_exc()

class AfterResponseMiddleware:
    def __init__(self, application, after_response_ext):
        self.application = application
        self.after_response_ext = after_response_ext

    def __call__(self, environ, after_response):
        iterator = self.application(environ, after_response)
        try:
            return ClosingIterator(iterator, [self.after_response_ext.flush])
        except Exception:
            traceback.print_exc()
            return iterator


app = Flask(__name__)
AfterResponse(app)
app.app_context().push()
import os
secret_key = os.urandom(24)
app.secret_key = secret_key
_photo = None
_prompt = ""
@app.route('/')
def index():
    return render_template('upload.html')



@app.route('/upload', methods=['POST'])
def upload():
    photo = request.files['photo']
    text = request.form['text']
    photo.save("temp.png")
    _photo = photo
    _prompt = text
    #stable = StableDiffusionManager()
    # Create a thread to process the image
    def process_image_thread(text, image):
        with app.app_context():
            pilImage = Image.open("temp.png")
            stable = StableDiffusionManager()
            stable.ImageGenerate(text, pilImage, False)


    process_thread = threading.Thread(target=process_image_thread, args=(_prompt, _photo))
    process_thread.start()
    process_thread.join()

    #Token indices sequence length is longer than the specified maximum sequence length for this model (201 > 77). Running this sequence through the model will result in indexing errors
    #Prompt =  Create a visually stunning image featuring a tranquil bamboo background. The bamboo forest should exude a sense of serenity, with tall, slender bamboo stalks gently swaying in the breeze. The setting should evoke a feeling of peace and harmony.  Specifics:  The bamboo forest should be dense, with bamboo stalks filling the frame from top to bottom. The stalks should be various shades of green, showcasing the natural beauty of bamboo. The sunlight filters through the canopy, casting dappled shadows on the ground, creating an enchanting play of light and shadow. The composition should capture the height of the bamboo, highlighting their verticality and elegance. Some stalks can be closer to the viewer, allowing for depth and a sense of immersion. Surrounding the bamboo forest, there could be hints of other elements like rocks, moss, or small streams, adding a touch of organic diversity to the scene. The overall mood of the image should convey tranquility and calmness, inviting viewers to step into the serene world of bamboo.

    result = []
    result.append("Inpainted_OutPainted.png")
    return render_template(
        'generatedImages.html', results=result
    )
    # Process the uploaded photo and text here


@app.route('/generated', methods=['GET'])
def getGenerated():
    image_list = []
    for filename in glob.glob('static//*.png'):
        print(filename)
        image_list.append(filename.split('\\')[1])

    return render_template(
        'generatedImages.html', results=image_list
    )


@app.route('/RemoveBackground', methods=['POST'])
def RemoveBackground():
    photo = request.files['photo']
    text = request.form['text']
    photo.save("temp_remove.png")
    _photo = photo
    _prompt = text

    def process_image_thread(text, image):
        with app.app_context():
            pilImage = Image.open("temp_remove.png")
            stable = StableDiffusionManager()
            stable.RemoveBackground(text, pilImage)

    process_thread = threading.Thread(target=process_image_thread, args=(_prompt, _photo))
    process_thread.start()
    process_thread.join()

    #Token indices sequence length is longer than the specified maximum sequence length for this model (201 > 77). Running this sequence through the model will result in indexing errors
    #Prompt =  Create a visually stunning image featuring a tranquil bamboo background. The bamboo forest should exude a sense of serenity, with tall, slender bamboo stalks gently swaying in the breeze. The setting should evoke a feeling of peace and harmony.  Specifics:  The bamboo forest should be dense, with bamboo stalks filling the frame from top to bottom. The stalks should be various shades of green, showcasing the natural beauty of bamboo. The sunlight filters through the canopy, casting dappled shadows on the ground, creating an enchanting play of light and shadow. The composition should capture the height of the bamboo, highlighting their verticality and elegance. Some stalks can be closer to the viewer, allowing for depth and a sense of immersion. Surrounding the bamboo forest, there could be hints of other elements like rocks, moss, or small streams, adding a touch of organic diversity to the scene. The overall mood of the image should convey tranquility and calmness, inviting viewers to step into the serene world of bamboo.

    result = []
    result.append("backGroundRemoved.png")
    result.append("BGRemoved_Inpainted_Upscaled.png")
    return render_template(
        'generatedImages.html', results=result
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6790, debug=False)


