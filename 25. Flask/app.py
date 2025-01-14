from flask import Flask, render_template, request
import numpy as np
# import keras.models
import re
import sys
import os
import base64
from PIL import Image
import imageio
sys.path.append(os.path.abspath("./model"))
from load import * 

# global graph, model
global model

# model, graph = init()
model = init()

app = Flask(__name__)


@app.route('/')
def index_view():
    return render_template('index.html')


def convertImage(imgData1):
    # searching for 'base64,' and saving the string after it to 'output.png' as an image
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    
    # Read the image using imageio
    img = imageio.imread('output.png', mode = 'L')
    img = np.invert(img)
    
    # Resize the image using Pillow
    img = Image.fromarray(img)
    img = img.resize((28, 28)) #, Image.ANTIALIAS)
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)

    # with graph.as_default():
        # out = model.predict(img)
        # print(out)
        # print(np.argmax(out, axis=1))

        # response = np.array_str(np.argmax(out, axis=1))
        # return response

    out = model.predict(img)
    print(out)
    print(np.argmax(out, axis=1))

    response = np.array_str(np.argmax(out, axis=1))
    return response




if __name__ == '__main__':
    app.run(debug=True, port=8000)