import sys
from keras.models import load_model
loaded_model = load_model('Tr_cnn.h5', custom_objects=None, compile=True)


import os
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# img_path = r"jeans.jpg"

# Get the image path from command-line arguments
image_path = sys.argv[1]

# Find the index of the last dot in the file path
last_dot_index = image_path.rfind('.')

# Remove all characters after the last dot (including the dot itself)
image_path = image_path[:last_dot_index]

# img_path = r"D:\GithubWindows\MERN\TT_MERN\server\jeans.jpg"

img = image.load_img(image_path, target_size=(200, 200))

# img = image.load_img(img_path, target_size=(200, 200))
plt.imshow(img)
# plt.show()
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)
result = loaded_model.predict(X)
prediction = np.argmax(result) 
    
if prediction == 0:
    print("Darksss")
else:
    print('Lightsss')

