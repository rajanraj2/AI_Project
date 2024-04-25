from keras.models import load_model
loaded_model = load_model('LD_cnn.h5')



import os
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

img_path = r"D:\AI\light_dark_cnn\test\907.jpg"

img = image.load_img(img_path, target_size=(200, 200))
plt.imshow(img)
plt.show()
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)
result = loaded_model.predict(X)
prediction = np.argmax(result) 
    
if prediction == 0:
    print("Dark")
else:
    print('Light')

