from PIL import Image
import numpy as np

# Load the image
image = Image.open('TestImg.jpg')

# Resize the image to 28x28 pixels
image = image.resize((28, 28))

# Convert to grayscale
image = image.convert('L')

# Flatten the image to a 1D array
image = np.array(image).flatten()

# Normalize pixel values
image = image / 255.0

print(image.shape)  # Output: (784,)


import matplotlib.pyplot as plt

# Reshape the flattened image back to 28x28
image = image.reshape((28, 28))

# Visualize the converted image
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
