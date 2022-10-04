import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('Brain_Tumor_Detection10epochCategorical.h5')

image = cv2.imread(
    'C:\\Users\\91854\\Desktop\\Brain tumour dataset\\pred\\pred23.jpg')

img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)
predict_img = model.predict(input_img)
classes_img = np.argmax(predict_img, axis=1)
print(classes_img)
