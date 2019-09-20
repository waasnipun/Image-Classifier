import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing import image

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

new_model = keras.models.load_model('Trained_model.h5')
#new_model.summary()
y_binary = to_categorical(y_test)
loss,acc = new_model.evaluate(x_test,y_binary)
print("Accuracy of the saved model: {:5.2f}%".format(100*acc))

class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#import image to this
file_name = input("Enter the image name :")
path = file_name+'.jpg'
img = image.load_img(path, target_size=(32, 32))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = new_model.predict(images, batch_size=10)
print(classes)
for i in range(len(class_names)):
    if classes[0][i]== 1:
        print(class_names[i])
        break