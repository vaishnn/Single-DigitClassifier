from cv2 import imread,resize,cvtColor,COLOR_BGR2GRAY
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#loading the saved model
model = tf.keras.models.load_model("digit_prediction.tf")
def what_number_is_this(path):
    #converting image to grayscale
    image = cvtColor(imread(path),COLOR_BGR2GRAY)
    #rescaling image 
    reszed_image = resize(image,[28,28])
    #reshaping image
    resized_image = np.array(reszed_image).reshape(1,28,28,1)
    #predicting the value
    Val = model.predict(resized_image)
    print(int(np.where(Val==1)[1]))
    plt.imshow(reszed_image)