import cv2
import numpy as np
from tensorflow import keras
import tensorflow

# Takes RGB Image,converts it to grayscale &
# then saves it to given path
def convert2gray(img_path, save_as, resize_to=None):
    """
    :param img_path: path of the image to convert
    :param save_as: filename to save the image as
    :param resize_to: tuple of order (width, height)
    :return: saves the image as grayscale with 1 channel
    """

    image = cv2.imread(img_path)
    print("Before conversion : ", image.shape)

    if resize_to != None:
        image = cv2.resize(image, resize_to)

    # converting to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("After conversion : ", gray_image.shape)

    # saving image
    cv2.imwrite(save_as, gray_image)


# Takes RGB image,resizes it,converts to grayscale
# image with 1 channel and returns it as numpy array
def convert2grayArray(img_path, resize_to=None):
    """
    :param img_path: path of the image to convert
    :param save_as: filename to save the image as
    :param resize_to: tuple of order (width, height)
    :return: the grayscale image as numpy array
    """

    image = cv2.imread(img_path)
    print("Before conversion : ", image.shape)

    if resize_to != None:
        image = cv2.resize(image, resize_to)

    # converting to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("After conversion : ", gray_image.shape)

    img_array = np.array(gray_image)
    return img_array


# ---------------------------------------------------------------------


# This is the main function which takes the path where the model is saved
# and returns the predicted age and gender

def get_gender_age(img_path, gender_model_path, age_model_path):
    """
    :param img_path: path of input image
    :param gender_model_path: path to gender prediction model
    :param age_model_path: path to age prediction model
    :return: [age:int,gender:probability value]
    """

    img = convert2grayArray(img_path, resize_to=(48, 48))

    # reshaping image to model input shape
    img = img.reshape(1, 48, 48, 1)

    # loading models
    print("loading models...")
    age_model = keras.models.load_model(age_model_path)
    gender_model = keras.models.load_model(gender_model_path)

    print("predicting gender and age....")
    predicted_age = age_model.predict(img)
    predicted_gender = gender_model.predict(img)

    age = predicted_age[0][0]
    gender = round(predicted_gender[0][0], 2)

    return [age, gender]



# ---------------------------------------------------------------------


if __name__=="__main__":

    print(tensorflow.__version__)

    age_model_path = r"C:\Users\dipesh\Desktop\Age-Gender Detection\models\Age_Prediction_model.h5"
    gender_model_path = r"C:\Users\dipesh\Desktop\Age-Gender Detection\models\Gender_Prediction_model.h5"
    img_path = r"C:\Users\dipesh\Desktop\women.jpg"

    prediction = get_gender_age(img_path,gender_model_path,age_model_path)
    print("Model Prediction : ",prediction)

    from tensorflow import keras

    age_model_path = r"C:\Users\dipesh\Desktop\Age-Gender Detection\models\Age_Prediction_model.h5"
    gender_model_path = r"C:\Users\dipesh\Desktop\Age-Gender Detection\models\Gender_Prediction_model.h5"

    age_model = keras.models.load_model(age_model_path)
    gender_model = keras.models.load_model(gender_model_path)


