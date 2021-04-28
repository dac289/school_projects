import os
import pandas as pd
import numpy as np

import pydicom
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

def predictions():
    test_dir = "stage_2_test_images/"
    test_list = os.listdir(test_dir)

    model = load_model("pneumonia_model.h5")

    predict_list = []
    for img_name in test_list:
        dicom_img = pydicom.dcmread(test_dir + img_name)
        img = dicom_img.pixel_array.astype(float)

        rescale_image = (np.maximum(img, 0) / img.max()) * 255
        image_rescaled = np.uint8(rescale_image)

        image = Image.fromarray(image_rescaled)
        NEW_SIZE = (256, 256)
        final_img = image.resize(NEW_SIZE)
        test_arr = np.array(final_img)
        test_arr = test_arr.reshape(-1,256,256,1)
        prediction = model.predict(test_arr)
        predict_list.append(int(prediction))

    predict_df = pd.DataFrame(test_list, columns=['patientId'])
    predict_df['PredictionString'] = predict_list
    print(predict_df)
    predict_df.to_csv('chest_xray_pneumonia_predictions.csv')

if __name__=="__main__":
    predictions()