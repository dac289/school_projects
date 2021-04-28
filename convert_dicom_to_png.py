import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image

def main():
    df = pd.read_csv('stage_2_train_labels.csv')
    image_dir = 'stage_2_train_images/'
    normal = df[df['Target'] == 0]
    pneumonia = df[df['Target'] == 1]

    if not os.path.exists('cleaned_train_images'):
        os.mkdir('cleaned_train_images')
    if not os.path.exists(('cleaned_train_images/normal/')):
        os.mkdir('cleaned_train_images/normal/')
    if not os.path.exists('cleaned_train_images/pneumonia/'):
        os.mkdir('cleaned_train_images/pneumonia/')

    for x in normal.patientId:
        img = pydicom.dcmread(image_dir + x + '.dcm')
        img = img.pixel_array.astype(float)

        rescale_image = (np.maximum(img,0)/img.max())*255
        image_rescaled = np.uint8(rescale_image)

        image = Image.fromarray(image_rescaled)
        NEW_SIZE = (256,256)
        image_new_size = image.resize(NEW_SIZE)
        image_info = np.array(image_new_size)
        final_image = Image.fromarray(image_info)
        final_image.save('cleaned_train_images/normal/' + x + '.png')

    for x in pneumonia.patientId:
        img = pydicom.dcmread(image_dir + x + '.dcm')
        img = img.pixel_array.astype(float)

        rescale_image = (np.maximum(img,0)/img.max())*255
        image_rescaled = np.uint8(rescale_image)

        image = Image.fromarray(image_rescaled)
        NEW_SIZE = (256,256)
        image_new_size = image.resize(NEW_SIZE)
        image_info = np.array(image_new_size)
        final_image = Image.fromarray(image_info)
        final_image.save('cleaned_train_images/pneumonia/' + x + '.png')

if __name__=='__main__':
    main()

    print("-------DONE-------")
