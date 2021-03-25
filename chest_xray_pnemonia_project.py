import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_io as tfio

def get_data(filename):
    image_bytes = tf.io.read_file('stage_2_train_images/0a0f91dc-6015-4342-b809-d19610854a21.dcm')
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)

    skipped = tfio.image.decode_dicom_image(image_bytes, on_error='skip', dtype=tf.uint8)

    lossy_image = tfio.image.decode_dicom_image(image_bytes, scale='auto', on_error='lossy', dtype=tf.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(np.squeeze(image.numpy()), cmap='gray')
    axes[0].set_title('image')
    axes[1].imshow(np.squeeze(lossy_image.numpy()), cmap='gray')
    axes[1].set_title('lossy image');
    plt.show()

def main():
    filename = 0
    get_data(filename)


if __name__=='__main__':
    main()