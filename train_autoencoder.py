import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
train_dir = "F:/RP/Intern/Internship/Data1/Train"  # Directory containing normal X-ray images

def load_images_from_directory(directory, img_size=(256, 256)):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
    return np.array(images)

train_images = load_images_from_directory(train_dir)
train_images = train_images / 255.0  # Normalize pixel values
train_images = np.expand_dims(train_images, axis=-1)

def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

autoencoder = build_autoencoder(train_images.shape[1:])
autoencoder.summary()

autoencoder.fit(train_images, train_images,
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_split=0.1)

# Save the trained model
autoencoder.save('autoencoder_model.h5')
