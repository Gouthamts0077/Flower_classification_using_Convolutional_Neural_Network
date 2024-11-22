import warnings
import os
import numpy as np
import random as rn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
rn.seed(42)
import tensorflow as tf

tf.random.set_seed(42)

# Define constants
IMG_SIZE = 150
FLOWER_CATEGORIES = {
    "Daisy": r"data/daisy",
    "Sunflower": r"data/sunflower",
    "Tulip": r"data/tulip",
    "Dandelion": r"data/dandelion",
    "Rose": r"data/rose",
}


# Function to prepare training data
def make_train_data(label, directory, X, Z):
    for img_file in os.listdir(directory):
        try:
            img_path = os.path.join(directory, img_file)
            img = plt.imread(img_path)
            img_resized = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()
            X.append(img_resized)
            Z.append(label)
        except Exception as e:
            print(f"Error processing file {img_file}: {e}")


# Prepare the dataset
def load_data():
    X, Z = [], []
    for label, directory in FLOWER_CATEGORIES.items():
        make_train_data(label, directory, X, Z)
    return np.array(X) / 255.0, Z


# Build the CNN model
def build_model():
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (5, 5),
            padding="Same",
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding="Same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(96, (3, 3), padding="Same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(96, (3, 3), padding="Same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(5, activation="softmax"))
    return model


# Plot training results
def plot_results(history):
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.legend()
    plt.show()

    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.show()


# Main function
def main():
    X, Z = load_data()
    le = LabelEncoder()
    Y = le.fit_transform(Z)
    Y = to_categorical(Y, 5)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42
    )

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=5,
        validation_data=(x_test, y_test),
        steps_per_epoch=x_train.shape[0] // 128,
    )

    plot_results(history)


if __name__ == "__main__":
    main()
