"""
Practical no 10: Study of classification of Dog and Cat Images

*Dataset Info: Over 1000 images of cats and dogs scraped off of Google images.
*The problem statement is to build a model that can classify between a cat and a dog in an image as accurately as possible.
*Image sizes range from roughly 100x100 pixels to 2000x1000 pixels.
*Image format is jpeg.
*Duplicates have been removed.

*Dataset link: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification?resource=download
"""

#Step I: Import necessary libraries/dependencies.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

import os
import PIL
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

try:
    base_dir = "archive/Dog_Cat_Images"
    train_images = os.path.join(base_dir, "train")
    test_images = os.path.join(base_dir, "test")

except Exception as e:
    print(f"Error: I can't find any dataset containing those images. Please specify the correct path")
    print(f"Or Extract the downloaded ZIP and place it in archive/")
    exit()

#Define Training Hyper-Parameters
BATCH_SIZE = 32
IMG_WIDTH = 180
IMG_HEIGHT = 180
NUM_EPOCHS = 30

#Step II: Loading Dataset and validation split
#Load the Dataset using keras.utils API
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_images,
    validation_split=.2,
    subset="training",
    seed=123,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_images,
    validation_split=.2,
    subset="validation",
    seed=111,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_images,
    seed=111,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE
)

#Get the class names
class_names = train_dataset.class_names
print(f"Class Names:\n", class_names)
print(f"'{class_names[0]}' will be mapped to 0 and '{class_names[1]}' will be mapped to 1")

num_cats_train = len(os.listdir(os.path.join(train_images, 'cats')))
num_dogs_train = len(os.listdir(os.path.join(train_images, 'dogs')))
num_cats_test = len(os.listdir(os.path.join(test_images, 'cats')))
num_dogs_test = len(os.listdir(os.path.join(test_images, 'dogs')))

print(f"\n--- Dataset Analysis ---")
print(f"Total training images for cats: {num_cats_train}")
print(f"Total training images for dogs: {num_dogs_train}")
print(f"Total validation/test images for cats: {num_cats_test}")
print(f"Total validation/test images for dogs: {num_dogs_test}")
print("The dataset appears to be well-balanced between the two classes.")

#Step III: Data Analysis & Visualization
#Display sample data
plt.figure(figsize=(10, 10))
plt.suptitle("Sample Images from training set", fontsize=18)
for images, labels in train_dataset.take(1): #For a single batch
    for idx in range(25):
        ax = plt.subplot(5, 5, idx + 1)
        plt.imshow(images[idx].numpy().astype(np.uint8))
        plt.title(class_names[labels[idx]])
        plt.axis("off")
plt.show()

#Step IV-A: Configure Dataset for Performance
AutoTune = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(501).prefetch(buffer_size=AutoTune)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AutoTune)
test_dataset = test_dataset.cache().shuffle(140).prefetch(buffer_size=AutoTune)

#Step IV-B: Data Augmentation
data_augmentation_layer = Sequential(
    [
        RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        RandomRotation(.15),
        RandomZoom(.12),
        RandomSaturation(.2),
        RandomGaussianBlur(.3),
        RandomContrast(.25)
    ]
)

#Step V: Model Training
#V-A: Define CNN Architecture

model = Sequential([
    data_augmentation_layer,
    Rescaling(1./255),

    Conv2D(16, 3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(32, 3,  padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, 3,  padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, 3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Dropout(.3),
    Flatten(),

    Dense(256, "relu"),
    Dense(64),
    LeakyReLU(.2),
    Dense(1, "sigmoid")

])

#V-B: Model Compilation
model.compile(keras.optimizers.Adam(0.0005), "binary_crossentropy", metrics=["acc"])

print("---Model Architecture Summary---\n")
model.summary()
c1 = keras.callbacks.TensorBoard("./logs", histogram_freq=2, write_images=True)
c2 = keras.callbacks.ModelCheckpoint("./weights/model.weights.h5",mode="max", save_best_only=True, save_weights_only=True)
#V-C: Train the model
print("\n---Started Model Training---\n")
history = model.fit(train_dataset, validation_data = validation_dataset, epochs=NUM_EPOCHS, callbacks=[c1, c2])
print("--- Model Training Completed ---")

#Step VI: Model Accuracy and Loss Chart
accuracy = history.history['acc']
validation_accuracy = history.history['val_acc']

loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(NUM_EPOCHS)
# Plot Train vs Val dataset accuracy
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(epochs_range, accuracy, color="green", linestyle="--", label="Training Accuracy", alpha=.75, marker="o")
plt.plot(epochs_range, validation_accuracy,color="orange" ,label="Validation Accuracy", linestyle="--", marker="o", alpha=.75)
plt.legend(loc="lower right")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.style.use("ggplot")
plt.grid(True)

#Plot Train vs Val dataset loss
plt.subplot(122)
plt.plot(epochs_range, loss, color="red",linestyle="--", label="Training Loss", alpha=.75, marker="o")
plt.plot(epochs_range, validation_loss,color="orange" ,label="Validation Loss", linestyle="--", marker="o", alpha=.75)
plt.legend(loc="upper right")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.suptitle(f"Model Learning Curve", fontsize=16)
plt.style.use("ggplot")
plt.tight_layout()
plt.style.use('ggplot')
plt.grid(True)

plt.show()

# Analysis of Learning Curves:
# - If training accuracy is high but validation accuracy is low, it indicates overfitting.
# - If both curves track each other closely, the model is generalizing well.
# - If both are low, the model might be underfitting (not powerful enough).

#Step VII: Model Evaluation
print("\n--- Evaluating Model on Unseen Test Data ---")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

#VII-A: Generate Predictions for Detailed Analysis
# We need to get the true labels and predicted labels for the entire test set.
y_pred_prob = model.predict(test_dataset)
y_pred = (y_pred_prob >= 0.5).astype("int32").flatten()  # Convert probabilities to 0 or 1
y_true = np.concatenate([y for x, y in test_dataset], axis=0)  # Extract true labels

#VII-B: Confusion Matrix
# A confusion matrix gives a detailed breakdown of correct and incorrect classifications.
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6), facecolor="black", edgecolor="white")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.style.use("ggplot")
plt.show()

# Analysis of Confusion Matrix:
# - Top-left: True Negatives (Correctly classified as 'cat')
# - Bottom-right: True Positives (Correctly classified as 'dog')
# - Top-right: False Positives (Incorrectly classified as 'dog')
# - Bottom-left: False Negatives (Incorrectly classified as 'cat')


#VII-C: Classification Report
# Provides key metrics like precision, recall, and f1-score for each class.
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# Interpretation of Metrics:
# - Precision: Of all images predicted as a class, how many were correct?
# - Recall: Of all true images of a class, how many did the model identify?
# - F1-score: The harmonic mean of precision and recall.


#Step VIII: Visualizing Predictions: Analyze Successes and Failures
# This is crucial for understanding the model's biases and where it struggles.
print("\n--- Visualizing Model Predictions on Test Images ---")
plt.figure(figsize=(15, 15))
plt.suptitle("Analysis of Model Predictions", fontsize=20)
# Get a batch of test images and labels
for images, labels in test_dataset.take(1):
    for i in range(16):  # Show 16 predictions
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype(np.uint8))
        # Get the model's prediction for this image
        img_array = tf.expand_dims(images[i], 0)  # Create a batch
        prediction_prob = model.predict(img_array)[0][0]
        prediction_label_index = 1 if prediction_prob >= 0.5 else 0
        predicted_class = class_names[prediction_label_index]
        true_class = class_names[labels[i]]

        # Set title color based on correctness
        title_color = 'green' if predicted_class == true_class else 'red'

        plt.title(f"True: {true_class}\nPred: {predicted_class} ({prediction_prob:.2f})", color=title_color)
        plt.axis("off")

plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
plt.show()