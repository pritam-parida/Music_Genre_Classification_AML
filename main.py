import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import cv2
import librosa
import librosa.feature
import librosa.display
from sklearn.metrics import roc_curve, auc
from keras.optimizers import Adam


# Function to load and preprocess image data with optional augmentation
def load_and_preprocess_image(file_path, target_size=(128, 128), augment=False):
    image = Image.open(file_path)
    image = image.resize(target_size)  # Resize image to target size
    image_array = np.asarray(image)
    # Check and convert image to RGB if necessary (remove alpha channel)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]  # Keep only RGB channels
    # Normalize pixel values to range [0, 1]
    image_array = image_array.astype('float32') / 255.0
    # Apply augmentation if enabled
    if augment:
        # Example augmentation: horizontal flip
        image_array = cv2.flip(image_array, 1)  # Horizontal flip
    return image_array


# Function to load image data from directory with optional data augmentation
def load_image_data(data_dir, augment=False, target_size=(128, 128)):
    X = []
    y = []
    genres = os.listdir(data_dir)
    for genre_idx, genre in enumerate(genres):
        genre_dir = os.path.join(data_dir, genre)
        for filename in os.listdir(genre_dir):
            if filename.endswith('.png'):
                file_path = os.path.join(genre_dir, filename)
                image_array = load_and_preprocess_image(file_path, target_size=target_size, augment=augment)
                X.append(image_array)
                y.append(genre_idx)
                # Optionally append augmented image
                if augment:
                    augmented_image = load_and_preprocess_image(file_path, target_size=target_size, augment=True)
                    X.append(augmented_image)
                    y.append(genre_idx)
    return np.array(X), np.array(y)

# Function to preprocess audio and make prediction
def predict_music_genre(audio_file_path, model, class_names, target_size=(128, 128)):
    # Load audio file and extract features (e.g., Mel spectrogram)
    y, sr = librosa.load(audio_file_path, duration=30)  # Load audio (30 seconds)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to decibels
    # Resize spectrogram to target size and convert to RGB
    image = Image.fromarray(spectrogram_db)
    image = image.resize(target_size)
    image_array = np.asarray(image)
    image_array = np.stack((image_array,) * 3, axis=-1)  # Convert to RGB
    # Normalize pixel values
    image_array = image_array.astype('float32') / 255.0
    # Make prediction
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    predicted_genre_idx = np.argmax(prediction)
    predicted_genre = class_names[predicted_genre_idx]
    return predicted_genre, prediction[0]

# Define directory paths
image_data_dir = 'Data/images_original'

# Load and preprocess image data with optional data augmentation and specific target size
target_size = (128, 128)
X_images, y_images = load_image_data(image_data_dir, augment=True, target_size=target_size)

# Split image data into train/validation/test sets
X_train_images, X_test_images, y_train_images, y_test_images = train_test_split(X_images, y_images, test_size=0.2,
                                                                                random_state=42)

# Define the number of genres and class names
num_genres = len(np.unique(y_images))
class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Define and train a more complex CNN model
image_model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_genres, activation='softmax')
])

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
image_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with more epochs and smaller batch size
history = image_model.fit(
    X_train_images, y_train_images, epochs=30, batch_size=64, validation_data=(X_test_images, y_test_images)
)

# Evaluate the model on the test set
test_loss, test_accuracy = image_model.evaluate(X_test_images, y_test_images)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Example usage to predict music genre from audio file
audio_file_path = 'Carry On Wayword Son.wav'  # Specify the path to your audio file
predicted_genre, probabilities = predict_music_genre(audio_file_path, image_model, class_names)

print("Predicted Genre:", predicted_genre)
print("Probabilities:", probabilities)

# Get predicted probabilities for the test set
y_score = image_model.predict(X_test_images)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_genres):
    fpr[i], tpr[i], _ = roc_curve(y_test_images == i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = plt.cm.get_cmap('tab10', num_genres)

for i in range(num_genres):
    plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
             label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Plot diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
