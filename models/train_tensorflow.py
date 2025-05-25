import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import os
import matplotlib.pyplot as plt

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define CNN architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Four classes: glioma, meningioma, notumor, pituitary
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_metrics(history, model_name):
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name}_loss_accuracy_plot.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['pytorch', 'tensorflow'], required=True)
    args = parser.parse_args()

    if args.model != 'tensorflow':
        print("This script is for TensorFlow. Use train_pytorch.py for PyTorch.")
        return

    # Data preprocessing and augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        preprocessing_function=lambda x: (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    )
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=lambda x: (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    )

    # Load dataset
    dataset_path = "/home/students-asn34/Documents/DJIBRIL_BRAIN_TUMOR_PROJECT/breast_cancer/breast_cancer"
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'training'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'testing'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Train model
    model = create_model()
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=test_generator,
        verbose=1
    )

    # Plot and save metrics
    plot_metrics(history, 'tensorflow')

    # Print final accuracies
    final_train_accuracy = history.history['accuracy'][-1] * 100
    final_val_accuracy = history.history['val_accuracy'][-1] * 100
    print(f"Final Train Accuracy: {final_train_accuracy:.2f}%")
    print(f"Final Validation Accuracy: {final_val_accuracy:.2f}%")

    # Save model
    model.save('DJIBRIL_model.tensorflow')
    print("TensorFlow model saved as DJIBRIL_model.tensorflow")

if __name__ == "__main__":
    main()