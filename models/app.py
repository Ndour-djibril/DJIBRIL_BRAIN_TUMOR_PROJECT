import torch
import tensorflow as tf
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template
import io
import os
import numpy as np

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Define PyTorch model
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 4)  # Four classes: glioma, meningioma, notumor, pituitary
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorch_model = BrainTumorCNN().to(device)
try:
    pytorch_model.load_state_dict(torch.load('DJIBRIL_model.torch', map_location=device))
    pytorch_model.eval()
except FileNotFoundError:
    print("Error: PyTorch model file 'DJIBRIL_model.torch' not found. Please train the model first.")
    exit(1)
tensorflow_model = tf.keras.models.load_model('DJIBRIL_model.tensorflow', compile=False)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(image):
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        model_type = request.form.get('model')
        file = request.files.get('image')
        if file and model_type:
            image = preprocess_image(file)
            if model_type == 'pytorch':
                with torch.no_grad():
                    image = image.to(device)
                    output = pytorch_model(image)
                    _, predicted = torch.max(output, 1)
                    prediction = class_names[predicted.item()]
            else:  # tensorflow
                image = image.numpy()
                image = image.transpose(0, 2, 3, 1)  # Reshape for TensorFlow
                prediction_probs = tensorflow_model.predict(image, verbose=0)
                prediction = class_names[np.argmax(prediction_probs)]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

