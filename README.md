# 🐱🐶 Cat vs Dog Classification Using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** to classify images as either **Cat** or **Dog**. The model learns visual features such as edges, shapes, textures, and patterns directly from training images.

## 📌 Project Overview

Image classification is one of the fundamental applications of Computer Vision and Deep Learning.

In this project, a CNN model is trained on a dataset containing cat and dog images. The trained model can analyze a new image and predict whether it belongs to the **Cat** or **Dog** class.

### 🎯 Objectives

* Build an image classification model using CNN.
* Preprocess and normalize image data.
* Train the model using labeled Cat and Dog images.
* Evaluate model performance using validation data.
* Predict the class of unseen images.
* Visualize training and validation performance.

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **Convolutional Neural Network (CNN)**
* **NumPy**
* **Matplotlib**
* **OpenCV / PIL**
* **Jupyter Notebook / VS Code**

---

## 📂 Project Structure

```text
Cat-Dog-Classification/
│
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   │
│   └── test/
│       ├── cats/
│       └── dogs/
│
├── models/
│   └── cat_dog_cnn.h5
│
├── notebooks/
│   └── cat_dog_classification.ipynb
│
├── images/
│   └── sample_predictions/
│
├── predict.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 🧠 How CNN Works in This Project

The CNN automatically extracts important visual features from the images.

```text
Input Image
     ↓
Image Preprocessing
     ↓
Convolution Layer
     ↓
ReLU Activation
     ↓
Max Pooling
     ↓
Convolution Layer
     ↓
Max Pooling
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Sigmoid Activation
     ↓
Cat / Dog Prediction
```

### Main CNN Components

**1. Convolution**

Extracts important features such as edges, textures, and shapes from images.

**2. ReLU Activation**

Introduces non-linearity into the network and helps the model learn complex patterns.

**3. Max Pooling**

Reduces the spatial dimensions of feature maps while retaining important information.

**4. Flatten**

Converts the extracted feature maps into a one-dimensional vector.

**5. Dense Layer**

Uses the extracted features to perform classification.

**6. Sigmoid**

Produces a probability for binary classification.

---

## 🔄 Data Preprocessing

Before training, the images are processed to make them suitable for the CNN.

Typical preprocessing steps include:

* Resize images to a fixed size.
* Convert images into numerical arrays.
* Normalize pixel values.
* Assign labels to Cat and Dog classes.
* Split the dataset into training and validation sets.

Example:

```python
IMG_SIZE = (150, 150)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```

---

## 🏗️ CNN Model Architecture

Example CNN architecture used for the project:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])
```

---

## ⚙️ Model Compilation

Since this is a binary classification problem, **Binary Cross-Entropy** can be used as the loss function.

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## 🚀 Model Training

The model is trained using the prepared training dataset.

```python
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=20
)
```

The training process allows the CNN to learn the visual characteristics that distinguish cats from dogs.

---

## 📊 Model Evaluation

The model can be evaluated using:

* Accuracy
* Loss
* Confusion Matrix
* Precision
* Recall
* F1-Score

Training and validation curves can also be plotted to identify overfitting or underfitting.

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])
plt.show()
```

---

## 🔮 Making Predictions

After training, the saved model can be used to classify a new image.

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img(
    "sample.jpg",
    target_size=(150, 150)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
```

---

## 💾 Saving the Model

```python
model.save("cat_dog_cnn.h5")
```

The saved model can later be loaded for prediction:

```python
from tensorflow.keras.models import load_model

model = load_model("cat_dog_cnn.h5")
```

---

## 📈 Results

The trained CNN is capable of distinguishing between Cat and Dog images based on learned visual features.

### Evaluation Metrics

| Metric              | Result          |
| ------------------- | --------------- |
| Training Accuracy   | Add your result |
| Validation Accuracy | Add your result |
| Test Accuracy       | Add your result |
| Loss                | Add your result |

> Replace the placeholder values with the actual results obtained during training.

---

## ⚠️ Challenges

Some challenges encountered during the project include:

* Different image sizes and resolutions.
* Variations in lighting and backgrounds.
* Different poses and orientations of animals.
* Preventing overfitting.
* Improving generalization to unseen images.

---

## 💡 Improvements

The model can be improved using:

* Data augmentation.
* Batch normalization.
* Dropout.
* Learning-rate scheduling.
* Transfer learning.
* Pre-trained models such as **VGG16, ResNet50, MobileNet, or EfficientNet**.

Example data augmentation:

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Cat-Dog-Classification.git
```

### 2. Navigate to the Project

```bash
cd Cat-Dog-Classification
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Model

```bash
python train.py
```

### 6. Predict an Image

```bash
python predict.py
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
tensorflow
numpy
matplotlib
opencv-python
pillow
scikit-learn
```

---

## 🔮 Future Scope

The project can be extended into a complete **AI-based pet recognition application** by adding:

* Web-based image upload.
* Real-time camera classification.
* Mobile application integration.
* Transfer learning for improved accuracy.
* Multi-class animal classification.
* Cloud deployment using AWS.
* REST API for model predictions.

---

## 🌟 Key Learning Outcomes

Through this project, I learned:

* Fundamentals of CNN architecture.
* Image preprocessing and normalization.
* Binary image classification.
* Model training and validation.
* Overfitting prevention.
* Model evaluation.
* Saving and loading deep learning models.
* Using TensorFlow and Keras for Computer Vision.

---

## 👩‍💻 Author

**Arathi S B**

B.E. Computer Science & Engineering
Specialization: AI & Data Science

### ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub!
