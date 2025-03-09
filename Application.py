import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16, DenseNet121
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


# Display text
st.title("🩸 Blood Cell Classification")
st.write("This is a simple web app to classify blood cells using Convolutional Neural Networks (CNN).")
st.title("📚 Dataset")
st.markdown("The dataset is available on Kaggle and can be downloaded from the following link: [Blood Cell Classification Dataset](https://www.kaggle.com/paultimothymooney/blood-cells)")
st.code("https://www.kaggle.com/paultimothymooney/blood-cells", language="python")
st.title("🔮 Predicted class and confidence")
# Define class labels
class_labels =  ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]   # Modify as needed

# Load dataset
def load_data(train_dir, test_dir, batch_size=32, target_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical')
    test_generator = datagen.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical')
    return train_generator, test_generator

train_dir = "C:\AMIT\PROJ\Blood_Cell_Classification\Data\TRAIN" 
test_dir = "C:\AMIT\PROJ\Blood_Cell_Classification\Data\TEST"  
train_generator, test_generator = load_data(train_dir, test_dir, batch_size=32)

# create model
def create_model(base_model, preprocess_input):
    inputs = Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(len(class_labels), activation='softmax')(x)  # Adjust class count
    model = Model(inputs, outputs)
    return model

# Load model
def load_model(model_name):
    if model_name == "BaseModel":
        # Load custom trained model
        model = tf.keras.models.load_model('C:\\AMIT\\PROJ\\Blood_Cell_Classification\\blood_cell_model.h5')
        # Use appropriate preprocessing (modify if you used different preprocessing)
        preprocess_input = resnet_preprocess  
        return model, preprocess_input
    
    base_model = None
    preprocess_input = None
    
    if model_name == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        preprocess_input = resnet_preprocess
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        preprocess_input = efficientnet_preprocess
    elif model_name == "VGG16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        preprocess_input = vgg_preprocess
    elif model_name == "DenseNet121":
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        preprocess_input = densenet_preprocess
    
    model = create_model(base_model, preprocess_input)
    return model, preprocess_input

# Streamlit UI
st.sidebar.header("🔄 Select Model")
model_choice = st.sidebar.selectbox("Choose a Pre-trained Model", ["BaseModel", "ResNet50", "EfficientNetB0", "VGG16", "DenseNet121"])
model, preprocess_input = load_model(model_choice)

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to classify image
def classify_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return class_labels[class_index], confidence

# Function to overlay label on image
def add_label_to_image(image, label):
    img_pil = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    draw.text((10, 10), label, fill=(255, 0, 0), font=font)
    return img_pil

uploaded_file = st.file_uploader("Upload a blood cell image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify Image"):
        predicted_class, confidence = classify_image(image)
        labeled_image = add_label_to_image(image, f"{predicted_class} ({confidence:.2f}%)")
        st.image(labeled_image, caption="Predicted Image", use_column_width=True)
        st.write(f"### Predicted Class: {predicted_class}")
        st.write(f"### Confidence: {confidence:.2f}%")

# Function to resize image
def apply_resizing(image, target_size=(224, 224)):
    return cv2.resize(image, target_size)

# Function to normalize pixel intensity
def apply_normalization(image):
    return image / 255.0  # Normalize pixel values to range [0, 1]

# Function to rotate image
def apply_rotation(image, angle):
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

# Function to translate image
def apply_translation(image, tx, ty):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

# Function to flip image
def apply_flipping(image, flip_horizontal, flip_vertical):
    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)
    return image

# Function to zoom image
def apply_zoom(image, zoom_factor):
    if zoom_factor == 1.0:
        return image
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    x1, y1 = max(center_x - new_w // 2, 0), max(center_y - new_h // 2, 0)
    x2, y2 = min(center_x + new_w // 2, w), min(center_y + new_h // 2, h)
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))

# Streamlit UI
st.title("🖼️ Image Preprocessing")
st.write("Upload an image and adjust the sliders to apply transformations.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    image = np.array(Image.open(uploaded_file)).astype(np.float32)

    # Convert grayscale images to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Resize image to uniform dimension
    image = apply_resizing(image)

    # Normalize image
    image = apply_normalization(image)

    # Sidebar Controls
    st.sidebar.header("🔄 Image Transformations")
    angle = st.sidebar.slider("Rotate Image (°)", -180, 180, 0)
    tx = st.sidebar.slider("Translate X", -100, 100, 0)
    ty = st.sidebar.slider("Translate Y", -100, 100, 0)
    zoom_factor = st.sidebar.slider("Zoom Factor", 1.0, 2.0, 1.0, 0.1)
    flip_horizontal = st.sidebar.checkbox("Flip Horizontally", False)
    flip_vertical = st.sidebar.checkbox("Flip Vertically", False)

    # Reset button
    if st.sidebar.button("Reset Transformations"):
        angle, tx, ty, zoom_factor, flip_horizontal, flip_vertical = 0, 0, 0, 1.0, False, False

    # Apply transformations
    transformed_image = apply_rotation(image, angle)
    transformed_image = apply_translation(transformed_image, tx, ty)
    transformed_image = apply_flipping(transformed_image, flip_horizontal, flip_vertical)
    transformed_image = apply_zoom(transformed_image, zoom_factor)

    # Display Images
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Resized & Normalized Image", use_column_width=True)
    
    with col2:
        st.image(transformed_image, caption="Transformed Image", use_column_width=True)

st.write("👆 Upload an image and modify rotation, translation, flipping, or zoom using the sliders!")
st.write("**Developed by:** Mark Wael ❤️")