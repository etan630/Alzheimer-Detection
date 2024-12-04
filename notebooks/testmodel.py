from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import pathlib
    
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('cnn.keras')

DATASET_PATH = "../Data/"
IMAGE_PATH = "Very mild Dementia/OAS1_0003_MR1_mpr-3_158.jpg"
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.3
IMAGE_SIZE = (248, 496)
data_dir = pathlib.Path(DATASET_PATH)
classes = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]

def compute_gradcam(model, image_array, last_conv_layer_name, class_idx=None):

    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        conv_outputs, predictions = grad_model(image_tensor)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_outputs)

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    gradcam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)
    gradcam = tf.maximum(gradcam, 0)  # Apply ReLU
    gradcam /= tf.reduce_max(gradcam)  # Normalize to [0, 1]

    return gradcam.numpy()



img = tf.keras.utils.load_img(DATASET_PATH + IMAGE_PATH, target_size=IMAGE_SIZE, color_mode="grayscale")
image = tf.keras.preprocessing.image.img_to_array(img)
image = np.expand_dims(image, axis=0)
# print(model.predict(image))

layer_name = 'conv2d_2'  

heatmap = compute_gradcam(model, image, layer_name)

image_original = np.array(image) / 255.0  

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_original.squeeze(), cmap='gray')  
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(heatmap, cmap='viridis')
plt.axis("off")

plt.show()