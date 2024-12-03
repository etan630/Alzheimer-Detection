from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import pathlib
    
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('cnn.keras')

DATASET_PATH = "../Data/"
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.3
IMAGE_SIZE = (248, 496)
data_dir = pathlib.Path(DATASET_PATH)
classes = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        labels="inferred",
        directory=data_dir,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=0,
        image_size=IMAGE_SIZE,
        subset='both',
        color_mode="grayscale")

for images, labels in train_ds.take(1):
    image = images[0]
    
    print(model.predict(image))
    break
    
    layer_name = 'conv2d_2'  
    last_conv_layer = model.get_layer(layer_name)
    
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(images)
        class_idx = np.argmax(predictions[0])  # Index of the predicted class
        loss = predictions[:, class_idx]

        # Compute gradients of the class score w.r.t. the feature map
        grads = tape.gradient(loss, conv_outputs)

        # Compute the mean intensity of the gradients over the feature map
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each channel by "how important it is" with respect to the predicted class
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)  # Normalize to [0, 1]

        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # Apply the heatmap to the input image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(images[0].astype('uint8'), 0.6, heatmap, 0.4, 0)
        
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.show()