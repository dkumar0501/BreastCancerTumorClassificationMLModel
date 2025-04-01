import keras
from PIL import Image, ImageOps
import numpy as np

def teachable_machine_classification(img, weights_file):
    try:
        # Load the model
        model = keras.models.load_model(weights_file)

        # Define image size
        size = (224, 224)

        # Resize image while maintaining aspect ratio
        image = ImageOps.fit(img, size, Image.LANCZOS)

        # Convert image to NumPy array
        image_array = np.asarray(image, dtype=np.float32)

        # Normalize the image (scale pixel values between -1 and 1)
        normalized_image_array = (image_array / 127.5) - 1

        # Ensure correct input shape
        data = np.expand_dims(normalized_image_array, axis=0)

        # Run inference
        prediction = model.predict(data)

        # Return the index of the highest probability
        return int(np.argmax(prediction))

    except Exception as e:
        print(f"Error: {e}")
        return None
