from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str.split(",")[1])
    img = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

def generate_mask(user_masked_image):
    """Generate a mask based on the provided user-masked image."""
    # Get the shape of the image
    height, width = user_masked_image.shape[:2]

    # Convert all pixels greater than zero to black, and black to white
    for i in range(height):
        for j in range(width):
            if user_masked_image[i, j].sum() > 0:
                user_masked_image[i, j] = [0, 0, 0]
            else:
                user_masked_image[i, j] = [255, 255, 255]
    
    return user_masked_image

@app.route('/inpaint', methods=['POST'])
def inpaint_image():
    data = request.get_json()
    original = base64_to_image(data['original'])
    user_masked_image = base64_to_image(data['mask'])

    # Generate mask using the custom logic
    python_generated_mask = generate_mask(user_masked_image)

    # Convert to grayscale for inpainting
    binary_mask = cv2.cvtColor(python_generated_mask, cv2.COLOR_BGR2GRAY)

    # Inpaint the image
    inpainted = cv2.inpaint(original, binary_mask, 3, cv2.INPAINT_NS)

    # Convert all images to base64
    response = {
        'originalImage': data['original'],
        'userMaskedImage': data['mask'],
        'pythonGeneratedMask': image_to_base64(python_generated_mask),
        'inpaintedImage': image_to_base64(inpainted),
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
