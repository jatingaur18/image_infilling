from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

layers = 7

class PartialConvLayer(nn.Module):
  def __init__(self, in_channels, out_channels, bn = True, bias = False, sample = "none-3", activation = "relu"):
    super().__init__()
    self.bn = bn
    self.activation = activation

    if sample == "down-7":
      self.input_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias = bias)
      self.mask_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias = False)

    elif sample == "down-5":
      self.input_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias = bias)
      self.mask_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias = False)

    elif sample == "down-3":
      self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias = bias)
      self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias = False)

    else:
      self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
      self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

    nn.init.constant_(self.mask_conv.weight, 1.0)
    nn.init.kaiming_normal_(self.input_conv.weight, a = 0, mode = "fan_in")

    for param in self.mask_conv.parameters():
      param.requires_grad = False

    if bn:
      self.batch_normalization = nn.BatchNorm2d(out_channels)

    if activation == "relu":
      self.activation = nn.ReLU()
    elif activation == "leaky_relu":
      self.activation = nn.LeakyReLU(negative_slope = 0.2)

  def forward(self, input_x, mask):
    # Ensure input_x and mask are the same size
    if input_x.size() != mask.size():
      raise ValueError(f"Input tensor and mask must have the same size. Got {input_x.size()} and {mask.size()}")

    output = self.input_conv(input_x * mask)
    with torch.no_grad():
      output_mask = self.mask_conv(mask)

    if self.input_conv.bias is not None:
      output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
    else:
      output_bias = torch.zeros_like(output)

    mask_is_zero = (output_mask == 0)
    mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)

    output = (output - output_bias) / mask_sum + output_bias
    output = output.masked_fill(mask_is_zero, 0.0)

    new_mask = torch.ones_like(output)
    new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)

    if self.bn:
      output = self.batch_normalization(output)

    return output, new_mask

class PartialConvUNet(nn.Module):
  def __init__(self, input_size = 256, layer = 7):
    if 2**(layers + 1) != input_size:
      raise AssertionError

    super().__init__()
    self.freeze_enc_bn = False
    self.layers = layers

    self.encoder_1 = PartialConvLayer(3, 64, bn=False, sample="down-7")
    self.encoder_2 = PartialConvLayer(64, 128, sample="down-5")
    self.encoder_3 = PartialConvLayer(128, 256, sample="down-3")
    self.encoder_4 = PartialConvLayer(256, 512, sample="down-3")

    for i in range(5, layers + 1):
      name = "encoder_{:d}".format(i)
      setattr(self, name, PartialConvLayer(512, 512, sample="down-3"))

    for i in range(5, layers + 1):
      name = "decoder_{:d}".format(i)
      setattr(self, name, PartialConvLayer(512 + 512, 512, activation="leaky_relu"))

    self.decoder_4 = PartialConvLayer(512 + 256, 256, activation="leaky_relu")
    self.decoder_3 = PartialConvLayer(256 + 128, 128, activation="leaky_relu")
    self.decoder_2 = PartialConvLayer(128 + 64, 64, activation="leaky_relu")
    self.decoder_1 = PartialConvLayer(64 + 3, 3, bn=False, activation="", bias=True)

  def forward(self, input_x, mask):
    encoder_dict = {}
    mask_dict = {}
    key_prev = "h_0"
    encoder_dict[key_prev], mask_dict[key_prev] = input_x, mask

    # Encoding path
    for i in range(1, self.layers + 1):
      encoder_key = "encoder_{:d}".format(i)
      key = "h_{:d}".format(i)
      encoder_dict[key], mask_dict[key] = getattr(self, encoder_key)(encoder_dict[key_prev], mask_dict[key_prev])
      key_prev = key

    out_key = "h_{:d}".format(self.layers)
    out_data, out_mask = encoder_dict[out_key], mask_dict[out_key]

    # Decoding path with careful handling of tensor sizes
    for i in range(self.layers, 0, -1):
      encoder_key = "h_{:d}".format(i - 1)
      decoder_key = "decoder_{:d}".format(i)
      
      # Interpolate both data and mask
      out_data = F.interpolate(out_data, scale_factor=2, mode='nearest')
      out_mask = F.interpolate(out_mask, scale_factor=2, mode='nearest')
      
      # Ensure consistent sizes before concatenation
      encoder_data = encoder_dict[encoder_key]
      encoder_mask = mask_dict[encoder_key]
      
      # Pad if necessary
      if out_data.size() != encoder_data.size():
        diff_h = encoder_data.size(2) - out_data.size(2)
        diff_w = encoder_data.size(3) - out_data.size(3)
        out_data = F.pad(out_data, (0, diff_w, 0, diff_h))
        out_mask = F.pad(out_mask, (0, diff_w, 0, diff_h))
      
      # Concatenate
      out_data = torch.cat([out_data, encoder_data], dim=1)
      out_mask = torch.cat([out_mask, encoder_mask], dim=1)
      
      # Apply decoder
      out_data, out_mask = getattr(self, decoder_key)(out_data, out_mask)
    
    return out_data

app = Flask(__name__)
CORS(app)  # Enable CORS

model = PartialConvUNet()
state_dict = torch.load("model.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
device = torch.device("cpu")
model.to(device)

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

    # Prepare input for the model
    # Convert original image and mask to torch tensors
    original_tensor = torch.from_numpy(original).permute(2, 0, 1).float() / 255.0
    original_tensor = original_tensor.unsqueeze(0)  # Add batch dimension

    python_generated_mask = generate_mask(user_masked_image)
    binary_mask = cv2.cvtColor(python_generated_mask, cv2.COLOR_BGR2GRAY)
    
    # Create mask tensor with 3 channels to match input
    mask_tensor = torch.from_numpy(binary_mask).float() / 255.0
    mask_tensor = mask_tensor.repeat(3, 1, 1).unsqueeze(0)  # Repeat to 3 channels, add batch dimension

    # Ensure mask and input are the same size
    if mask_tensor.size()[2:] != original_tensor.size()[2:]:
        mask_tensor = F.interpolate(mask_tensor, size=original_tensor.size()[2:], mode='nearest')

    # Run model prediction
    with torch.no_grad():
        pred = model(original_tensor, mask_tensor)
    
    # Convert prediction back to numpy and scale
    pred_np = pred.squeeze().permute(1, 2, 0).numpy() * 255
    pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)

    response = {
        'originalImage': data['original'],
        'userMaskedImage': data['mask'],
        'pythonGeneratedMask': image_to_base64(python_generated_mask),
        'inpaintedImage': image_to_base64(pred_np),
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
