import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from torchvision import transforms
from torchvision import models
from grad_cam import GradCAM
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torch.nn as nn

softmax = nn.Softmax(dim=-1)
path = "./test.jpg"
file_name = 'out.jpg'
image_model_path = "./result/resnet18/final_model.prm"
image_model_save_point = torch.load(image_model_path)
model = models.resnet18(pretrained=False, num_classes=2)

image = Image.open(path)
def get_label(model, img, label_id):

    output = model(img)
    output = softmax(output)
    _, pred = output.max(dim=1)
#    print(label_id[pred])
    print(output, pred.item())
    return pred

state_dict = torch.load(image_model_path)
model.load_state_dict(state_dict)

model.eval()
label_id = {0: 'butterfly',1: 'ibis'}






grad_cam = GradCAM(model=model, feature_layer=list(model.layer4.modules())[-1])

VISUALIZE_SIZE = (224, 224)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

image_transform = transforms.Compose([
        transforms.Resize(VISUALIZE_SIZE),
        transforms.ToTensor(),
        normalize])

image.thumbnail(VISUALIZE_SIZE, Image.ANTIALIAS)

# save image origin size
image_orig_size = image.size # (W, H)

img_tensor = image_transform(image)
img_tensor = img_tensor.unsqueeze(0)

get_label(model,img_tensor, label_id)

model_output = grad_cam.forward(img_tensor)
target = model_output.argmax(1).item()

grad_cam.backward_on_target(model_output, target)

# Get feature gradient
feature_grad = grad_cam.feature_grad.data.numpy()[0]
# Get weights from gradient
weights = np.mean(feature_grad, axis=(1, 2))  # Take averages for each gradient
# Get features outputs
feature_map = grad_cam.feature_map.data.numpy()
grad_cam.clear_hook()


# Get cam
cam = np.sum((weights * feature_map.T), axis=2).T
cam = np.maximum(cam, 0)  # apply ReLU to cam

cam = cv2.resize(cam, VISUALIZE_SIZE)
cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1



cam = np.uint8(cam*255)  # Scale between 0-255 to visualize
activation_heatmap = np.expand_dims((255 - cam), axis=0).transpose(1,2,0)
org_img = np.asarray(image.resize(VISUALIZE_SIZE))

activation_heatmap = np.float32(cv2.applyColorMap(np.uint8(activation_heatmap), cv2.COLORMAP_JET))

img_with_heatmap = np.multiply(np.float32(activation_heatmap*0.4), np.float32(org_img))
img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
org_img = cv2.resize(org_img, image_orig_size)


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(org_img)
plt.subplot(1,2,2)
plt.imshow(cv2.resize(np.uint8(255 * img_with_heatmap), image_orig_size))
plt.savefig(file_name)



