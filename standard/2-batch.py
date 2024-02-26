import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
import os

# Step 1: Load the image
directory ="../ImageNet-Datasets-Downloader/imagenet_images/"
batch_size=5
paths = [directory + f for f in os.listdir(directory) if
                        os.path.isfile(os.path.join(directory, f))]
image_paths = paths[:batch_size]

# Load weights
weights = ResNet50_Weights.DEFAULT

# Load the model
# model = resnet50(weights=weights)
model = torch.load("models/resnet50.pth")
model.eval()

# Define the inference transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Preprocess the images and stack them into a single tensor
input_tensors = []
for image_path in image_paths:
    img = read_image(image_path)
    img = transforms.ToPILImage()(img)
    input_tensor = transform(img).unsqueeze(0)
    input_tensors.append(input_tensor)
input_batch = torch.cat(input_tensors, dim=0)

# Perform batch inference
with torch.no_grad():
    output_batch = model(input_batch)

# Post-process the results
predictions_batch = output_batch.softmax(1)
top_predictions_batch = torch.topk(predictions_batch, k=1).indices.squeeze(1)

# Get the category names from the metadata
categories = weights.meta["categories"]

# Print the predicted categories for each image
for i, top_prediction in enumerate(top_predictions_batch):
    image_path = image_paths[i]
    category_index = top_prediction.item()
    category_name = categories[category_index]
    probability = predictions_batch[i, category_index].item()
    print(f"Image: {image_path}, Predicted Category: {category_name}, Probability: {probability:.4f}")
