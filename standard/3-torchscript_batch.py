import os
import time

import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.io import read_image

def read_paths_images(directory, dataset_size):
    paths = [directory + f for f in os.listdir(directory) if
                        os.path.isfile(os.path.join(directory, f))]
    paths = paths[:dataset_size]
    return paths

def load_images(urls):
    images = []
    for image_path in urls:
        image = Image.open(image_path)
        image = image.convert('RGB')
        images.append(image)
    return images
def transform(images):
    imagenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    transformed_images = []
    for img in images:
        transformed_image = imagenet_transform(img)
        transformed_images.append(transformed_image)
    transformed_batch = torch.stack(transformed_images, dim=0)
    return transformed_batch

def inference(model, transformed_batch):
    with torch.no_grad():
        output = model(transformed_batch)
    predictions_batch = output.softmax(1)
    top_predictions_batch = torch.topk(predictions_batch, k=1).indices.squeeze(1)
    weights = ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]
    labels = []
    probabilities = []
    for i, top_prediction in enumerate(top_predictions_batch):
        category_index = top_prediction.item()
        category_name = categories[category_index]
        probability = predictions_batch[i, category_index].item()
        labels.append(category_name)
        probabilities.append(probability)
    results = [{'prob': float(prob), 'label': label} for prob, label in zip(probabilities, labels)]

    return results
def main(dataset_size):
    directory ="../ImageNet-Datasets-Downloader/imagenet_images/"
    paths = read_paths_images(directory, dataset_size)
    images_data = load_images(paths)
    transformed_batch = transform(images_data)
    model_path = "models/torchscript_resnet50.pt"
    # Load the torchscript model
    model = torch.jit.load(model_path,torch.device('cpu'))
    results = inference(model, transformed_batch)
    for url, result in zip(paths,results):
        print(f'{url} {result}')

if __name__ == '__main__':
    DATASET_SIZE = 5
    main(DATASET_SIZE)

