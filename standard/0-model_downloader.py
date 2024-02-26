import torch
import torchvision
from torchvision.models import list_models

classification_models = list_models(module=torchvision.models)
print(classification_models)

# Get pre-trained ImageNet ResNet-50 model
resnet50_weights = torchvision.models.ResNet50_Weights.DEFAULT
resnet50 = torchvision.models.resnet50(weights=resnet50_weights)
resnet50.eval()

# Save the model
torch.save(resnet50, 'models/resnet50.pth')

# Pass it to torchscript
resnet50_scripted = torch.jit.script(resnet50)

# Save the model
torch.jit.save(resnet50_scripted, 'models/torchscript_resnet50.pt')