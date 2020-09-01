import torchvision
import torch.nn as nn
import torch.nn.functional as F


class FineTuneModel(nn.Module):
    def __init__(self, original_model, num_outputs):
        super(FineTuneModel, self).__init__()
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.modelName = 'VGG19-Downsample'
        self.outputs = nn.Sequential(

        )
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        return f


vgg19 = torchvision.models.vgg19(pretrained=True)
model = FineTuneModel(vgg19, 512)
print(model)
