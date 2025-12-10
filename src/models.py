import torch
import torch.nn as nn
import timm


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))   # reduces to 64×1×1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



def get_model(model_name="efficientnet_b0", num_classes=2):
    if model_name == "baseline":
        return BaselineCNN(num_classes)

    model = timm.create_model(model_name, pretrained=True)
    in_features = model.get_classifier().in_features
    model.reset_classifier(num_classes)
    return model


if __name__ == "__main__":
    model = get_model("efficientnet_b0")
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)
