# Task 1

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, device='cpu'):
        super().__init__()

        self.conv1 =nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_channels)
            )

        self.to(device)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # Add skip connection
        x = self.relu(x + identity)
        return x
    
class MinePredictionNet(nn.Module):
    def __init__(self, input_size=(12, 22, 22), device='cpu'):
        super().__init__()

        # Initial convolutional layer
        self.conv = nn.Conv2d(input_size[0], 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16, dilation=1, device=device),
            ResidualBlock(16, 16, dilation=1, device=device),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32, dilation=2, device=device),
            ResidualBlock(32, 32, dilation=2, device=device),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64, dilation=4, device=device),
            ResidualBlock(64, 64, dilation=4, device=device),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(64, 128, dilation=8, device=device),
            ResidualBlock(128, 128, dilation=8, device=device),
        )

        self.last_conv = nn.Conv2d(128, 1, kernel_size=1)

        self.to(device)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.last_conv(x)

        return x
    

def main():
    pass

if __name__ == "__main__":
    main()