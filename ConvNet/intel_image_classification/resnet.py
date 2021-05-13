import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, last_layer_down_sample=None) -> None:
        super(Block, self).__init__()
        self.expansion = 4 # 64 -> 256, 128 -> 512 ...
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size = 1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = last_layer_down_sample
    
    def forward(self, input):
        identity = input
        output = self.relu(self.bn1(self.conv1(input)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        
        if self.last_layer_down_sample is not None:
            identity = self.identity_downsample(output)

        output += identity
        output = self.relu(output)
        return output
    
class ResNet(nn.Module):
    def __init__(self, Block, in_channels, num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()

        # resnet50 layer
        self.layer1 = self._make_layer(Block, num_res_block = 3, out_channel=64, stride=1)
        self.layer2 = self._make_layer(Block, num_res_block = 4, out_channel=128, stride=2)
        self.layer3 = self._make_layer(Block, num_res_block = 6, out_channel=256, stride=2)
        self.layer4 = self._make_layer(Block, num_res_block = 3, out_channel=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 4, num_classes)
    
    def forward(self, input):
        output = self.conv1(input)
        output = self.relu(self.bn1(output))
        output = self.maxpool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output


    def _make_layer(self, block, num_res_block, out_channel, stride):
        identity_downsample = None
        layers = []

        for _ in range(num_res_block - 1):
            layers.append(Block(self.in_channels, out_channel, stride=stride))

        # last block output size is different
        if stride != 1 or self.in_channels != out_channel * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channel * 4, kernel_size=1, stride = stride),
                nn.BatchNorm2d(out_channel * 4)
            )
        layers.append(Block(self.in_channels, out_channel, stride, last_layer_down_sample=identity_downsample))
        
        
def resnet50(in_channels, num_classes):
    return ResNet(Block, in_channels, num_classes)
resnet50(3, 6)




