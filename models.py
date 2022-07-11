# Here is where I will do feature extraction.
# I have decided to do the feature extraction more manually in order to achieve
# more flexibility such as switching max pooling with avg pooling

from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
from utils import StyleLoss, ContentLoss


class VGG19(nn.Module):

    style_layers = ["relu1_1", "relu2_1",
                    "relu3_1", "relu4_1", "relu5_1"]

    content_layers = ["relu4_2"]

    positions = [1, 6, 11, 20, 22, 29]

    # after trying average pooling, the results seem buggy so I've decided
    # to default it to max pooling until I figure out what's going on
    pooling = "max"

    def __init__(self, content_img, style_img, device="cpu"):
        super(VGG19, self).__init__()
        features = vgg19(
            weights=VGG19_Weights.DEFAULT).features.eval().to(device)
        features.requires_grad_(False)

        self.style_feature1 = nn.Sequential()
        self.style_feature2 = nn.Sequential()
        self.style_feature3 = nn.Sequential()
        self.style_feature4 = nn.Sequential()
        self.content_feature = nn.Sequential()
        self.style_feature5 = nn.Sequential()

        layers = [self.style_feature1, self.style_feature2, self.style_feature3,
                  self.style_feature4, self.content_feature, self.style_feature5]

        self.content_losses = []
        self.style_losses = []

        pool_cnt, relu_count, conv_count = 1, 1, 1
        prev_pos = 0
        for layer, pos in zip(layers, self.positions):
            for j in range(prev_pos, pos+1):
                x = features[j]
                if isinstance(x, nn.Conv2d):
                    name = f"conv{pool_cnt}_{conv_count}"
                    layer.add_module(name, x)
                    conv_count += 1
                elif isinstance(x, nn.ReLU):
                    name = f"relu{pool_cnt}_{relu_count}"
                    layer.add_module(name, x)
                    relu_count += 1
                else:
                    name = f"pool{pool_cnt}"
                    if self.pooling == "avg":
                        layer.add_module(name, nn.AvgPool2d(2, 2))
                    else:
                        layer.add_module(name, x)
                    relu_count = 1
                    conv_count = 1
                    pool_cnt += 1

            content_img = layer(content_img)
            style_img = layer(style_img)

            if name in self.style_layers:
                loss_module = StyleLoss(style_img)
                layer.add_module("style_loss", loss_module)
                self.style_losses.append(loss_module)
            elif name in self.content_layers:
                loss_module = ContentLoss(content_img)
                layer.add_module("content_loss", loss_module)
                self.content_losses.append(loss_module)
            else:
                raise Exception(f"layer name [{name}] not recognized")

            prev_pos = pos+1

    def forward(self, input):
        content = 0
        style = 0

        x = self.style_feature1(input)
        x = self.style_feature2(x)
        x = self.style_feature3(x)
        x = self.style_feature4(x)
        x = self.content_feature(x)
        x = self.style_feature5(x)

        # calculating the losses
        for style_loss in self.style_losses:
            style += style_loss.loss
        for content_loss in self.content_losses:
            content += content_loss.loss

        return content, style
