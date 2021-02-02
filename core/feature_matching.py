
import torch.nn as nn


class Feature_Matching(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Feature_Matching, self).__init__()
        self.conv2d = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


def FM(input_channel, output_channel):
    return Feature_Matching(input_channel, output_channel).cuda()