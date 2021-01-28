
import torch.nn as nn


class Feature_Matching(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Feature_Matching, self).__init__()
        self.linear = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        out = self.linear(x)
        return out



def FM(input_channel, output_channel):
    return Feature_Matching(input_channel, output_channel).cuda()