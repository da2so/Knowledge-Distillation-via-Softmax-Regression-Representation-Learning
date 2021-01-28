import numpy as np

import torch

from network.utils import load_model_arch

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def load_teacher(model_path, model_name = None, num_classes = None):
    assert ('.pt' or '.pth') in model_path

    if torch.typename(torch.load(model_path)) == 'OrderedDict':
        model = load_model_arch(model_name, num_classes)
        model.load_state_dict(torch.load(model_path))
    else:
        model = torch.load(model_path)

    model.eval()
    if cuda_available():
        model.cuda()

    return model


def get_finalconv_channel_num(model, img_shape):
    """
    Get filter size of last conv layer for feature matching between teacher and student
    """
    x = torch.unsqueeze(torch.randn(img_shape), 0).cuda()

    for name, layer in model._modules.items():

        if type(layer) is torch.nn.modules.linear.Linear or \
            type (layer) is torch.nn.modules.pooling.AdaptiveAvgPool2d or \
            type (layer) is torch.nn.modules.pooling.AvgPool2d:
            break
        else:
            x = layer(x)
    
    finalconv_channel_num = np.shape(x)[1]
    
    return  finalconv_channel_num

def get_finalconv(model):

    for name, layer in model._modules.items():
        if type(layer) is torch.nn.modules.linear.Linear:
            finalconv = layer
            break
        else:
            pass
    
    return finalconv 

def get_T_classifer(model):
    """
    Get teacher's classifier for calculating softmax regression loss
    """
    isClassifier = False
    layers = []
    for name, layer in model._modules.items():
        if type(layer) is torch.nn.modules.linear.Linear:
            isClassifier = True
        
        if isClassifier == True:
            layers.append(layer)

    T_classifier = torch.nn.Sequential(*layers)
    return T_classifier   