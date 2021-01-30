import torch
import torchvision.transforms as transforms
from torchvision.datasets import  CIFAR10, CIFAR100
from torch.utils.data import DataLoader 

import network

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def transformer(dataset_name):
        
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        train_trans = transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        test_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    else:
        raise ValueError('Invalid dataset name : {}'.format(dataset_name))

    return train_trans, test_trans

def load_dataset(dataset_name, dataset_path, bs):
    
    transform_train, transform_test = transformer(dataset_name)

    if dataset_name == 'cifar10':
        train_dt = CIFAR10(dataset_path, transform = transform_train, download=True)
        test_dt = CIFAR10(dataset_path, train = False, transform = transform_test, download = True)
        train_dt_len = len(train_dt)
        test_dt_len = len(test_dt)
        num_classes = 10
    elif dataset_name == 'cifar100':
        train_dt = CIFAR100(dataset_path, transform = transform_train, download = True)
        test_dt = CIFAR100(dataset_path, train = False, transform = transform_test, download = True)
        train_dt_len = len(train_dt)
        test_dt_len = len(test_dt)
        num_classes = 100

    data_train_loader = DataLoader(train_dt, batch_size = bs, shuffle = True, num_workers = 0)
    data_test_loader = DataLoader(test_dt, batch_size = bs, num_workers = 0)

    return data_train_loader, data_test_loader, train_dt_len, test_dt_len, num_classes



def load_model_arch(model_name,num_classes):
    
    try:
        model_func = getattr(network, model_name) 
    except:
        raise ValueError('Invalid model name : {}'.format(model_name))
    
    model = model_func(num_classes)
    if cuda_available():
        model.cuda()

    return model

 