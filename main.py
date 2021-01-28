import argparse
import os

from core.knowledge_distillation import KD

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type = str, default = 'cifar10', help ='[cifar10, cifar100]' )
    parser.add_argument('--teacher_path', type = str, default = './cache/models/resnet34_cifar10.pt', help ='Teacher network path' )
    """
    Available student list :[
        mobilenetv2, 
        vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
        resnet10, resnet18, resnet34, resnet50, resnet101, resnet152,
        wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
    ]
    """
    parser.add_argument('--student_name', type = str, default = 'resnet10', help = 'Style dataset path' )
    parser.add_argument('--bs', type = str, default = 256, help = 'Batch size')
    parser.add_argument('--epochs', type = int, default = 150, help = 'The number of epochs')
    parser.add_argument('--lr', type = float, default = 0.1, help = 'Learning rate')
    parser.add_argument('--img_shape', type = list, default = [3,32,32], help = 'Learning rate')
    parser.add_argument('--save_path', type = str, default = './result/', help = 'Save path for student')
    parser.add_argument('--teacher_name', type = str, default = None, help = 'Teahcer architecture name')

    args = parser.parse_args()

    kd_obj = KD(
                dataset_name = args.dataset_name,
                teacher_path = args.teacher_path,
                student_name = args.student_name,
                bs           = args.bs,
                epochs       = args.epochs,
                lr           = args.lr,
                img_shape    = args.img_shape,
                save_path    = args.save_path,
                teacher_name = args.teacher_name
                )
    kd_obj.build()
