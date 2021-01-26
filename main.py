import argparse
import os

from styler import Styler

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='[cifar10, cifar100]' )

    parser.add_argument('--teacher_path', type=str, default='./', help='teacher network path' )
    parser.add_argument('--student_path', type=str, default='./90424.jpg', help='style dataset path' )

    parser.add_argument('--data', type=str, default='./cache/data/')
    parser.add_argument('--teacher_dir', type=str, default='./cache/models/')
    parser.add_argument('--bs', type=str, default=224, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs of training')
    parser.add_argument('--lr', type=float, default=1e-2)

    args = parser.parse_args()

    style_obj = Styler(
                    vgg_path = args.vgg_path,
                    content_path = args.content_path,
                    style_path = args.style_path,
                    lr = args.lr,
                    bs = args.bs,
                    epochs = args.epochs
                    )
    style_obj.build()
