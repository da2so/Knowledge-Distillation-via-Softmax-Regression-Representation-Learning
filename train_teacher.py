import os
import argparse
from tqdm import tqdm

import torch
from torch.autograd import Variable 

from network.utils import load_dataset, load_model_arch


class Trainer():
    def __init__(self, dataset_name, teacher_name, bs, lr, epochs, 
                dataset_path = './cache/dataset/', model_path = './cache/models/'):

        self.epochs = epochs
        self.bs = bs
        self.dataset_name = dataset_name
        self.teacher_name = teacher_name
        self.dataset_path = dataset_path
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        self.epochs = epochs
        self.bs = bs
        
        self.tr_loader, self.te_loader, self.tr_len, self.te_len, n_classes = load_dataset(
                                                                        self.dataset_name,
                                                                        self.dataset_path,
                                                                        self.bs
                                                                        )

        self.teacher = load_model_arch(self.teacher_name, n_classes)

        self.optimizer = torch.optim.SGD(self.teacher.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.epochs)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

            
    def train(self,epoch):
        self.teacher.train()
        total_correct = 0

        loop = tqdm(enumerate(self.tr_loader), total= len(self.tr_loader), leave=False)
        for i, (images, labels) in loop :

            images, labels = Variable(images).cuda(), Variable(labels).cuda()
    
            self.optimizer.zero_grad()
            output = self.teacher(images)
            loss = self.criterion(output, labels)

            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            acc = float(total_correct) / self.tr_len

    
            if i == len(self.tr_loader) -1 :
                print()
                #print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
            
            loss.backward()
            self.optimizer.step()
    
            loop.set_description(f"Train - Epoch [{epoch}/{self.epochs}]")
            loop.set_postfix(Loss = loss.data.item(), Acc = acc)
            
    def test(self):
        self.teacher.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.te_loader):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                output = self.teacher(images)
                avg_loss += self.criterion(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        avg_loss /= self.te_len
        acc = float(total_correct) / self.te_len

        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    
    
    def build(self):
        print('-'*30+' Train teacher start '+'-'*30)

        for epoch in range(1, self.epochs):
            self.train(epoch)
            self.test()
            self.scheduler.step()

        torch.save(self.teacher, self.model_path + f'{self.teacher_name}_{self.dataset_name}.pt')

        print('-'*30+' Train teacher end '+'-'*30)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Train Teacher with PyTorch')

    parser.add_argument('--dataset_name', type = str, default = 'cifar10', help = '["cifar10", "cifar100"]')
    """
    Available teacher list :[
        mobilenetv2, 
        vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
        resnet10, resnet18, resnet34, resnet50, resnet101, resnet152,
        wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
    ]
    """
    parser.add_argument('--teacher_name', type = str, default = 'resnet34', help = 'Teacher model name')
    parser.add_argument('--bs', type = int, default = 256, help = 'Batch size')
    parser.add_argument('--lr', type = int, default = 0.1, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, default = 150, help = 'The number of epochs')
    args = parser.parse_args()

    trainer_obj = Trainer(
                        dataset_name = args.dataset_name,
                        teacher_name = args.teacher_name,
                        bs           = args.bs,
                        lr           = args.lr,
                        epochs       = args.epochs
                        )
    trainer_obj.build()
