import os 
from tqdm import tqdm

import torch
from torch.nn.functional import softmax
from torch.autograd import Variable 

from network.utils import load_dataset, load_model_arch
from core.utils import load_teacher, get_finalconv, get_finalconv_channel_num, get_T_classifer 
from core.feature_matching import FM
class KD():
    def __init__(self, dataset_name, teacher_path, student_name, bs, epochs, lr,
                img_shape, save_path, teacher_name = None ,dataset_path='./cache/dataset/'):

        self.student_name = student_name
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.bs = bs
        self.epochs = epochs

        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.tr_loader, self.te_loader, self.tr_len, self.te_len, n_classes = load_dataset(
                                                                            self.dataset_name, 
                                                                            self.dataset_path, 
                                                                            self.bs)

        self.teacher = load_teacher(teacher_path, teacher_name, n_classes)
        self.student = load_model_arch(student_name, n_classes)

        # Activations of teacher and student 
        self.T_activations = dict()
        self.S_activations = dict()

        def forward_hook_T(module, input, output):
            self.T_activations['value'] = input[0]
            return None
        def forward_hook_S(module, input, output):
            self.S_activations['value'] = input[0]
            return None
        
        # Get the last conv layer or a module that contains the last con layer
        self.T_finalconv = get_finalconv(self.teacher)
        self.S_finalconv = get_finalconv(self.student)
        
        # And output channel number of the last conv layer of teacher and student for feature matching (FM) loss
        T_finalconv_num_ch = get_finalconv_channel_num(self.teacher, img_shape)
        S_finalconv_num_ch = get_finalconv_channel_num(self.student, img_shape)

        # Hook for FM loss
        self.T_handle = self.T_finalconv.register_forward_hook(forward_hook_T)
        self.S_handle = self.S_finalconv.register_forward_hook(forward_hook_S)

        # Get classifer of teacher for softmax reression loss
        self.T_classifer = get_T_classifer(self.teacher)
        
        # Feature matching module
        self.use_fm = False
        if T_finalconv_num_ch != S_finalconv_num_ch:
            self.use_fm = True
        
        if self.use_fm == True:
            self.fm = FM(S_finalconv_num_ch, T_finalconv_num_ch)
            self.optimizer = torch.optim.SGD(list(self.student.parameters()) + list(self.fm.parameters()), lr=lr, momentum = 0.9, weight_decay = 5e-4)
        else:
            self.optimizer = torch.optim.SGD(self.student.parameters(), lr=lr, momentum = 0.9, weight_decay = 5e-4)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.ce_loss = torch.nn.CrossEntropyLoss().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.bce_loss = torch.nn.BCELoss().cuda()
    def train(self, epoch):
        self.student.train()
        total_correct = 0

        loop = tqdm(enumerate(self.tr_loader), total= len(self.tr_loader), leave=False)
        for i, (images, labels) in loop :

            images, labels = Variable(images).cuda(), Variable(labels).cuda()
    
            self.optimizer.zero_grad()
            output_T = self.teacher(images)
            output_S = self.student(images)
            
            ce_loss = self.ce_loss(output_S, labels)

            # Summation per channel
            sum_activations_T = self.T_activations['value']
            sum_activations_S = self.S_activations['value']

            if self.use_fm == True:
                # Operate feature matching
                sum_activations_S = self.fm(sum_activations_S)
                sum_activations_S = torch.squeeze(sum_activations_S)

            fm_loss = self.mse_loss(sum_activations_S, sum_activations_T)

            #print(fm_loss)
            softmax_output_T = softmax(self.T_classifer(sum_activations_T), dim = 1)
            softmax_output_S = softmax(self.T_classifer(sum_activations_S), dim = 1)
            
            sr_loss = self.mse_loss(softmax_output_S, softmax_output_T.detach())

            total_loss = ce_loss + fm_loss + sr_loss 
            
            pred = output_S.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            acc = float(total_correct) / self.tr_len

            # For printing training progress bar
            if i == len(self.tr_loader) -1 :
                print()
    
            loop.set_description(f"Train - Epoch [{epoch}/{self.epochs}]")
            loop.set_postfix(ce_loss = ce_loss.data.item(), fm_loss = fm_loss.data.item(), sr_loss = sr_loss.data.item(), Acc = acc)

            total_loss.backward()
            self.optimizer.step()

                
    def test(self):
        self.student.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.te_loader):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                output = self.student(images)
                avg_loss += self.ce_loss(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        avg_loss /= self.te_len
        acc = float(total_correct) / self.te_len

        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    

    def build(self):
        
        print('-'*30+' Train student start '+'-'*30)

        for epoch in range(1, self.epochs):
            self.train(epoch)
            self.test()
            self.scheduler.step()

        self.S_handle.remove()
        torch.save(self.student, f'{self.save_path}{self.student_name}_{self.dataset_name}_cuda1.pt')

        print('-'*30+' Train student end '+'-'*30)

