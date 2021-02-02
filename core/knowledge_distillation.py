import os 
from tqdm import tqdm

import torch
from torch.nn.functional import softmax
from torch.autograd import Variable 

from network.utils import load_dataset, load_model_arch
from core.utils import load_model, get_finalconv, get_finalconv_channel_num, get_T_classifer 
from core.feature_matching import FM
class KD():
    def __init__(self, dataset_name, teacher_path, student_name, bs, epochs, lr,
                img_shape, save_dir, teacher_name = None ,dataset_dir='./cache/dataset/'):
        self.best_acc = 0.0
        self.student_name = student_name
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.save_path = f'{self.save_dir}{self.student_name}_{self.dataset_name}.pt'
        self.bs = bs
        self.epochs = epochs

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.tr_loader, self.te_loader, self.tr_len, self.te_len, self.n_classes = load_dataset(
                                                                            self.dataset_name, 
                                                                            self.dataset_dir, 
                                                                            self.bs)

        self.teacher = load_model(teacher_path, teacher_name, self.n_classes)
        self.student = load_model_arch(self.student_name, self.n_classes)

        # Activations of teacher and student 
        self.T_activations = dict()
        self.S_activations = dict()

        def forward_hook_T(module, input, output):
            self.T_activations['value'] = output
            return None
        def forward_hook_S(module, input, output):
            self.S_activations['value'] = output
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
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [60,120,160], gamma=0.1, last_epoch=-1, verbose=False)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [150,250,320], gamma=0.1, last_epoch=-1, verbose=False)

        self.ce_loss = torch.nn.CrossEntropyLoss().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.bce_loss = torch.nn.BCELoss().cuda()

    def train(self, epoch):
        self.student.train()
        self.teacher.eval()
        if self.use_fm == True:
            self.fm.train()
        total_correct = 0

        loop = tqdm(enumerate(self.tr_loader), total= len(self.tr_loader)-1, leave=False)
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

            softmax_output_T = softmax(self.T_classifer(sum_activations_T), dim = 1)
            softmax_output_S = softmax(self.T_classifer(sum_activations_S), dim = 1)
            
            sr_loss = self.mse_loss(softmax_output_S, softmax_output_T.detach())

            total_loss = ce_loss + fm_loss + sr_loss 

            pred = output_S.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            acc = float(total_correct) / self.tr_len
    
            loop.set_description(f"Train - Epoch [{epoch}/{self.epochs}]")
            loop.set_postfix(Acc = acc, CE_loss = ce_loss.data.item(), FM_loss = fm_loss.data.item(), SR_loss = sr_loss.data.item())

            total_loss.backward()
            self.optimizer.step()

            # For printing training progress bar
            if i == len(self.tr_loader) -1:
                print()

                
    def test(self, model, forResult=False):
        model.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.te_loader):
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                output = model(images)
                avg_loss += self.ce_loss(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
    
        avg_loss /= self.te_len
        acc = float(total_correct) / self.te_len

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.student.state_dict(), self.save_path)
        
        if forResult == False:
            print('Test - Acc: %f, Loss: %f' % (acc, avg_loss.data.item()))
        else:
            return acc

    def print_result(self):
        teacher_acc = self.test(self.teacher, forResult=True)
        student = load_model(self.save_path, self.student_name, self.n_classes)
        student_acc = self.test(student, forResult=True)

        print('\n\n')
        result_text = '-'*30+' Result '+'-'*30
        result_text_len = len(result_text)
        print('-'*30+' Result '+'-'*30)
        print(f'\nTest accuracy of Teacher is {teacher_acc}')
        print(f'Test accuracy of Student is {student_acc}\n')
        print('-'*result_text_len)

    def build(self):
        
        print('-'*30+' Train student start '+'-'*30)

        for epoch in range(1, self.epochs):
            self.train(epoch)
            self.test(self.student)
            self.scheduler.step()

        self.S_handle.remove()

        print('-'*30+' Train student  end  '+'-'*30)

        self.print_result()


