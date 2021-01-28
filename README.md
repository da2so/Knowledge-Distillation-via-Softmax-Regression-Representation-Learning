# Knowledge Distillation via Softmax Regression Representation Learning PyTorch


![Python version support](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch version support](https://img.shields.io/badge/pytorch-1.7.0-red.svg)

:star: Star us on GitHub — it helps!!

PyTorch implementation for *[Knowledge Distillation via Softmax Regression Representation Learning](https://openreview.net/pdf?id=ZzwDy_wiWv)*


## Install

You will need a machine with a GPU and CUDA installed.  
Then, you prepare runtime environment:

   ```shell
   pip install -r requirements.txt
   ```


## Use


### Train a teacher network

If you want to train a network for yourself:

   ```shell
   CUDA_VISIBLE_DEVICES=0 python train_network.py --dataset_name=cifar10 --teacher_name=resnet34
   ```

Arguments:

- `dataset_name` - Select a dataset ['cifar10' or 'cifar100']
- `teacher_name` - Trainable network names
   - Available list
      - VGG: ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
      - ResNet: ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
      - Wide ResNet: ['wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2']
      - MobileNet: ['mobilenetv2']
- `bs` - Batch size
- `epochs` - The number of epochs
- `lr` - Learning rate
   - Default is 0.1 

The trained model will be saved in `./cache/models/` directory.


### Knowledge distillation for student network

   ```shell
   CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name=./cache/models/resnet34_cifar10.pt --student_name=resnet10
   ```

- `dataset_name` - Select a dataset ['cifar10' or 'cifar100']
- `teacher_path` - Teacher network path
- `student_name` - Student network name
   - Available list
      - VGG: ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
      - ResNet: ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
      - Wide ResNet: ['wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2']
      - MobileNet: ['mobilenetv2']
- `bs` - Batch size
- `epochs` - The number of epochs
- `lr` - Learning rate
   - Default is 0.1 
- `img_shape` - Batch size
- `save_path` - Input shape for a network
- `teacher_name` - Save path for student network
	- Use only when you save a model using `state_dict()`


## Understanding this method(algorithm)

:white_check_mark: Check my blog!!
[Here](https://da2so.github.io/2021-01-24-Knowledge_Distillation_via_Softmax_Regression_Representation_Learning/)