import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2
import argparse
from torchvision import models, transforms
from PIL import Image


transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
#  transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 #transforms.Normalize(                      #[5]
 #mean=[0.485, 0.456, 0.406],                #[6]
 #std=[0.229, 0.224, 0.225]                  #[7]
 ])


visualisation = []

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    visualisation.append(output.data)



net = models.resnet18(pretrained=True)
net.eval()


net.layer1.register_forward_hook(printnorm)
net.layer2.register_forward_hook(printnorm)
net.layer3.register_forward_hook(printnorm)
net.layer4.register_forward_hook(printnorm)

##Input image 1 processing
test_im1 = Image.open("/home/tandem-team/Work_Folder/Challenge_images/50.jpg")
# test_im1 = Image.open("/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/strawberries.jpg")
# plt.imshow(cv2.imread("/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/strawberries.jpg"))
# plt.show()
test_im1=transform(test_im1)
test_im1=torch.unsqueeze(test_im1,0)

net(test_im1)

print(visualisation[0].shape)
print(visualisation[1].shape)
print(visualisation[2].shape)
print(visualisation[3].shape)

pool_op = nn.AdaptiveAvgPool2d((1,1))

layer1_out = pool_op(visualisation[0].squeeze(0))
layer2_out = pool_op(visualisation[1].squeeze(0))
layer3_out = pool_op(visualisation[2].squeeze(0))
layer4_out = pool_op(visualisation[3].squeeze(0))
test_tensor1 = visualisation[0].squeeze(0)
torch.ops.load_library("/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/build/lib.linux-aarch64-3.6/rlcustom_operator.cpython-36m-aarch64-linux-gnu.so")

final_out1 = torch.ops.Pytorch_ONNX_ex.reduction(layer1_out.reshape([1,-1]),layer2_out.reshape([1,-1]),layer3_out.reshape([1,-1]),layer4_out.reshape([1,-1]))


#Input image 2 processing
# test_im2 = Image.open("/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/black_color.jpg")
test_im2 = Image.open("/home/tandem-team/Work_Folder/Challenge_images/34.jpg")
# plt.imshow(cv2.imread("/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/black_color.jpg"))
# plt.show()
test_im2=transform(test_im2)
test_im2=torch.unsqueeze(test_im2,0)

net = models.resnet18(pretrained=True)
net.eval()


net.layer1.register_forward_hook(printnorm)
net.layer2.register_forward_hook(printnorm)
net.layer3.register_forward_hook(printnorm)
net.layer4.register_forward_hook(printnorm)


net(test_im2)

print(visualisation[4].shape)
print(visualisation[5].shape)
print(visualisation[6].shape)
print(visualisation[7].shape)

layer1_out2 = pool_op(visualisation[4].squeeze(0))
layer2_out2 = pool_op(visualisation[5].squeeze(0))
layer3_out2 = pool_op(visualisation[6].squeeze(0))
layer4_out2 = pool_op(visualisation[7].squeeze(0))

final_out2 = torch.ops.Pytorch_ONNX_ex.reduction(layer1_out2.reshape([1,-1]),layer2_out2.reshape([1,-1]),layer3_out2.reshape([1,-1]),layer4_out2.reshape([1,-1]))


cos = nn.CosineSimilarity(dim=1, eps=1e-06)

print(final_out1.size())
print(final_out2.size())
print(len(visualisation))
print(cos(final_out1, final_out2))

vec_out=cos(final_out1, final_out2)
print(vec_out.mean())
print("@@@@@@@@@@@@@@@@@@@@@@")
# print("test1 and test2\n", final_out1, "\n", final_out2)

direct1_out1 = torch.repeat_interleave(layer1_out.reshape([1,-1]),8)
direct1_out2 = torch.repeat_interleave(layer2_out.reshape([1,-1]),4)
direct1_out3 = torch.repeat_interleave(layer3_out.reshape([1,-1]),2)
direct1_out4 = layer4_out.reshape([1,-1]).squeeze(0)
print(direct1_out1.size())
print(direct1_out2.size())
print(direct1_out3.size())
print(direct1_out4.size())

direct2_out1 = torch.repeat_interleave(layer1_out2.reshape([1,-1]),8)
direct2_out2 = torch.repeat_interleave(layer2_out2.reshape([1,-1]),4)
direct2_out3 = torch.repeat_interleave(layer3_out2.reshape([1,-1]),2)
direct2_out4 = layer4_out2.reshape([1,-1]).squeeze(0)

out1=torch.cat((direct1_out1.unsqueeze(0),direct1_out2.unsqueeze(0),direct1_out3.unsqueeze(0),direct1_out4.unsqueeze(0)))
out2=torch.cat((direct2_out1.unsqueeze(0),direct2_out2.unsqueeze(0),direct2_out3.unsqueeze(0),direct2_out4.unsqueeze(0)))

print("@@@@@@@@@@@@@@@@@@@@@@")
print(out1.size())
print(out2.size())
print(torch.mean(out1,0).size())

print(torch.mean(out2,0))
print(torch.all(torch.eq(final_out1.squeeze(0), torch.mean(out1,0).unsqueeze(0))))
print(torch.all(torch.eq(final_out2.squeeze(0), torch.mean(out2,0).unsqueeze(0))))