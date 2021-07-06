import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2
import argparse
from torchvision import models, transforms
from PIL import Image

#transform from Imagenet dataset on which the model was pre-trained
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225])                  #[7]
 ])

#List to store resnet layer feature maps from the hook
visualisation = []

#Function to attach hook onto the resnet layers
def printnorm(self, input, output):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    visualisation.append(output.data)

#Resnet18 pre-trained model from torchvision
net = models.resnet18(pretrained=True)
net.eval()

#Attaching hooks onto Resnet layers
net.layer1.register_forward_hook(printnorm)
net.layer2.register_forward_hook(printnorm)
net.layer3.register_forward_hook(printnorm)
net.layer4.register_forward_hook(printnorm)

##Input image 1 processing
test_im1 = Image.open("/home/tandem-team/Work_Folder/Challenge_images/50.jpg")
test_im1=transform(test_im1)
test_im1=torch.unsqueeze(test_im1,0)

#Passing the image through the net stores the feature maps in variable "visualisation"
net(test_im1)

#Pooling operation to reduce the dimensions to [1,x] ->x is 64 or 128 or 256 or 512
pool_op = nn.AdaptiveAvgPool2d((1,1))
layer1_out = pool_op(visualisation[0].squeeze(0))
layer2_out = pool_op(visualisation[1].squeeze(0))
layer3_out = pool_op(visualisation[2].squeeze(0))
layer4_out = pool_op(visualisation[3].squeeze(0))

#Loading the custom operator implemented in cpp file rlcustom_operator
torch.ops.load_library("/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/build/lib.linux-aarch64-3.6/rlcustom_operator.cpython-36m-aarch64-linux-gnu.so")

#Final vector from the reduction and interleave
final_out1 = torch.ops.Pytorch_ONNX_ex.reduction(layer1_out.reshape([1,-1]),layer2_out.reshape([1,-1]),layer3_out.reshape([1,-1]),layer4_out.reshape([1,-1]))

#Input image 2 processing
test_im2 = Image.open("/home/tandem-team/Work_Folder/Challenge_images/strawberries.jpg")
test_im2=transform(test_im2)
test_im2=torch.unsqueeze(test_im2,0)

net = models.resnet18(pretrained=True)
net.eval()
net.layer1.register_forward_hook(printnorm)
net.layer2.register_forward_hook(printnorm)
net.layer3.register_forward_hook(printnorm)
net.layer4.register_forward_hook(printnorm)
net(test_im2)
layer1_out2 = pool_op(visualisation[4].squeeze(0))
layer2_out2 = pool_op(visualisation[5].squeeze(0))
layer3_out2 = pool_op(visualisation[6].squeeze(0))
layer4_out2 = pool_op(visualisation[7].squeeze(0))


#Final output vector for second image
final_out2 = torch.ops.Pytorch_ONNX_ex.reduction(layer1_out2.reshape([1,-1]),layer2_out2.reshape([1,-1]),layer3_out2.reshape([1,-1]),layer4_out2.reshape([1,-1]))
print(final_out1.size())
print(final_out2.size())
#Cosine similarity between the two vectors
cos = nn.CosineSimilarity(dim=0, eps=1e-06)
vec_out=cos(final_out1, final_out2)

print(vec_out)