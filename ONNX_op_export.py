def reduction(g, layerOne, layerTwo, layerThree, layerFour):
    return g.op("Pytorch_ONNX_ex::reduction", layerOne, layerTwo, layerThree, layerFour)

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic("Pytorch_ONNX_ex::reduction", reduction, 9)


import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2
import argparse
from torchvision import models, transforms
from PIL import Image

torch.ops.load_library("/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/build/lib.linux-aarch64-3.6/rlcustom_operator.cpython-36m-aarch64-linux-gnu.so")
visualisation=[]

def hook_attach_func(self, input_t, output):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    visualisation.append(output.data)

def export_custom_op():
    class CustomModel(torch.nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.transform = transforms.Compose([            #[1]
                                            transforms.ToPILImage(),
                                            transforms.Resize(256),                    #[2]
                                            transforms.CenterCrop(224),                #[3]
                                            transforms.ToTensor(),                     #[4]
                                            transforms.Normalize(                      #[5]
                                            mean=[0.485, 0.456, 0.406],                #[6]
                                            std=[0.229, 0.224, 0.225]),                  #[7]
                                            
                                            ])
            self.net = models.resnet18(pretrained=True)
            self.net.layer1.register_forward_hook(hook_attach_func)
            self.net.layer2.register_forward_hook(hook_attach_func)
            self.net.layer3.register_forward_hook(hook_attach_func)
            self.net.layer4.register_forward_hook(hook_attach_func)

            self.pool_op = nn.AdaptiveAvgPool2d((1,1))

            self.cos = nn.CosineSimilarity(dim=0, eps=1e-06)
        
        def forward(self, input_img1, input_img2):
            input_img1 = self.transform(input_img1)
            input_img2 = self.transform(input_img2)

            self.net(input_img1.unsqueeze(0))
            layerOne1 = self.pool_op(visualisation[0].squeeze(0))
            layerTwo1 = self.pool_op(visualisation[1].squeeze(0))
            layerThree1 = self.pool_op(visualisation[2].squeeze(0))
            layerFour1 = self.pool_op(visualisation[3].squeeze(0))
            
            self.net(input_img2.unsqueeze(0))
            layerOne2 = self.pool_op(visualisation[4].squeeze(0))
            layerTwo2 = self.pool_op(visualisation[5].squeeze(0))
            layerThree2 = self.pool_op(visualisation[6].squeeze(0))
            layerFour2 = self.pool_op(visualisation[7].squeeze(0))


            return [torch.ops.Pytorch_ONNX_ex.reduction(layerOne1.reshape([1,-1]), layerTwo1.reshape([1,-1]), layerThree1.reshape([1,-1]), layerFour1.reshape([1,-1])),\
                            torch.ops.Pytorch_ONNX_ex.reduction(layerOne2.reshape([1,-1]), layerTwo2.reshape([1,-1]), layerThree2.reshape([1,-1]), layerFour2.reshape([1,-1]))]



    input_img1 = torch.from_numpy(np.zeros([3, 400, 400]))
    input_img2 = torch.from_numpy(np.ones([3, 400, 400]))
    
    inputs = (input_img1, input_img2)

    f = './model_first.onnx'
    torch.onnx.export(CustomModel(), inputs, f,
                       opset_version=9,
                       example_outputs=None,
                       input_names=["input_img1", "input_img2"], output_names=["Y", "Z"],
                       custom_opsets={"Pytorch_ONNX_ex": 9})

export_custom_op()
