import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import csv
#Loading the custom operator implemented in cpp file rlcustom_operator
torch.ops.load_library("./build/lib.linux-aarch64-3.6/rlcustom_operator.cpython-36m-aarch64-linux-gnu.so")

def Vector_from_Image(path):
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
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
    test_im1 = Image.open(path)
    test_im1= transform(test_im1)
    test_im1= torch.unsqueeze(test_im1,0)

    #Passing the image through the net stores the feature maps in variable "visualisation"
    net(test_im1)

    #Pooling operation to reduce the dimensions to [1,x] ->x is 64 or 128 or 256 or 512
    pool_op = nn.AdaptiveAvgPool2d((1,1))
    layer1_out = pool_op(visualisation[0].squeeze(0))
    layer2_out = pool_op(visualisation[1].squeeze(0))
    layer3_out = pool_op(visualisation[2].squeeze(0))
    layer4_out = pool_op(visualisation[3].squeeze(0))
    
    #Final vector from the reduction and interleave
    final_out = torch.ops.Pytorch_ONNX_ex.reduction(layer1_out.reshape([1,-1]),layer2_out.reshape([1,-1]),layer3_out.reshape([1,-1]),layer4_out.reshape([1,-1]))
    return final_out


g_path = "/home/tandem-team/Work_Folder/Challenge_images/"
output_array = np.zeros([51, 51])
cos = nn.CosineSimilarity(dim=0, eps=1e-06)

with open('Image_similarity_measure.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    for i in range(0, 51):
        print("row is : ", i)
        img_vector1 = Vector_from_Image(g_path+str(i)+'.jpg')
        for j in range(i+1, 51):
            if(j==51):
                break
            img_vector2 = Vector_from_Image(g_path+str(j)+'.jpg')
            output_array[i][j] = cos(img_vector1, img_vector2)
            output_array[j][i] = output_array[i][j]
    csv_writer.writerow(output_array)
