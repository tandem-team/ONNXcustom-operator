import json
import sys
import os
import time
import numpy as np
import cv2

# import onnx
import onnxruntime
# from onnx import numpy_helper

path1 = "/home/tandem-team/Work_Folder/Challenge_images/3.jpg"
path2 = "/home/tandem-team/Work_Folder/Challenge_images/4.jpg"
img1, im2 = cv2.imread(path1), cv2.imread(path2)

model = "/home/tandem-team/Work_Folder/Pytorch_ONNX_ex/model_first.onnx"
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(input_name)
print(output_name)