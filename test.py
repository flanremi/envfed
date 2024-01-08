import os
import random
import time

import torch

# print(torch.cuda.is_available())  # 判断CUDA是否可用
# if torch.cuda.is_available():
#     print("当前系统中有可用的GPU")
# else:
#     print("当前系统没有可用的GPU")
#
# # 输出所有可用的GPU设备信息
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"当前选定的设备为 {device}")
# print(f"当前设备数量为 {torch.cuda.device_count()} ")
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")
#


# model_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results\\{}\\client{}\\train\\results.csv"
#
# type_names = ["crossing","high_way","main_road","total"]
# # type_names = ["total"]
#
# filter = [0 for i in range(128)]
# for i in range(15):
#     filter[i * 3] = 1
# for i in range(41):
#     filter[i * 2 + 45 ] = 1
#


# if __name__ == '__main__':
#     for type_name in type_names :
#         for i in range(25):
#             for j in range(128):
#                     if filter[j] == 1:
#                         os.remove(model_url.format(type_name, i, j))

# with open(model_url.format(type_names[0],0), "r") as file:
#     a = file.read().split("\n")
#     print(1)
#
# b = [random.randint(0,10000) for i in range(5000010)]
#
# t = time.time()
#
# for i in range(5000000):
#     a = b[i] + b[i + 1]
#
# print(time.time() - t)
#
# t = time.time()
#
# for i in range(5000000):
#     a = b[i] + b[i + 1] + b[i + 2] + b[i + 3]
#
# print(time.time() - t)