import math
import os
import random
import time

import numpy
import numpy as np
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

# a = np.array([1,2,3,4,5])
# a = np.pad(a, (0, 4))
# b = numpy.resize(a, (3,3))
#
# print(b)


# def get_all_filenames(folder_path):
#     filenames = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             filenames.append(os.path.join(root, file))
#     return filenames
#
#
# dir_names = ["results"]
# env_names = ["crossing", "high_way", "main_road", "total"]
# model_dir = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\{}\\{}\\{}\\train\\weights\\"
# for dir_name in dir_names:
#     for env_name in env_names:
#         for i in range(25):
#             url = model_dir.format(dir_name, env_name, "client" + str(i))
#             file_names = get_all_filenames(url)
#             for file_name in file_names:
#                 if file_name.find("epoch") != -1:
#                     if random.randint(0,1) == 1:
#                         os.remove(file_name)
