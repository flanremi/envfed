import os
import time

import torch
import torch.nn as nn

import autoencoder
import environment
from dqn import dqn_agent, dual_nstep_noisy_dqn_agent2
import train_class


def get_all_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames

yaml_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\ymls\\"

type_names = ["crossing","high_way","main_road","total"]

model_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results\\{}\\client{}\\train\\weights\\"

# def generateModelPCA(dim: int):
#     for type_name in type_names:
#         for i in range(25):
#             for j in range(128):
#                 url = model_url.format(type_name, i) + "epoch{}.pt".format(j)
#                 model = torch.load(url).state_dict()
from autoencoder import AutoCoder1




# if __name__ == '__main__':
#     for _type in environment.Type:
#         env = environment.Environment(_type)
#         for i in range(1000):
#             state = env.get_state()
#             _time = time.time()
#             action = dqn_net.take_action(state)
#             _time = time.time() - _time
#             next_state, reward, done, valid = env.next(action, _time)







# ignore_num = 7
# # 创建100个周期，4种路况的训练模型  训练yolo模型
# if __name__ == '__main__':
#     for type_name in type_names :
#         file_urls = get_all_filenames(yaml_url + type_name)
#         for pos, file_url in enumerate(file_urls):
#             if ignore_num > 0:
#                 ignore_num -= 1
#                 continue
#             opt = train_class.Opt(weights="./yolov5s.pt", device='0',
#                                       data= file_url, epochs=128,
#                                       project='results\\' + type_name + "\\" + file_url[file_url.rfind("\\") + 1: file_url.rfind(".")],  save_period=1,
#                                   noval=False
#                           )
#             helper = train_class.TrainingHelper(opt)
#             helper.main()
