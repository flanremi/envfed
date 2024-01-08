import json
import time

from sklearn.datasets import load_iris  # 鸢尾花数据集
from sklearn.decomposition import PCA

import train_class
import numpy as np

def generate_metadata():
    type_names = ["crossing", "high_way", "main_road", "total"]
    val_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results\\{}\\client{}\\train\\results.csv"
    model_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results\\{}\\client{}\\train\\weights\\epoch127.pt"
    target_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results\\"
    results = {}
    for type_name in type_names:
        t_list = []
        for i in range(25):
            with open(val_url.format(type_name, i), "r") as file:
                contents = file.read().split("\n")[1:129]
            t_val = ""
            for content in contents:
                t_val += content + "\n"
            client = {"pos": i,
                      "type": type_name,
                      "model_url": model_url.format(type_name, i),
                      "PCA25": [-1 for i in range(100)],
                      "val": t_val,
                      }
            t_list.append(client)
        results.update({type_name: t_list})
    with open(target_url + "metadata", "w+") as file:
        file.write(json.dumps(results))

def generate_metadata2():
    type_names = ["crossing", "high_way", "main_road", "total"]
    val_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\{}\\client{}\\train\\weights\\val.txt"
    model_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\{}\\client{}\\train\\weights\\last.pt"
    target_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\"
    results = {}
    for type_name in type_names:
        t_list = []
        for i in range(25):
            with open(val_url.format(type_name, i), "r") as file:
                t_val = json.loads(file.read())
            client = {"pos": i,
                      "type": type_name,
                      "model_url": model_url.format(type_name, i),
                      "PCA25": [-1 for i in range(100)],
                      "val": t_val,
                      }
            t_list.append(client)
        results.update({type_name: t_list})
    with open(target_url + "metadata", "w+") as file:
        file.write(json.dumps(results))


def getPCA25():
    meta_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\"
    yolov5s_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yolov5s.pt"
    with open(meta_url + "metadata", "r") as file:
        metadata = json.loads(file.read())
    for key, value in metadata.items():
        tmp = [train_class.getParamlistByModel(value[i]["model_url"]) for i in range(25)]
        # tmp.append(train_class.getParamlistByModel(yolov5s_url))
        tmp = np.array(tmp)
        pca = PCA(n_components=25)  # 降到25维
        t = time.time()
        results = pca.fit_transform(tmp).tolist()
        for pos, result in enumerate(results):
            # if pos == 25:
            #     value.append({
            #         "pos": 25,
            #         "type": value[0].get("type"),
            #         "model_url": yolov5s_url,
            #         "PCA25": result,
            #     })
            # else:
            value[pos].update({"PCA25": result})
        print(time.time() - t)
    with open(meta_url + "metadata", "w") as file:
        file.write(json.dumps(metadata))

generate_metadata2()
getPCA25()
# meta_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results\\"
# with open(meta_url + "metadata", "r") as file:
#     metadata = json.loads(file.read())
#     print(1)