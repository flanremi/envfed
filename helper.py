import json
import os
import random
import shutil
import time

from sklearn.datasets import load_iris  # 鸢尾花数据集
from sklearn.decomposition import PCA

# import train_class
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


def get_all_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


# 划分val
def split_val():
    file_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\clinet{}\\{}\\"
    val_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\clinet{}_val\\"
    type_names = ["crossing", "high_way", "main_road", "total"]
    fold_names = ["images", "labels"]
    for _type_name in type_names:
        for i in range(25):
            images = get_all_filenames(file_url.format(_type_name, i, fold_names[0]))
            if not os.path.exists(val_url.format(_type_name, i)):
                os.makedirs(val_url.format(_type_name, i))
                os.makedirs(val_url.format(_type_name, i) + fold_names[0])
                os.makedirs(val_url.format(_type_name, i) + fold_names[1])
            for pos, image in enumerate(images):
                if pos % 10 == 5:
                    # print(file_url.format(_type_name, i, fold_names[1]) + image[image.rfind("\\") + 1:image.rfind(".")] + ".txt"
                    #             , val_url.format(_type_name, i) + fold_names[1] + "\\" + image[image.rfind("\\") + 1: image.rfind(".")] + ".txt")
                    shutil.move(image,
                                val_url.format(_type_name, i) + fold_names[0] + "\\" + image[image.rfind("\\") + 1:])
                    shutil.move(file_url.format(_type_name, i, fold_names[1]) + image[image.rfind("\\") + 1:image.rfind(
                        ".")] + ".txt"
                                , val_url.format(_type_name, i) + fold_names[1] + "\\" + image[image.rfind(
                            "\\") + 1: image.rfind(".")] + ".txt")
                    # a = 1 / 0


def move_val():
    file_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\clinet{}\\{}\\"
    val_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\clinet{}_val\\"


# split_val()
# generate_metadata2()
# getPCA25()
# meta_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results\\"
# with open(meta_url + "metadata", "r") as file:
#     metadata = json.loads(file.read())
#     print(1)

# 把fastyolo的数据集转为yolo数据集
def fast_2_yolo5():
    root_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_crossing\\client{}\\"
    for i in range(10):
        for ba in ["", "_val"]:
            tmp_url = root_url.format(str(i) + ba)
            if not os.path.exists(tmp_url + "images\\"):
                os.makedirs(tmp_url + "images\\")
            if not os.path.exists(tmp_url + "labels\\"):
                os.makedirs(tmp_url + "labels\\")
            files = get_all_filenames(tmp_url)
            for file in files:
                if file.find("jpg") != -1:
                    shutil.move(file, tmp_url + "images\\")
                if file.find("txt") != -1:
                    shutil.move(file, tmp_url + "labels\\")


def val_2_yolo5():
    root_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_crossing_val\\"
    if not os.path.exists(root_url + "images\\"):
        os.makedirs(root_url + "images\\")
    if not os.path.exists(root_url + "labels\\"):
        os.makedirs(root_url + "labels\\")
    files = get_all_filenames(root_url)
    for file in files:
        if file.find("jpg") != -1:
            shutil.move(file, root_url + "images\\")
        if file.find("txt") != -1:
            shutil.move(file, root_url + "labels\\")


def create_yml():
    url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_high_way\\client{}\\images\\"
    # val_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_main_road\\client{}_val\\images\\"
    val_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_high_way_val\\images\\"
    for i in range(10):
        body = ""
        body += "train: " + url.format(i) + "\n"
        body += "val: " + val_url.format(i) + "\n"
        body += "names: " + "\n"
        body += "  0: car" + "\n"
        body += "  1: bus" + "\n"
        body += "  2: traffic light" + "\n"
        body += "  3: traffic sign" + "\n"
        body += "  4: person" + "\n"
        body += "  5: scooter" + "\n"
        body += "  6: bicycle" + "\n"
        body += "  7: bus station" + "\n"
        body += "  8: bridge opening" + "\n"
        with open("./yml/high_way/client{}.yml".format(i), "w") as file:
            file.write(body)


# 将模型汇聚到一起
def aggregate():
    target_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\main_road\\"
    for i in range(25):
        for back in ['', '_val']:
            tmp_url = target_url + "clinet" + str(i) + back
            files = get_all_filenames(tmp_url)
            for file in files:
                if file.find("labels.cache") == -1:
                    shutil.move(file, target_url)


def generate_yolo5_data():
    url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\main_road\\"
    target_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_main_road\\"
    val_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_main_road_val\\"
    files = get_all_filenames(url)
    random.shuffle(files)
    pos = 0
    # 筛选出测试集
    for file in files:
        if file.find("jpg") != -1:
            shutil.move(file, val_url + "images\\")
            shutil.move(file[:file.rfind(".")] + ".txt", val_url + "labels\\")
            pos += 1
        if pos == 80:
            break
    # 分组
    files = get_all_filenames(url)
    client_data_num = int(len(files) / 10 * 85 / 100 / 2)
    val_data_num = int(len(files) / 10 * 15 / 100 / 2)
    pos = 0
    for i in range(10):
        if not os.path.exists(target_url + "client" + str(i)):
            os.makedirs(target_url + "client" + str(i))
            os.makedirs(target_url + "client" + str(i) + "\\labels\\")
            os.makedirs(target_url + "client" + str(i) + "\\images\\")
            os.makedirs(target_url + "client" + str(i) + "_val\\images\\")
            os.makedirs(target_url + "client" + str(i) + "_val\\labels\\")
    for file in files:
        i = int(pos / (client_data_num + val_data_num))
        if i >= 10:
            break
        if file.find("jpg") != -1:
            shutil.move(file, target_url + "client" + str(i))
            shutil.move(file[:file.rfind(".")] + ".txt", target_url + "client" + str(i))
            pos += 1
    for i in range(10):
        _pos = 0
        tmp = target_url + "client" + str(i)
        files = get_all_filenames(tmp)
        random.shuffle(files)
        for file in files:
            if file.find(".jpg") != -1:
                if _pos >= val_data_num:
                    shutil.move(file, target_url + "client" + str(i) + "\\images\\")
                    shutil.move(file[:file.rfind(".")] + ".txt", target_url + "client" + str(i) + "\\labels\\")
                else:
                    shutil.move(file, target_url + "client" + str(i) + "_val\\images\\")
                    shutil.move(file[:file.rfind(".")] + ".txt", target_url + "client" + str(i) + "_val\\labels\\")
                _pos += 1


def back():
    url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_high_way\\client{}\\"
    target = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\high_way\\"
    for i in range(10):
        for b in ["", "_val"]:
            t = url.format(str(i) + b)
            files = get_all_filenames(t)
            for f in files:
                shutil.move(f, target)


def back2():
    for i in range(10):
        for b in ["labels", "images"]:
            tmp = (
                "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_high_way\\client{}\\{}"
                .format(i, b))
            files = get_all_filenames(tmp)
            for file in files:
                shutil.move(file,
                            "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_high_way\\client{}\\".format(
                                i))


# def create_total():
#     target = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_total\\"
#     target_val = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_total_val\\"
#     from_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_{}\\client4\\"
#     from_val_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_{}\\client4_val\\"
#     _types = ["crossing","main_road","high_way"]
#     _back = ["images", "labels"]
#     files = [get_all_filenames(from_url.format(_types[i]) + _back[0]) for i in range(3)]
#     files2 = [get_all_filenames(from_url.format(_types[i]) + _back[1]) for i in range(3)]
#
#     for i in range(240):
#         file = files[i % 3][i]
#         file2 = files2[i % 3][i]
#         shutil.copy(file, target + _back[0])
#         shutil.copy(file2, target + _back[1])
#
#     files = [get_all_filenames(from_val_url.format(_types[i]) + _back[0]) for i in range(3)]
#     files2 = [get_all_filenames(from_val_url.format(_types[i]) + _back[1]) for i in range(3)]
#     for i in range(36):
#         file = files[i % 3][i]
#         file2 = files2[i % 3][i]
#         shutil.copy(file, target_val + _back[0])
#         shutil.copy(file2, target_val + _back[1])

def delete_img():
    _types = ["crossing", "main_road", "high_way"]
    url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_{}\\client{}\\{}\\"
    back = ["images", "labels"]
    num = 50
    for _t in _types:
        for p, pos in enumerate([3]):
            tmp = url.format(_t, pos, back[1])
            labels = get_all_filenames(tmp)
            random.shuffle(labels)
            if len(labels) > num:
                # 清除
                for i in range(num, len(labels)):
                    label = labels[i]
                    image = url.format(_t, pos, back[0]) + label[label.rfind("\\") + 1: label.rfind(".")] + ".jpg"
                    os.remove(image)
                    os.remove(label)

                # labels = get_all_filenames(tmp)
                # # 減少標記
                # for i in range(len(labels)):
                #     label = labels[i]
                #     image = url.format(_t, pos, back[0]) + label[label.rfind("\\") + 1: label.rfind(".")] + ".jpg"
                #     with open(label, "r") as file:
                #         content = file.read()
                #     content = content.split("\n")
                #     t_content = ""
                #     for t in content:
                #         if random.randint(0, 1) == 0:
                #             continue
                #         t_content += t + "\n"
                #     if len(t_content) > 5:
                #         with open(label, "w") as file:
                #             file.write(t_content[:-1])
                #     else:
                #         os.remove(label)
                #         os.remove(image)

def check_img():
    _types = ["crossing", "main_road", "high_way"]
    url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\yolo5_{}\\client{}\\{}\\"
    back = ["images", "labels"]
    for _t in _types:
        for i in range(10):
            images = get_all_filenames(url.format(_t, i, back[0]))
            for image in images:
                label = url.format(_t, i, back[1]) + image[image.rfind("\\") + 1 : image.rfind(".")] + ".txt"
                if not os.path.exists(label):
                    # os.remove(image)
                    print(label)
            labels = get_all_filenames(url.format(_t, i, back[1]))
            images = get_all_filenames(url.format(_t, i, back[0]))
            print("{}---{}--{}---{}".format(_t, i, len(labels), len(images)))



if __name__ == '__main__':
    delete_img()
    check_img()
    # aggregate()
    # generate_yolo5_data()
    # back()
    # back2()
    # create_yml()
