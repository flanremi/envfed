import os

import train_class

# if __name__ == '__main__':
#     opt = train_class.Opt(weights=train_class.ROOT / 'runs\\train\\train7\\weights\\best.pt', device='0',
#                           data=train_class.ROOT / 'data\\tank.v5i.yolov8\\data.yaml', epochs=1,
#                           project='site3/client1/', name='client1From3', save_period=1
#               )
#     helper = train_class.TrainingHelper(opt)
#     helper.main()
#     a = helper.weight["model"].state_dict()
#     for key, value in a.items():
#         print(2)
#     print(1)

def get_all_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames

yaml_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\ymls\\"

type_names = ["crossing","high_way","main_road","total"]

ignore_num = 0
# 创建100个周期，4种路况的训练模型
if __name__ == '__main__':
    for type_name in type_names :
        file_urls = get_all_filenames(yaml_url + type_name)
        for pos, file_url in enumerate(file_urls):
            if ignore_num > 0:
                ignore_num -= 1
                continue
            opt = train_class.Opt(weights="./yolov5s.pt", device='0',
                                      data= file_url, epochs=128,
                                      project='results2\\' + type_name + "\\" + file_url[file_url.rfind("\\") + 1: file_url.rfind(".")],  save_period=32,
                                  noval=False
                          )
            helper = train_class.TrainingHelper(opt)
            helper.main()
