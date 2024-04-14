import torch
import torch.nn.functional as F


def get_params_by_url(url):
    model = torch.load(url)
    tmp = []
    for key, value in model.items():
        tmp.append(value.view(-1))
    tmp = torch.cat(tmp)
    # tmp = F.pad(tmp, (0, 262144-251664), "constant", 0)
    # tmp = tmp.reshape([1,262144])
    return tmp


def get_params_by_model(model):
    tmp = []
    for key, value in model.items():
        tmp.append(value.view(-1))
    tmp = torch.cat(tmp)
    # tmp = F.pad(tmp, (0, 262144-251664), "constant", 0)
    # tmp = tmp.reshape([1,262144])
    return tmp


# step 从0开始
def sum_model(old_model, new_model, step):
    tmp_dict = {}
    for key, var in old_model.items():
        old = var.clone()
        new = new_model[key].clone()
        _r = old * ((step + 1) / (step + 2)) + new * (1 / (step + 2))
        tmp_dict.update({key: _r})
    return tmp_dict
