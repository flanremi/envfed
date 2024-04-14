import os
import time

import numpy as np
import torch
from sklearn.decomposition import PCA

from model_util import get_params_by_model


def  get_pca_by_model(models: list):
    tmp = []
    for model in models:
        tmp.append(get_params_by_model(model).cpu().numpy())
    tmp = np.array(tmp)
    pca = PCA(n_components=4)  # 降到10维
    results = pca.fit_transform(tmp).tolist()
    return results

