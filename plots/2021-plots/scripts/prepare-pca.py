"""
Usage: 
    prepare.py --dir-name=NAME --type=TYPE --output-dir=PATH
"""
from docopt import docopt
import ast
import pickle
import torch
from tqdm import tqdm
import sys
import numpy as np
sys.path.append("/home/savi/ehsan/FederatedLearning")
import matplotlib.pyplot as plt
from federated_learning.helper import utils
from sklearn.decomposition import PCA
arguments = docopt(__doc__)

def get_all_params_flattend(model_state):
    params = np.array([], dtype=np.float32)
    for ll_name, ll_data in model_state.items():
        params = np.concatenate((params, ll_data.reshape(-1)))
    return params

# from sklearn.preprocessing import StandardScaler
# def standardization(all_params):
#     all_params = StandardScaler().fit_transform(all_params)
#     return all_params

# from sklearn.preprocessing import MinMaxScaler
# def normalization(all_params):
#     norm = MinMaxScaler().fit_transform(all_params)
#     return all_params

# def draw_pca_model(params_list, colors, title):
#     pca = PCA(n_components=2)
#     proj = pca.fit_transform(params_list)
#     figure = plt.figure(figsize=(18, 8))
#     plt.title(title)
#     plt.scatter(proj[:, 0], proj[:,1], c = colors, cmap="rainbow") #gist_rainbow
  

def main(last_round, log_path, type_name, attackers_idx, round_no):
    all_params = np.array([], dtype=np.float32).reshape(-1, 431570) 
    # 431570 is the nmber of all params in a model

    attackers_num = len(attackers_idx)
    workers_idx = ["worker_{}".format(ii) for ii in range(30)]
    normal_idx = set(workers_idx) - set(attackers_idx)

    all_params = np.array([], dtype=np.float32).reshape(-1, 431080) 
    # 431080 is the nmber of all params in a model
    for ww in workers_idx:
        model_path = "{}/models/workers_R{}/{}_{}_{}_R{}".format(
            log_path, round_no,
            type_name, attackers_num, ww, round_no)
        params = get_all_params_flattend(torch.load(model_path)).reshape(1, -1)
        all_params = np.concatenate((all_params, params))
    pca = PCA(n_components=2)
    points = pca.fit_transform(all_params)
    points_with_type = []
    for ww_ii, ww in enumerate(workers_idx):
        id_ = "1" if ww in attackers_idx else "2"
        points_with_type.append([id_] + points[ww_ii].tolist())
    return points_with_type
    
if __name__ == '__main__':
    dir_name = arguments['--dir-name']
    output_dir = arguments['--output-dir']
    type_name = arguments['--type']
    utils.check_create_dir(output_dir)
    prefix = "/home/savi/ehsan/FederatedLearning/data_output"
    log_path = "{}/{}".format(prefix, dir_name)
    last_round = int(utils.get_last_round_num(log_path, "accuracy"))
    attackers_idx = utils.load_object(log_path, "attackers")
    
    for round_no in tqdm(range(last_round)):
        projected_points = main(
            last_round, log_path, type_name, attackers_idx, round_no)
        output_file_name = "{}_{}_R{}.txt".format(type_name, len(attackers_idx), round_no)
        with open("{}/{}".format(output_dir, output_file_name), 'w') as f:
            for pp in projected_points:
                f.write("{} {} {}\n".format(pp[1], pp[2], pp[0]))
            f.close()
        