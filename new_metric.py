import tifffile
import numpy as np
from glob import glob
import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything
import numpy as np
import torch
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything
import numpy as np
import torch
import wandb
import pandas as pd 
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes

from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer
from DLIP.utils.post_processing.distmap2inst import DistMapPostProcessor


def get_cfg_clusters(run_lst, cluster_cfg_lst):
    run_lst_cp = deepcopy(run_lst)
    cfgs= [run_lst_cp[ix].config for ix in range(len(run_lst_cp))]

    for ix in range(len(cfgs)):
        for key in cluster_cfg_lst:
            del cfgs[ix][key]

    selected_id = list()
    cluster_lst = list()

    for ix in range(len(cfgs)):
        if ix not in selected_id:
            cluster_lst.append([ix])
            selected_id.append(ix)
            for iy in range(ix+1,len(cfgs)):
                if cfgs[ix]==cfgs[iy]:
                    cluster_lst[-1].append(iy)
                    selected_id.append(iy)

    
    return cluster_lst


post_pro = DistMapPostProcessor(
    sigma_cell=1.0,
    th_cell=0.03,
    th_seed=0.6,
    do_splitting=False,
    do_area_based_filtering=False,
    do_fill_holes=False,
    valid_area_median_factors=[0.25,3]
)

metrics = {}
for label in glob('/home/ws/kg2371/datasets/isbi14_challenge/all_test/labels/*.tiff'):
    mask = tifffile.imread(label)
    max_val = np.sum((mask.sum(0)>0*1.)*len(mask))
    min_val = np.sum(mask.sum(0)>0*1.)
    val = np.sum(mask.sum(0))
    metric = (val-min_val) / (max_val-min_val)
    metric = metric * (len(mask)/10)
    metrics[metric] = label.split('/')[-1]


config_files = '/home/ws/kg2371/projects/sem-segmentation/DLIP/experiments/configurations/bmt_unet.yaml'
result_dir = './'

cfg_yaml = merge_configs(config_files)
base_path=os.path.expandvars(result_dir)
experiment_name=cfg_yaml['experiment.name']['value']

cfg_yaml['wandb.mode']['value'] = 'disabled'

experiment_dir, config_name = prepare_directory_structure(
    base_path=base_path,
    experiment_name=experiment_name,
    data_module_name=cfg_yaml['data.datamodule.name']['value'],
    model_name=cfg_yaml['model.name']['value']
)

config = initialize_wandb(
    cfg_yaml=cfg_yaml,
    experiment_dir=experiment_dir,
    config_name=config_name
)

config.update({'data.datamodule.arguments.root_dir':f"{config['prefix']}/{config['root_dir_base']}"},allow_val_change=True) 

seed_everything(seed=cfg_yaml['experiment.seed']['value'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])

from tqdm import tqdm

api = wandb.Api()
runs = api.sweep("lucare/bmt_unet_vs_maskedrcnn/4g3qp5dp").runs


cluster_lst_seed = get_cfg_clusters(runs, ["experiment.seed"])

all_ajis = {}
for cluster in tqdm(cluster_lst_seed):
    print(runs[cluster[0]].config['root_dir_base'])
    ajis_cluster = []
    for run_id in cluster:
        run_i = runs[run_id]
        weights_path = f'/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/UnetInstance/{run_i.name.split("_")[-1]}/dnn_weights.ckpt'
        model = load_model(parameters_splitted["model"],checkpoint_path_str=weights_path)
        model.eval()
        ajis = []
        for key in tqdm(sorted(metrics.keys())):
            x = tifffile.imread(f'/home/ws/kg2371/datasets/isbi14_challenge/all_test/samples/{metrics[key]}')
            y_true = tifffile.imread(f'/home/ws/kg2371/datasets/isbi14_challenge/all_test/labels/{metrics[key]}')
            y_true_masks = np.stack([x for x in y_true if np.sum(x) > 0])
            y_true_masks = sorted(y_true_masks,key=lambda x: np.sum(x))
            y_true_masks_summed = np.zeros_like(y_true[0])*0.
            for j in range(len(y_true_masks)):
                y_true_masks_summed[y_true_masks[j] > 0] = j+1
            y_true = y_true_masks_summed
            y_pred = model(torch.tensor(x).unsqueeze(0).unsqueeze(0) / 255.)[0,0].detach().cpu().numpy()
            y_pred = post_pro.process(y_pred,None)
            aji = get_fast_aji_plus(remap_label(y_true),remap_label(y_pred))
            ajis.append(aji)
        ajis_cluster.append(ajis)
    all_ajis[runs[cluster[0]].config['root_dir_base']] = ajis_cluster
    

config_files = '/home/ws/kg2371/projects/sem-segmentation/DLIP/experiments/configurations/bmt_maskedrcnn.yaml'
result_dir = './'

cfg_yaml = merge_configs(config_files)
base_path=os.path.expandvars(result_dir)
experiment_name=cfg_yaml['experiment.name']['value']

cfg_yaml['wandb.mode']['value'] = 'disabled'

experiment_dir, config_name = prepare_directory_structure(
    base_path=base_path,
    experiment_name=experiment_name,
    data_module_name=cfg_yaml['data.datamodule.name']['value'],
    model_name=cfg_yaml['model.name']['value']
)

config = initialize_wandb(
    cfg_yaml=cfg_yaml,
    experiment_dir=experiment_dir,
    config_name=config_name
)

config.update({'data.datamodule.arguments.root_dir':f"{config['prefix']}/{config['root_dir_base']}"},allow_val_change=True) 

seed_everything(seed=cfg_yaml['experiment.seed']['value'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])


MIN_MASK_SIZE_THESHOLD = 10

OVERLAPPING_THRESHOLD = 0.41
MIN_ACTIVATION = 0.53
# conf: lower_threshold = THRESHOLD_CONF, high_threshold = 1.0
THRESHOLD_CONF = 0.8
# not_conf: lower_threshold = THRESHOLD_NOT_CONF, high_threshold = THRESHOLD_CONF
THRESHOLD_NOT_CONF = 0.55

def post_process_masked_rcnn(pred,lower_threshold,high_threshold,H,W):
    scores = pred['scores']
    masks = pred['masks']   
    masks = masks[(scores > lower_threshold) & (scores <= high_threshold)]
    scores = scores[(scores > lower_threshold) & (scores <= high_threshold)]
    masks_summed = torch.zeros((H,W))
    if len(masks) == 0:
        return masks_summed
    final_mask = torch.stack(sorted(masks,key=lambda x: torch.sum(x>MIN_ACTIVATION)))
    for j in range(len(final_mask)):
        masks_summed[(final_mask[j]> MIN_ACTIVATION).squeeze()] = j+1
    return masks_summed.to(torch.uint8)



from tqdm import tqdm

api = wandb.Api()
runs = api.sweep("lucare/bmt_unet_vs_maskedrcnn/1me7ydtt").runs

cluster_lst_seed = get_cfg_clusters(runs, ["experiment.seed"])

all_ajis_masked = {}
for cluster in (cluster_lst_seed):
    print(runs[cluster[0]].config['root_dir_base'])
    ajis_cluster_maskedrcnn = []
    for run_id in cluster:
        run_i = runs[run_id]
        weights_path = f'/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/MaskedRCNN/{run_i.name.split("_")[-1]}/dnn_weights.ckpt'
        model = load_model(parameters_splitted["model"],checkpoint_path_str=weights_path)
        model.eval()
        ajis = []
        for key in tqdm(sorted(metrics.keys())):
            x = tifffile.imread(f'/home/ws/kg2371/datasets/isbi14_challenge/all_test/samples/{metrics[key]}')
            y_true = tifffile.imread(f'/home/ws/kg2371/datasets/isbi14_challenge/all_test/labels/{metrics[key]}')
            y_true_masks = np.stack([x for x in y_true if np.sum(x) > 0])
            y_true_masks = sorted(y_true_masks,key=lambda x: np.sum(x))
            y_true_masks_summed = np.zeros_like(y_true[0])*0.
            for j in range(len(y_true_masks)):
                y_true_masks_summed[y_true_masks[j] > 0] = j+1
            y_true = y_true_masks_summed
            y_pred = model(torch.tensor(x).unsqueeze(0).unsqueeze(0) / 255.)[0]
            y_pred = post_process_masked_rcnn(y_pred,THRESHOLD_CONF,1.0,512,512).detach().cpu().numpy()
            aji = get_fast_aji_plus(remap_label(y_true),remap_label(y_pred))
            ajis.append(aji)
        ajis_cluster_maskedrcnn.append(ajis)
    all_ajis_masked[runs[cluster[0]].config['root_dir_base']] = ajis_cluster_maskedrcnn
    
import pickle

with open('all_ajis.pickle', 'wb') as handle:
    pickle.dump(all_ajis, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('all_ajis.pickle', 'rb') as handle:
#     b = pickle.load(handle)
    
with open('all_ajis_masked.pickle', 'wb') as handle:
    pickle.dump(all_ajis_masked, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('all_ajis_masked.pickle', 'rb') as handle:
#     c = pickle.load(handle)
