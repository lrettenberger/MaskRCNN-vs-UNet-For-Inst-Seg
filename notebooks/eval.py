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
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

OVERLAPPING_THRESHOLD = 0.41
MIN_ACTIVATION = 0.53
# conf: lower_threshold = THRESHOLD_CONF, high_threshold = 1.0
THRESHOLD_CONF = 0.8
# not_conf: lower_threshold = THRESHOLD_NOT_CONF, high_threshold = THRESHOLD_CONF
THRESHOLD_NOT_CONF = 0.55

def post_process_preds(ort_outs,overlapping_threshold,min_activation,lower_threshold,high_threshold):
    scores = ort_outs['scores'].detach().numpy()
    masks = ort_outs['masks'].detach().numpy()
    C,H,W = masks[0].shape
    masks = masks[(scores > lower_threshold) & (scores <= high_threshold)]
    scores = scores[(scores > lower_threshold) & (scores <= high_threshold)]
    final_mask = np.zeros((H,W), dtype=np.float32)
    local_instance_number = 1    
    for i in range(len(masks)):
        if scores[i] == 0:
            continue
        for j in range(i+1, len(masks)):
            if scores[j] == 0:
                continue
            intersection = np.logical_and(masks[i], masks[j])
            union = np.logical_or(masks[i], masks[j])
            IOU_SCORE = np.sum(intersection) / np.sum(union)
            if IOU_SCORE > overlapping_threshold:
                scores[j] = 0
    for mask, score in zip(masks, scores):
            # as scores is already sorted        
            if score == 0:
                continue        
            mask = mask.squeeze()
            mask[mask > min_activation] = local_instance_number
            mask[mask < min_activation] = 0
            local_instance_number += 1
            temp_filter_mask = np.where(final_mask > 1, 0., 1.)
            temp_filter_mask = (final_mask < 1)*1.
            mask = mask * temp_filter_mask        
            final_mask += mask    
    return final_mask

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

seed_everything(seed=cfg_yaml['experiment.seed']['value'])
parameters_splitted = split_parameters(config, ["model", "train", "data"])




api = wandb.Api()
runs = api.sweep("lucare/bmt_unet_vs_maskedrcnn/1me7ydtt").runs


cluster_lst_seed = get_cfg_clusters(runs, ["experiment.seed"])

for cluster in cluster_lst_seed:
    ajis_cluster = []
    for run_id in cluster:
        run_i = runs[run_id]
        ajis_cluster.append(run_i.summary['test/aji'])
    print(f'{runs[cluster[0]].config["root_dir_base"]}: {np.mean(ajis_cluster)*100:.2f} +- {np.std(ajis_cluster)*100:.2f}')

