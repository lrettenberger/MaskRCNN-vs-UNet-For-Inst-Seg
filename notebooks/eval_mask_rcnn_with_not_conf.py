import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label


from DLIP.utils.loading.initialize_wandb import initialize_wandb
from DLIP.utils.loading.load_data_module import load_data_module
from DLIP.utils.loading.load_model import load_model
from DLIP.utils.loading.load_trainer import load_trainer
from DLIP.utils.loading.merge_configs import merge_configs
from DLIP.utils.loading.parse_arguments import parse_arguments
from DLIP.utils.loading.prepare_directory_structure import prepare_directory_structure
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.cross_validation.cv_trainer import CVTrainer

OVERLAPPING_THRESHOLD = 0.3
MIN_ACTIVATION = 0.45
# conf: lower_threshold = THRESHOLD_CONF, high_threshold = 1.0
THRESHOLD_CONF = 0.9
# not_conf: lower_threshold = THRESHOLD_NOT_CONF, high_threshold = THRESHOLD_CONF
THRESHOLD_NOT_CONF = 0.6 

def post_process_preds(ort_outs,overlapping_threshold,min_activation,lower_threshold,high_threshold):
    scores = ort_outs['scores'].detach().numpy()
    masks = ort_outs['masks'].detach().numpy()
    masks = masks[(scores > lower_threshold) & (scores <= high_threshold)]
    scores = scores[(scores > lower_threshold) & (scores <= high_threshold)]
    final_mask = np.zeros((1024, 1024), dtype=np.float32)
    local_instance_number = 1    
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
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

logging.basicConfig(level=logging.INFO)
logging.info("Initalizing model")

config_files = '/home/ws/kg2371/projects/sem-segmentation/DLIP/experiments/configurations/inst_seg_low_mag_maskedrcnn.yaml'
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

model = load_model(parameters_splitted["model"],
                 checkpoint_path_str='/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/MaskedRCNN/0865/dnn_weights.ckpt'
)

data = load_data_module(parameters_splitted["data"])
trainer = load_trainer(parameters_splitted['train'], experiment_dir, wandb.run.name, data)


x = torch.zeros((0,1,1024,1024))
y_true = torch.zeros((0,1024,1024,1))
y_true_not_confident = torch.zeros((0,1024,1024,1))
y_pred = []

model.eval()
for batch in tqdm(data.test_dataloader()):
    x_step, y_true_step, y_true_not_confident_step   = batch
    y_true_not_confident_filtered = torch.where(y_true_step==0,torch.unsqueeze(y_true_not_confident_step,3),torch.zeros_like(y_true_step))
    x = torch.concat((x,x_step))
    y_true = torch.concat((y_true,y_true_step))
    y_true_not_confident = torch.concat((y_true_not_confident,y_true_not_confident_filtered))
    y_pred_step = model(x_step,None)
    y_pred.extend(y_pred_step)

f = open(f"/home/ws/kg2371/projects/sem-segmentation/hypersearch/hyperparams_big_imgs.txt", "a")
f.write('overlapping_threshold,min_activation,threshold_conf,threshold_not_conf,aji_conf,aji_not_conf \n')
for overlapping_threshold in tqdm(np.arange(0.0,0.5,0.05)):
    for min_activation in tqdm(np.arange(0.2,0.7,0.05)):
        for threshold_conf in np.arange(0.7,0.99,0.05):
            for threshold_not_conf in np.arange(0.4,threshold_conf,0.05):
                ajis_confident = []
                ajis_not_confident = []
                for k in range(len(x)):
                    masks_summed_confident = post_process_preds(y_pred[k],overlapping_threshold,min_activation,threshold_conf,1.0)
                    masks_summed_not_confident = post_process_preds(y_pred[k],overlapping_threshold,min_activation,threshold_not_conf,threshold_conf)
                    try:
                        aji_confident = get_fast_aji_plus(y_true[k].detach().cpu().numpy().astype(np.uint8),np.expand_dims(masks_summed_confident.astype(np.uint8),2))
                        ajis_confident.append(aji_confident)
                        aji_not_confident = get_fast_aji_plus(remap_label(y_true_not_confident[k].detach().cpu().numpy().astype(np.uint8)),np.expand_dims(masks_summed_not_confident.astype(np.uint8),2))
                        ajis_not_confident.append(aji_not_confident)
                    except Exception as e:
                        print(e)
                f.write(f'{overlapping_threshold},{min_activation},{threshold_conf},{threshold_not_conf},{np.mean(np.array(ajis_confident))},{np.mean(np.array(ajis_not_confident))}\n')
                f.flush()
f.close()