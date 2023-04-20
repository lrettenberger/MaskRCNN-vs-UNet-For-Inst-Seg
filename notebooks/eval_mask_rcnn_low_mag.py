import os
import wandb
import logging
from pytorch_lightning.utilities.seed import seed_everything
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

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

OVERLAPPING_THRESHOLD = 0.3
MIN_ACTIVATION = 0.45
# conf: lower_threshold = THRESHOLD_CONF, high_threshold = 1.0
THRESHOLD_CONF = 0.9
# not_conf: lower_threshold = THRESHOLD_NOT_CONF, high_threshold = THRESHOLD_CONF
THRESHOLD_NOT_CONF = 0.55


def post_process_preds(ort_outs,overlapping_threshold,min_activation,lower_threshold,high_threshold):
    scores = ort_outs['scores'].detach().numpy()
    masks = ort_outs['masks'].detach().numpy()
    masks = masks[(scores > lower_threshold) & (scores <= high_threshold)]
    scores = scores[(scores > lower_threshold) & (scores <= high_threshold)]
    final_mask = np.zeros((masks[0].shape[1], masks[0].shape[2]), dtype=np.float32)
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
                 checkpoint_path_str=f'/home/ws/kg2371/projects/sem-segmentation/results/first-shot/GenericSegmentationDataModule/MaskedRCNN/0906/dnn_weights.ckpt'

)

data = load_data_module(parameters_splitted["data"])
trainer = load_trainer(parameters_splitted['train'], experiment_dir, wandb.run.name, data)

x = torch.zeros((0,1,1024,1024))
y_true = torch.zeros((0,1024,1024,1))
y_true_not_confident = torch.zeros((0,1024,1024,1))
y_pred = []

model.eval()
for batch in tqdm(data.test_dataloader()):
    if len(batch) == 3:
        x_step, y_true_step, y_true_not_confident_step   = batch
        y_true_not_confident_filtered = torch.where(y_true_step==0,torch.unsqueeze(y_true_not_confident_step,3),torch.zeros_like(y_true_step))
        y_true_not_confident = torch.concat((y_true_not_confident,y_true_not_confident_filtered))
    elif len(batch) == 2:
        x_step, y_true_step   = batch
    x = torch.concat((x,x_step))
    y_true = torch.concat((y_true,y_true_step))
    y_pred_step = model(x_step,None)
    y_pred.extend(y_pred_step)
    
    
    
i=5

pred_masks_summed_confident = post_process_preds(y_pred[i],OVERLAPPING_THRESHOLD,MIN_ACTIVATION,THRESHOLD_CONF,1.0)

aji_confident = get_fast_aji_plus(remap_label(y_true[i].detach().cpu().numpy().astype(np.uint8)),np.expand_dims(pred_masks_summed_confident.astype(np.uint8),2))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(18.5, 10.5)

ax1.imshow(x[i].permute(1,2,0),cmap='gray')
ax1.set_title('x')
ax1.axis('off')

ax2.imshow(pred_masks_summed_confident)
ax2.set_title(f'Pred Confident (aji+ {aji_confident:.2f})')
ax2.axis('off')

ax3.imshow(y_true[i])
ax3.set_title('True Confident')
ax3.axis('off')