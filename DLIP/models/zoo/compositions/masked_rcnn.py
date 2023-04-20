import pytorch_lightning as pl
import torchvision

import torch
import wandb
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from DLIP.models.zoo.compositions.base_composition import BaseComposition
from DLIP.utils.metrics.inst_seg_metrics import get_fast_aji_plus, remap_label
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes

# i=15
# cv2.imwrite('boxes.png',draw_bounding_boxes((x[i]*255).to(torch.uint8),preds[i]['boxes'][preds[i]['scores']>0.5]).permute(1,2,0).detach().cpu().numpy())

MIN_MASK_SIZE_THESHOLD = 10

OVERLAPPING_THRESHOLD = 0.41
MIN_ACTIVATION = 0.53
# conf: lower_threshold = THRESHOLD_CONF, high_threshold = 1.0
THRESHOLD_CONF = 0.8
# not_conf: lower_threshold = THRESHOLD_NOT_CONF, high_threshold = THRESHOLD_CONF
THRESHOLD_NOT_CONF = 0.55

def post_process(pred,lower_threshold,high_threshold,H,W):
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


class MaskedRCNN(BaseComposition):
    def __init__(self, add_not_confident_labels, sample_labels_from_gaussian, num_classes=2, hidden_layer=25,*args, **kwargs):
        super().__init__()
        self.detector = torchvision.models.detection.maskrcnn_resnet50_fpn(
            num_classes=num_classes,
            pretrained=False, 
            min_size=512,
            max_size=512,
            image_mean=[0,0,0], # disable norm
            image_std=[1,1,1],
        )
        
        self.add_not_confident_labels = add_not_confident_labels
        self.sample_labels_from_gaussian = sample_labels_from_gaussian


    def forward(self, x,targets=None):
        return self.detector(x,targets)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y_true, y_true_not_confident   = batch
            y_true_not_confident_filtered = torch.where(y_true==0,torch.unsqueeze(y_true_not_confident,3).cuda(),torch.zeros_like(y_true))
        elif len(batch) == 2:
            x, y_true = batch
        else:
            raise Exception('Wrong dimension for training batch')
        targets = []
        for k in range(len(y_true)):
            y_true_k = torch.stack([x for x in y_true[k] if torch.sum(x) > 0])
            if torch.sum(y_true_k) == 0 or len([(y_true_k==i)*1 for i in range(1,int(torch.max(y_true_k)+1)) if torch.sum((y_true_k==i)*1) > MIN_MASK_SIZE_THESHOLD]) == 0:
                # negative sample
                masks = torch.zeros(0, 100, 100, dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)
            else:
                #masks = torch.stack([(y_true_k==i)*1 for i in range(1,int(torch.max(y_true_k)+1)) if torch.sum((y_true_k==i)*1) > MIN_MASK_SIZE_THESHOLD])
                masks = y_true_k
                boxes =  masks_to_boxes(masks)
                if self.sample_labels_from_gaussian:
                    labels = torch.normal(mean=torch.ones((len(masks))), std=torch.ones((len(masks)))*0.02)
                else:
                    labels = torch.ones((len(masks)))
                
                if self.add_not_confident_labels:
                    if torch.sum(y_true_not_confident_filtered[k]) > 0 and len([(y_true_not_confident_filtered[k]==i)*1 for i in range(1,int(torch.max(y_true_not_confident_filtered[k])+1)) if torch.sum((y_true_not_confident_filtered[k]==i)*1) > MIN_MASK_SIZE_THESHOLD]) > 0:
                        masks_not_conf = torch.stack([(y_true_not_confident_filtered[k]==i)*1 for i in range(1,int(torch.max(y_true_not_confident_filtered[k])+1)) if torch.sum((y_true_not_confident_filtered[k]==i)*1) > MIN_MASK_SIZE_THESHOLD])
                        boxes_not_conf =  masks_to_boxes(masks_not_conf[:,:,:,0])
                        if self.sample_labels_from_gaussian:
                            labels_not_confident = torch.normal(mean=torch.ones((len(masks_not_conf)))*0.75, std=torch.ones((len(masks_not_conf)))*0.02)
                        else:
                            labels_not_confident = torch.ones((len(masks_not_conf)))*0.75 # 0.75 for not confident
                        masks = torch.concat((masks,masks_not_conf))
                        labels = torch.concat((labels,labels_not_confident))
                        boxes = torch.concat((boxes,boxes_not_conf))
            labels = torch.clamp(labels,min=0.6,max=1.0)
            targets.append({
                'boxes': boxes.cuda(),
                'labels': labels.cuda().to(torch.int64),
                'masks': masks.cuda(),
            })
        preds = self.forward(x,targets)
        loss = sum(loss for loss in preds.values())
        for k, v in preds.items():
            self.log(f"train/{k}", v, prog_bar=True, on_epoch=True,on_step=False)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True,on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y_true, y_true_not_confident   = batch
            y_true_not_confident_filtered = torch.where(y_true==0,torch.unsqueeze(y_true_not_confident,3).cuda(),torch.zeros_like(y_true))
        elif len(batch) == 2:
            x, y_true = batch
        else:
            raise Exception('Wrong dimension for training batch')

        targets = []
        for k in range(len(y_true)):
            y_true_k = torch.stack([x for x in y_true[k] if torch.sum(x) > 0])
            if torch.sum(y_true_k) == 0 or len([(y_true_k==i)*1 for i in range(1,int(torch.max(y_true_k)+1)) if torch.sum((y_true_k==i)*1) > MIN_MASK_SIZE_THESHOLD]) == 0:
                # negative sample
                masks = torch.zeros(0, 100, 100, dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)
            else:
                #masks = torch.stack([(y_true_k==i)*1 for i in range(1,int(torch.max(y_true_k)+1)) if torch.sum((y_true_k==i)*1) > MIN_MASK_SIZE_THESHOLD])
                masks = y_true_k
                boxes =  masks_to_boxes(masks)
                labels = torch.ones((len(masks)))
                
                if self.add_not_confident_labels:
                    if torch.sum(y_true_not_confident_filtered[k]) > 0 and len([(y_true_not_confident_filtered[k]==i)*1 for i in range(1,int(torch.max(y_true_not_confident_filtered[k])+1)) if torch.sum((y_true_not_confident_filtered[k]==i)*1) > MIN_MASK_SIZE_THESHOLD]) > 0:
                        masks_not_conf = torch.stack([(y_true_not_confident_filtered[k]==i)*1 for i in range(1,int(torch.max(y_true_not_confident_filtered[k])+1)) if torch.sum((y_true_not_confident_filtered[k]==i)*1) > MIN_MASK_SIZE_THESHOLD])
                        boxes_not_conf =  masks_to_boxes(masks_not_conf[:,:,:,0])
                        labels_not_confident = torch.ones((len(masks_not_conf)))*0.75 # 0.75 for not confident
                        masks = torch.concat((masks,masks_not_conf))
                        labels = torch.concat((labels,labels_not_confident))
                        boxes = torch.concat((boxes,boxes_not_conf))
                        
            labels = torch.clamp(labels,max=1.0)
            targets.append({
                'boxes': boxes.cuda(),
                'labels': labels.cuda().to(torch.int64),
                'masks': masks.cuda(),
            })
        self.train()
        preds = self.forward(x,targets)
        self.eval()
        loss = sum(loss for loss in preds.values())
        for k, v in preds.items():
            self.log(f"val/{k}", v, prog_bar=True, on_epoch=True,on_step=False)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True,on_step=False)
        
        if batch_idx == 0 and self.current_epoch > 20 and self.current_epoch%2==0:
            # log bboxes
            preds = self.forward(x,targets)
            boxes = [wandb.Image(draw_bounding_boxes((x[i]*255).to(torch.uint8),preds[i]['boxes'][preds[i]['scores'] > THRESHOLD_CONF], colors='red').permute(1,2,0).numpy()) for i in range(len(x))]
            wandb.log({"bounding boxes": boxes,})
            # log masks
            for i in range(len(targets)):
                if len(targets[i]['masks']) > 0:
                    H,W = targets[i]['masks'][0].shape
                    break
            pred_masks_conf = torch.stack([post_process(pred,THRESHOLD_CONF,1.0,H,W) for pred in preds]).unsqueeze(3)
            masks = [wandb.Image((pred_masks_conf[i]*(256/(torch.max(pred_masks_conf[i])+0.001))).to(torch.uint8).numpy()) for i in range(len(pred_masks_conf))]
            wandb.log({"masks": masks,})
        
        return loss
        # for k, v in preds.items():
        #     self.log(f"val/{k}", v, prog_bar=True, on_epoch=True,on_step=False)
        # for i in range(len(targets)):
        #     if len(targets[i]['masks']) > 0:
        #         H,W = targets[i]['masks'][0].shape
        #         break
        # pred_masks_conf = torch.stack([post_process(pred,THRESHOLD_CONF,1.0,H,W) for pred in preds]).unsqueeze(3)
        # y_true_masks = [torch.stack([x for x in y_true[i] if torch.sum(x) > 0]) for i in range(len(y_true))]
        # y_true_masks = [torch.stack(sorted(y_true_masks[i],key=lambda x: torch.sum(x))) for i in range(len(y_true_masks))]
        # y_true_masks_summed = torch.zeros_like(pred_masks_conf)
        # for i in range(len(y_true_masks)):
        #     for j in range(len(y_true_masks[i])):
        #         y_true_masks_summed[i][y_true_masks[i][j].unsqueeze(2) > 0] = j+1
        # ajis_conf = torch.tensor([get_fast_aji_plus(remap_label(y_true_masks_summed[i].detach().cpu().numpy()),remap_label(pred_masks_conf[i].detach().cpu().numpy())) for i in range(len(y_true))])
        # ajis_conf_mean = torch.mean(ajis_conf)
                
        if self.add_not_confident_labels:
            pred_masks_not_conf = torch.stack([post_process(pred,THRESHOLD_NOT_CONF,THRESHOLD_CONF) for pred in preds]).unsqueeze(3)
            ajis_not_conf = torch.tensor([get_fast_aji_plus(remap_label(y_true_not_confident_filtered[i].detach().cpu().numpy()),remap_label(pred_masks_not_conf[i].detach().cpu().numpy())) for i in range(len(y_true_not_confident_filtered))])
            ajis_not_conf_mean = torch.mean(ajis_not_conf)
            
        if self.add_not_confident_labels:
            self.log("val/aji_conf", ajis_conf_mean, prog_bar=True, on_epoch=True,on_step=False)
            self.log("val/aji_not_conf", ajis_not_conf_mean, prog_bar=True, on_epoch=True,on_step=False)
            self.log("val/aji", ajis_conf_mean+ajis_not_conf_mean, prog_bar=True, on_epoch=True,on_step=False)
            self.log("val/loss", (1-ajis_conf)+(1-ajis_not_conf), prog_bar=True, on_epoch=True,on_step=False)
            return (1-ajis_conf)+(1-ajis_not_conf)
        else:
            self.log("val/aji", ajis_conf, prog_bar=True, on_epoch=True,on_step=False)
            self.log("val/loss", 1-ajis_conf, prog_bar=True, on_epoch=True,on_step=False)
            return 1-ajis_conf


    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y_true, y_true_not_confident   = batch
            y_true_not_confident_filtered = torch.where(y_true==0,torch.unsqueeze(y_true_not_confident,3).cuda(),torch.zeros_like(y_true))
        elif len(batch) == 2:
            x, y_true = batch
        else:
            raise Exception('Wrong dimension for training batch')
        targets = []
        for k in range(len(y_true)):
            y_true_k = torch.stack([x for x in y_true[k] if torch.sum(x) > 0])
            if torch.sum(y_true_k) == 0 or len([(y_true_k==i)*1 for i in range(1,int(torch.max(y_true_k)+1)) if torch.sum((y_true_k==i)*1) > MIN_MASK_SIZE_THESHOLD]) == 0:
                # negative sample
                masks = torch.zeros(0, 100, 100, dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)
            else:
                #masks = torch.stack([(y_true_k==i)*1 for i in range(1,int(torch.max(y_true_k)+1)) if torch.sum((y_true_k==i)*1) > MIN_MASK_SIZE_THESHOLD])
                masks = y_true_k
                boxes =  masks_to_boxes(masks)
                labels = torch.ones((len(masks)))
                
                if self.add_not_confident_labels:
                    if torch.sum(y_true_not_confident_filtered[k]) > 0 and len([(y_true_not_confident_filtered[k]==i)*1 for i in range(1,int(torch.max(y_true_not_confident_filtered[k])+1)) if torch.sum((y_true_not_confident_filtered[k]==i)*1) > MIN_MASK_SIZE_THESHOLD]) > 0:
                        masks_not_conf = torch.stack([(y_true_not_confident_filtered[k]==i)*1 for i in range(1,int(torch.max(y_true_not_confident_filtered[k])+1)) if torch.sum((y_true_not_confident_filtered[k]==i)*1) > MIN_MASK_SIZE_THESHOLD])
                        boxes_not_conf =  masks_to_boxes(masks_not_conf[:,:,:,0])
                        labels_not_confident = torch.ones((len(masks_not_conf)))*0.75 # 0.75 for not confident
                        masks = torch.concat((masks,masks_not_conf))
                        labels = torch.concat((labels,labels_not_confident))
                        boxes = torch.concat((boxes,boxes_not_conf))
            labels = torch.clamp(labels,max=1.0)
            targets.append({
                'boxes': boxes.cuda(),
                'labels': labels.cuda().to(torch.int64),
                'masks': masks.cuda(),
            })
        preds = self.forward(x,targets)
        for i in range(len(targets)):
            if len(targets[i]['masks']) > 0:
                H,W = targets[i]['masks'][0].shape
                break
        pred_masks_conf = torch.stack([post_process(pred,THRESHOLD_CONF,1.0,H,W) for pred in preds]).unsqueeze(3)
        y_true_masks = [torch.stack([x for x in y_true[i] if torch.sum(x) > 0]) for i in range(len(y_true))] # 16,256,512,512
        y_true_masks = [torch.stack(sorted(y_true_masks[i],key=lambda x: torch.sum(x))) for i in range(len(y_true_masks))] # [0] 4,512,512
        y_true_masks_summed = torch.zeros_like(pred_masks_conf) # pred mask conf = 16,512,512,1
        for i in range(len(y_true_masks)):
            for j in range(len(y_true_masks[i])):
                y_true_masks_summed[i][y_true_masks[i][j].unsqueeze(2) > 0] = j+1
        ajis_conf = torch.tensor([get_fast_aji_plus(remap_label(y_true_masks_summed[i].detach().cpu().numpy()),remap_label(pred_masks_conf[i].detach().cpu().numpy())) for i in range(len(y_true))])
        ajis_conf_mean = torch.mean(ajis_conf)
        
        if self.add_not_confident_labels:
            pred_masks_not_conf = torch.stack([post_process(pred,THRESHOLD_NOT_CONF,THRESHOLD_CONF) for pred in preds]).unsqueeze(3)
            ajis_not_conf = torch.tensor([get_fast_aji_plus(remap_label(y_true_not_confident_filtered[i].detach().cpu().numpy()),remap_label(pred_masks_not_conf[i].detach().cpu().numpy())) for i in range(len(y_true_not_confident_filtered))])
            ajis_not_conf_mean = torch.mean(ajis_not_conf)
            
        if self.add_not_confident_labels:
            self.log("test/aji_conf", ajis_conf_mean, prog_bar=True, on_epoch=True,on_step=False)
            self.log("test/aji_not_conf", ajis_not_conf_mean, prog_bar=True, on_epoch=True,on_step=False)
            self.log("test/aji", ajis_conf_mean+ajis_not_conf_mean, prog_bar=True, on_epoch=True,on_step=False)
            self.log("test/loss", (1-ajis_conf)+(1-ajis_not_conf), prog_bar=True, on_epoch=True,on_step=False)
            return (1-ajis_conf)+(1-ajis_not_conf)
        else:
            self.log("test/aji", ajis_conf, prog_bar=True, on_epoch=True,on_step=False)
            self.log("test/loss", 1-ajis_conf, prog_bar=True, on_epoch=True,on_step=False)
            return 1-ajis_conf


    def log_imgs(self,x,y,y_true,max_items=2):
        
        y = [y_item.cpu().detach().numpy() for y_item in y]
        y_true = [y_item.cpu().detach().numpy() for y_item in y_true]
        y = [y_item*(255/np.max(y_item)).astype(np.uint8) for y_item in y]
        y_true = [y_item*(255/np.max(y_item)).astype(np.uint8) for y_item in y_true]


        x_wandb = [wandb.Image(x_item.permute(1,2,0).cpu().detach().numpy()) for x_item in x]
        y_wandb = [wandb.Image(y_item) for y_item in y]
        y_true_wandb = [wandb.Image(y_item) for y_item in y_true]
        wandb.log({
            "x": x_wandb[:max_items],
            "y": y_wandb[:max_items],
            "y_true": y_true_wandb[:max_items]
        })