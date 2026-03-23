import os
import torch
import torch.nn.functional as F
import tqdm
from typing import *
from .base_trainer import BaseTrainer
from utils.meter import AverageMeter
from utils.tools import compute_cluster_centroids
from losses.cm import ClusterMemoryAMP
from losses.ce_loss import CrossEntropyLabelSmooth

class BaselineTrainer(BaseTrainer):
    """Trainer for baseline model."""
    def __init__(self, cfg, logger, device, model, optimizer, scheduler, evaluator):
        super().__init__(cfg, logger, device, model, optimizer, scheduler, evaluator)
    
    def train(self, train_loader, train_loader_normal, val_loader):

        # evaluation metrics
        meters = {k: AverageMeter() for k in ["total", "ce", "pcl", "ce_acc"]}
        
        # grad scaler
        scaler = torch.amp.GradScaler()

        # loss functions
        xent = CrossEntropyLabelSmooth(self.model.num_classes)

        # training loop
        for epoch in range(1, self.cfg.SOLVER.MAX_EPOCHS + 1):
            # reset meters
            for v in meters.values():
                v.reset()

            # schedule learning rate
            self.scheduler.step(epoch)
            self.logger.info("Learning rate is changed to {:.2e}".format(self.scheduler._get_lr(epoch)[0]))

            # prepare memories
            reid_features = []
            labels = []
            self.model.eval()
            with torch.no_grad():
                for _, (path, image, pid, _, _) in enumerate(tqdm.tqdm(train_loader_normal, desc="Extracting reid features")):
                    image = image.to(self.device)
                    with torch.amp.autocast(self.device):
                        f_reid = self.model.infer_image(image, after_bn=True)
                    reid_features.append(f_reid)
                    labels.append(pid)
            reid_features = torch.cat(reid_features, dim=0)
            labels = torch.cat(labels, dim=0)
            memory = ClusterMemoryAMP(temp=self.cfg.PCL.MEMORY_TEMP,
                                      momentum=self.cfg.PCL.MEMORY_MOMENTUM,
                                      use_hard=self.cfg.PCL.HARD_MEMORY_UPDATE).to(self.device)
            memory.features = compute_cluster_centroids(F.normalize(reid_features.float(), dim=1), labels, l2_norm=True).to(self.device)
            self.logger.info(f"PCL memory shape: {memory.features.shape}")

            # forwarding
            self.model.train()
            tqdm_loader = tqdm.tqdm(train_loader, total=len(train_loader))
            for n_iter, (path, image, target, _, _) in enumerate(tqdm_loader):
                self.optimizer.zero_grad()
                image, target = image.to(self.device), target.to(self.device)
                loss_dict = {}
                with torch.amp.autocast(self.device):
                    # ReID forwarding
                    logit, f_reid = self.model(image)

                    # loss computation
                    loss_dict["ce"] = xent(logit, target) * self.cfg.MODEL.ID_LOSS_WEIGHT
                    loss_dict["pcl"] = memory(F.normalize(f_reid, dim=1), target) * self.cfg.MODEL.PCL_LOSS_WEIGHT
                    loss = sum(loss_dict.values())
                    loss_dict["total"] = loss
                
                # backwarding
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # update meters
                for k in meters.keys():
                    if k == "ce_acc":
                        acc = (logit.max(1)[1] == target).float().mean()
                        meters[k].update(acc)
                    else:
                        meters[k].update(loss_dict[k].item(), image.shape[0])
                torch.cuda.synchronize()
                desc = "Epoch [{}/{}]: ReID loss={:.4e}, ce acc={:.1%}".format(
                    epoch,
                    self.cfg.SOLVER.MAX_EPOCHS,
                    meters["ce"].avg+meters["pcl"].avg,
                    meters["ce_acc"].avg
                )
                tqdm_loader.set_description(desc=desc)
            self.logger.info(desc)

            # evaluation
            if epoch % self.cfg.SOLVER.EVAL_PERIOD == 0:
                self.eval(epoch, val_loader)

    def eval(self, epoch, val_loader):
        self.evaluator.reset()
        torch.save(self.model.state_dict(), 
                   os.path.join(self.cfg.OUTPUT_DIR, "reid_encoder@epoch_{}.pth".format(epoch)))
        self.logger.info(f"ReID encoder checkpoint is saved.")
        self.model.eval()
        for n_iter, (_, img, vid, camid, _) in enumerate(tqdm.tqdm(val_loader, desc="Inference")):
            with torch.no_grad():
                with torch.amp.autocast(self.device):
                    img = img.to(self.device)
                    feat_bn = self.model.infer_image(img, after_bn=True)
                self.evaluator.update((feat_bn, vid, camid))
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        self.logger.info("Validation Results - Epoch: {}".format(epoch))
        self.logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        torch.cuda.empty_cache()


    def inference(self, val_loader, checkpoint_path):
        self.evaluator.reset()
        self.model.to("cpu")
        self.model.load_param_inference(checkpoint_path)
        self.model.to(self.device)

        self.model.eval()
        for n_iter, (_, img, vid, camid, _) in enumerate(tqdm.tqdm(val_loader, desc="Inference")):
            with torch.no_grad():
                with torch.amp.autocast(self.device):
                    img = img.to(self.device)
                    feat_bn = self.model.infer_image(img, after_bn=True)
            self.evaluator.update((feat_bn, vid, camid))
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        self.logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        torch.cuda.empty_cache()