import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_clip_to_cpu, weights_init_classifier

logger = logging.getLogger("logger")

class CLIP_ImageEncoder(nn.Module):
    def __init__(self, cfg, num_classes, num_cams, num_views) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_cams = num_cams
        self.num_views = num_views
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-cfg.MODEL.STRIDE_SIZE[0])//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-cfg.MODEL.STRIDE_SIZE[1])//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.num_patches = self.h_resolution * self.w_resolution
        
        clip_model = load_clip_to_cpu(cfg.MODEL.NAME, self.h_resolution, self.w_resolution, cfg.MODEL.STRIDE_SIZE[0]).cuda()
        
        self.image_encoder = clip_model.visual
        self.output_dim = self.image_encoder.output_dim
        
        # classification head
        self.fc = nn.Linear(self.image_encoder.output_dim, num_classes, bias=False)
        self.fc.apply(weights_init_classifier)
        
        # bnneck
        self.bnneck = nn.BatchNorm1d(self.image_encoder.output_dim)
        self.bnneck.apply(self._init_bnneck)
        
        self.enable_cam_emb = cfg.MODEL.ENABLE_CAM_EMB
        if self.enable_cam_emb:
            self.cam_emb = nn.Parameter(torch.empty(num_cams, 768))
            nn.init.trunc_normal_(self.cam_emb.data, std=0.02)
            logger.info(f"[ReID] Enable camera embedding with shape {self.cam_emb.shape}.")
        
        
        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        if cfg.MODEL.FREEZE_PATCH_PROJ:
            for _, v in self.image_encoder.conv1.named_parameters():
                v.requires_grad_(False)
            logger.info(f"[ReID] Freeze patch projection layer with shape {self.image_encoder.conv1.weight.shape}.")
        else:
            logger.info("[ReID] Do not freeze patch projection layer.")
        
    def forward(self, img=None, cam_label=None, view_label=None, return_patch_embeds=False):
        """
        img: [B, C, H, W]
        """
        if cam_label is not None:
            cv_emb = self.cam_emb[cam_label]
        else:
            cv_emb = None
        _, _, xproj = self.image_encoder(img, cv_emb) # [B, L, D]
        B = xproj.shape[0]
        reid_embeds = xproj[:,0,:] # [B, D]
        patch_embeds = xproj[:, 1:, :] # [B, L-1, D]
        
        bn_reid_embeds = self.bnneck(reid_embeds)
        logit = self.fc(bn_reid_embeds)
        
        if return_patch_embeds:
            return logit, bn_reid_embeds, patch_embeds
        else:
            return logit, bn_reid_embeds

    @torch.no_grad()
    def infer_image(self, img, cam_label=None, view_label=None, after_bn=True):
        """
        img: [B, C, H, W]
        """
        _, _, xproj = self.image_encoder(img, cam_label) # [B, L, D]
        B = xproj.shape[0]
        reid_embeds = xproj[:,0,:] # [B, D]
        
        bn_reid_embeds = self.bnneck(reid_embeds)
        
        if after_bn:
            return bn_reid_embeds
        else:
            return reid_embeds
        
    @torch.no_grad()
    def infer_image_all(self, img, cam_label=None, view_label=None, after_bn=True):
        """
        Return all features including patch features.
        img: [B, C, H, W]
        """
        _, _, xproj = self.image_encoder(img, cam_label) # [B, L, D]
        B = xproj.shape[0]
        reid_embeds = xproj[:,0,:] # [B, D]
        patch_embeds = xproj[:,1:,:] # [B, L-1, D]
        
        bn_reid_embeds = self.bnneck(reid_embeds)
        
        if after_bn:
            return bn_reid_embeds, patch_embeds
        else:
            return reid_embeds, patch_embeds
        
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location="cpu")
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        logger.info(f"Loading pretrained model from {trained_path}")
    
    def load_param_inference(self, trained_path):
        cnt = 0
        param_dict = torch.load(trained_path, map_location="cpu")
        for i in param_dict:
            if "text_encoder" in i or "prompt_learner" in i or "fc" in i:
                if "image_encoder" not in i:
                    cnt += 1
                    continue # ignore num_class related layers
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        logger.info(f"Loading pretrained model from {trained_path}, ignore {cnt} layers.")
    
    def _init_bnneck(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
            m.bias.requires_grad_(False)