import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import ipdb
st = ipdb.set_trace
import numpy as np
import torch
from dataclasses import dataclass
from arguments import args
import yaml
from transformers import ViTFeatureExtractor
from SOLQ.models.position_encoding import PositionalEncoding3D, PositionEmbeddingSine


class VisualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Open yaml config file
        with open(config, 'r') as ymlfile:
            self.config = yaml.load(ymlfile)
        
        self.init_backbone()    
        self.init_positional_encoding()
        
        
    def init_backbone(self, backbone: str = "resnet", pretrained=False):
        backbone = self.config['backbone']
        model = backbone['model']
        pretrained = backbone['pretrained']
        
        # Set Visual Backbone
        if model == "resnet":
            self.visual_backbone = models.resnet50(pretrained=pretrained)
            self.visual_backbone = nn.Sequential(*list(self.visual_backbone.children())[:-2])
        elif model == "resnet101":
            self.visual_backbone = models.resnet101(pretrained=pretrained)
            self.visual_backbone = nn.Sequential(*list(self.visual_backbone.children())[:-2])
        elif model == 'ViT':
            # default checkpoint: 'google/vit-base-patch16-224'
            if 'checkpoint' in backbone:
                checkpoint = backbone['checkpoint']
            else:
                checkpoint = 'google/vit-base-patch16-224'
            self.visual_backbone = ViTFeatureExtractor.from_pretrained(checkpoint)          
            
            
    def init_positional_encoding(self, position_embedding: str = "sine"):
        position_embedding = self.config['position_embedding']
        if position_embedding == "sine":
            self.positional_encoding = PositionEmbeddingSine()
        elif position_embedding == "learned":
            self.positional_encoding = PositionalEncoding3D()
        else:
            raise ValueError(f"not supported {position_embedding}")  