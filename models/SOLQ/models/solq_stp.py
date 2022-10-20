# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from SOLQ.util import box_ops
from SOLQ.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .position_encoding import PositionalEncoding3D, PositionEmbeddingLearned3D
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import numpy as np
import cv2
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer_stp import build_deforamble_transformer
# from .deformable_transformer_cp import build_deforamble_transformer as build_cp_deforamble_transformer
from .dct import ProcessorDCT
from detectron2.structures import BitMasks
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom
import copy
import functools
import time
# from torchvision.ops import RoIPool
from torchvision.ops import roi_pool, roi_align
import utils.geom

import ipdb
st = ipdb.set_trace
from arguments import args
print = functools.partial(print, flush=True)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check_zeros(tensor):
    nonzero = ~torch.all(tensor.reshape(tensor.shape[0], -1)==0, dim=1)
    return nonzero

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SOLQ(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_actions, aux_loss=True, with_box_refine=False, two_stage=False, with_vector=False, 
                 processor_dct=None, vector_hidden_dim=256, actions2idx=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.transformer.num_actions = num_actions
        self.class_embed = nn.Linear(hidden_dim, num_classes)

        # policy specific params
        self.add_action_history = args.add_action_history
        self.max_objs = args.max_objs
        self.do_deformable_atn_decoder = args.do_deformable_atn_decoder
        self.only_intermediate_objs_memory = args.only_intermediate_objs_memory
        self.within_demo = args.within_demo
        self.roi_pool = args.roi_pool
        self.roi_align = args.roi_align
        self.query_per_action = args.query_per_action
        self.stop_gradient_backbone_history = args.stop_gradient_backbone_history
        self.retrieve_knn_demos_subgoal = args.retrieve_knn_demos_subgoal
        self.use_action_ghost_nodes = args.use_action_ghost_nodes
        self.cos_frame_embedding = args.cos_frame_embedding
        self.use_3d_obj_centroids = args.use_3d_obj_centroids
        self.use_object_tracklets = args.use_object_tracklets
        self.use_3d_img_pos_encodings = args.use_3d_img_pos_encodings
        self.add_whole_image_mask = args.add_whole_image_mask
        self.do_decoder_2d3d = args.do_decoder_2d3d
        self.learned_3d_pos_enc = args.learned_3d_pos_enc
        self.keep_one_object_instance = args.keep_one_object_instance
        if self.within_demo:
            self.topk = 1
        else:
            self.topk = args.topk
        self.use_3d_pos_enc = args.use_3d_pos_enc
        self.max_distance_grid = args.max_distance_grid
        if self.use_3d_pos_enc and self.add_action_history:
            assert(False) # probably should not have both?

        if self.use_3d_img_pos_encodings:
            self.fov = args.fov
            self.W = args.W
            self.H = args.H
            hfov = float(self.fov) * np.pi / 180.
            self.pix_T_camX = np.array([
                [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
                [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
                [0., 0.,  1, 0],
                [0., 0., 0, 1]])
            self.pix_T_camX[0,2] = self.W/2.
            self.pix_T_camX[1,2] = self.H/2.
            self.pix_T_camX = torch.from_numpy(self.pix_T_camX).to(device).unsqueeze(0).float()

        
        # extra losses
        self.intermediate_embed = nn.Linear(hidden_dim, 2)
        if self.query_per_action:
            self.action_embed = nn.Linear(hidden_dim, 1)
        else:
            self.action_embed = nn.Linear(hidden_dim, num_actions)

        # action token
        if self.add_action_history:
            self.action_token = nn.Embedding(num_actions, hidden_dim)

        # tag for intermediate objects in the demo
        if self.within_demo or self.retrieve_knn_demos_subgoal: 
            if not self.only_intermediate_objs_memory:
                self.interm_embed = nn.Parameter(torch.Tensor(2, hidden_dim))
                normal_(self.interm_embed)
            if self.retrieve_knn_demos_subgoal: 
                self.knn_embed = nn.Parameter(torch.Tensor(self.topk, hidden_dim))
                normal_(self.knn_embed)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.processor_dct.n_keep, 3)
        self.num_feature_levels = num_feature_levels

        # projection for multiscale object queries 
        self.multiscale_query_proj = nn.Linear(hidden_dim*num_feature_levels, hidden_dim)

        if self.use_3d_pos_enc:
            # self.max_distance_grid # max distance from current position to consider
            self.step_size = args.STEP_SIZE
            self.dt = args.DT
            self.horizon_dt = args.HORIZON_DT
            self.pitch_range = args.pitch_range
            if self.learned_3d_pos_enc:
                # 5 dimensions - x,y,z,yaw,pitch
                self.pos_enc_3d = PositionEmbeddingLearned3D(5, hidden_dim)
            else:
                num_bins_spatial = int(self.max_distance_grid/self.step_size)*2 # parameterize 2d floor position with bin
                num_yaw = 360//self.dt
                num_pitch = 360//self.horizon_dt
                num_bins_rotation = int(num_yaw*2*num_pitch*2)
                pos_enc_3d = PositionalEncoding3D(hidden_dim)
                z = torch.zeros((1,num_bins_spatial,num_bins_spatial,num_bins_rotation,hidden_dim))
                self.pos_enc_3d = pos_enc_3d(z).squeeze(0)#.to(device)
                self.rot2dto1d_indices = torch.arange(num_bins_rotation).reshape(num_yaw*2,num_pitch*2).to(device)
        
        if not self.use_3d_pos_enc or self.do_decoder_2d3d:
            if self.cos_frame_embedding: 
                self.frame_embed = positionalencoding1d(args.max_steps_per_subgoal+2, hidden_dim).transpose(1,0).to(device)
            else:
                # embedding for history time step + current time step
                if args.old_frame_embed:
                    add_ = 1
                else:
                    add_ = 2
                self.frame_embed = nn.Parameter(torch.Tensor(args.max_steps_per_subgoal+add_, hidden_dim))
                normal_(self.frame_embed)            

        # # learned embeddings for target/supporters
        # if args.pred_one_object and args.use_supporters:
        #     self.target_emb = nn.Embedding(50, num_pos_feats)

        if not two_stage:
            num_queries += 1 # add an extra query for action prediction
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        else:
            if self.use_action_ghost_nodes:
                if not self.use_3d_pos_enc:
                    assert(False) # not using 3d pos encodings here does not make sense
                # st()
                # list(actions2idx.keys())
                # torch.zeros().to(torch.bool)
                self.num_movement_actions = sum(("Move" in k or "Rotate" in k or "Look" in k) for k in list(actions2idx.keys())) # all movement embeddings get the same node with different positional encodings
                # positions_actions = {
                #     'MoveAhead': []
                # }
                print("Using action ghost nodes")
                self.query_embed = nn.Embedding(num_actions - self.num_movement_actions + 1, hidden_dim*2) # action prediction
                if self.learned_3d_pos_enc:
                    self.action_distance_mapping = {
                        "MoveAhead": torch.tensor([0., 0., self.step_size, 0., 0.]), 
                        "RotateRight": torch.tensor([0., 0., 0., self.dt, 0.]),
                        "RotateLeft": torch.tensor([0., 0., 0., -self.dt, 0.]),
                        "LookDown": torch.tensor([0., 0., 0., 0., self.horizon_dt]), # to replace in forward()
                        "LookUp": torch.tensor([0., 0., 0., 0., -self.horizon_dt]), # to replace in forward()
                        "PickupObject": torch.tensor([0., 0., 0., 0., 0.]),
                        "PutObject": torch.tensor([0., 0., 0., 0., 0.]),
                        "OpenObject": torch.tensor([0., 0., 0., 0., 0.]),
                        "CloseObject": torch.tensor([0., 0., 0., 0., 0.]),
                        "SliceObject": torch.tensor([0., 0., 0., 0., 0.]),
                        "ToggleObjectOn": torch.tensor([0., 0., 0., 0., 0.]),
                        "ToggleObjectOff": torch.tensor([0., 0., 0., 0., 0.]),
                        "Done": torch.tensor([0., 0., 0., 0., 0.]),
                        }
                else:
                    self.action_distance_mapping = {
                        "MoveAhead": torch.tensor([0., self.step_size, 0., 0.]), 
                        "RotateRight": torch.tensor([0., 0., self.dt, 0.]),
                        "RotateLeft": torch.tensor([0., 0., -self.dt, 0.]),
                        "LookDown": torch.tensor([0., 0., 0., self.horizon_dt]), # to replace in forward()
                        "LookUp": torch.tensor([0., 0., 0., -self.horizon_dt]), # to replace in forward()
                        "PickupObject": torch.tensor([0., 0., 0., 0.]),
                        "PutObject": torch.tensor([0., 0., 0., 0.]),
                        "OpenObject": torch.tensor([0., 0., 0., 0.]),
                        "CloseObject": torch.tensor([0., 0., 0., 0.]),
                        "SliceObject": torch.tensor([0., 0., 0., 0.]),
                        "ToggleObjectOn": torch.tensor([0., 0., 0., 0.]),
                        "ToggleObjectOff": torch.tensor([0., 0., 0., 0.]),
                        "Done": torch.tensor([0., 0., 0., 0.]),
                        }
                for k in self.action_distance_mapping.keys():
                    if self.action_distance_mapping[k] is not None:
                        self.action_distance_mapping[k] = self.action_distance_mapping[k].to(device)
            elif self.query_per_action:
                print("Using query per action")
                self.query_embed = nn.Embedding(num_actions, hidden_dim*2) # action prediction
            else:
                print("Using single action query")
                self.query_embed = nn.Embedding(1, hidden_dim*2) # action prediction
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.intermediate_embed = _get_clones(self.intermediate_embed, num_pred)
            self.action_embed = _get_clones(self.action_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.intermediate_embed = nn.ModuleList([self.intermediate_embed for _ in range(num_pred)])
            self.action_embed = nn.ModuleList([self.action_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if args.do_self_atn_for_queries:
            self.transformer.multiscale_query_proj = self.multiscale_query_proj

    def forward(
        self, 
        samples, 
        masks, 
        text, 
        action_history=None, 
        memory=None,
        positions=None,
        obj_centroids=None,
        depth=None,
        obj_instance_ids=None,
        ):

        """ The forward expects a NestedTensor, which consists of:
               - samples: batched images, of shape [batch_size x nviews x 3 x H x W]
               - instance_masks: object masks in list batch, nviews, HxW

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        self.device = samples.device

        # if self.use_action_ghost_nodes:
        #     # get relative for move ahead (defined to be last position in our dataloader)
        #     positions_moveahead = positions[:,-1]
        #     positions = positions[:,:-1]

        history_frame_inds = np.arange(samples.shape[1])
        frame_inds = history_frame_inds.copy()

        if memory is not None: 
            '''
            For retrieved demos to modulate network 
            '''
            B, SH, C, H, W = samples.shape
            num_mem_frames = memory['images'][0].shape[1]
            mem_images = torch.cat(list(memory['images'].values()), dim=1)
            mem_interms = torch.cat(list(memory['interms'].values()), dim=1)
            mem_masks = torch.cat(list(memory['masks'].values()), dim=1)
            SM = mem_images.shape[1]
            mem_inds = torch.zeros(SH+SM).to(self.device).to(torch.bool)
            mem_inds[SH:SH+SM] = True
            samples = torch.cat([samples, mem_images], dim=1)
            masks = torch.cat([masks, mem_masks], dim=1)
            mem_frame_inds = np.expand_dims(np.arange(num_mem_frames), 0).repeat(self.topk,0).flatten() # frame numbers for each demo
            frame_inds = np.concatenate([frame_inds, mem_frame_inds], axis=0)
        
        B, S, C, H, W = samples.shape
        M = masks.shape[2]

        if self.stop_gradient_backbone_history:
            '''
            Do not compute gradients for history encodings in backbone
            '''
            features, pos = self.apply_backbone_sg(samples)
            num_feature_levs = len(features)
            samples = samples.contiguous().view(B*S, C, H, W).unbind(0)
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            
        else:
            # time1 = time.time()
            samples = samples.contiguous().view(B*S, C, H, W).unbind(0)
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples) 
            num_feature_levs = len(features)
            # time2 = time.time()
            # print("time backbone:", time2-time1)
        
        # time1 = time.time()
        srcs_t = []
        masks_t = []
        poss_t = []
        poss_t_3d = []
        # objects to track
        obj_feats = torch.ones([B,S,self.num_feature_levels,self.max_objs,self.hidden_dim]).to(self.device).to(features[-1].tensors.dtype)
        # obj_poss = torch.ones([B,S,self.num_feature_levels,self.max_objs,self.hidden_dim]).to(device).to(features[-1].tensors.dtype)
        # if self.do_decoder_2d3d:
        #     obj_poss_3d = torch.ones([B,S,self.num_feature_levels,self.max_objs,self.hidden_dim]).to(device).to(features[-1].tensors.dtype)
        # else:
        #     obj_poss_3d = None
        # obj_masks = torch.ones([B,S,self.num_feature_levels,self.max_objs]).to(device).to(torch.bool)
        # for l, feat in enumerate(features):
        for l in range(self.num_feature_levels):

            #####%%%% Extract multiscale image features %%%####
            if l > num_feature_levs - 1:
                if l == num_feature_levs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            else:
                feat = features[l]
                pos_l = pos[l]
                src, mask = feat.decompose()
                src = self.input_proj[l](src)
            _, C_l, H_l, W_l = src.shape

            ####%%% Extract object features %%%#####
            # src_interp = F.interpolate(src, mode='bilinear', size=masks.shape[-2:])
            
            _, C_pos_l, _, _ = pos_l.shape

            src = src.reshape(B,S,C_l,H_l,W_l)
            # src_interp = src_interp.reshape(B,S,C_l,H,W)
            mask = mask.reshape(B,S,H_l,W_l)
            pos_l = pos_l.reshape(B,S,C_l,H_l,W_l)

            if self.roi_pool or self.roi_align or masks.shape[-1]==4:

                ########%%%%%%%% HISTORY %%%%%%%%############
                masks = masks.reshape(-1, M, 4)

                if l==0: 
                    '''
                    only need to compute pos enc & masks for first layer since 
                    we pool object features from all layers for history
                    '''

                    # history pos encodings
                    pos_l_objs = self.get_history_positional_encoding(
                                        B,S,M,
                                        obj_centroids=obj_centroids,
                                        positions=positions,
                                        masks=masks,
                                        W_l=W_l, H_l=H_l, C_l=C_l,
                                        pos_l=pos_l,
                                        frame_inds=frame_inds
                                    )
                    
                    # transformer mask
                    obj_masks_ = ~check_zeros(masks.reshape(-1,4)).view(B,S,M)
                    
                    if self.do_decoder_2d3d:
                        obj_poss_3d = pos_l_objs[1] # second positional enc is 3d
                        pos_l_objs = pos_l_objs[0] # first positional enc is 2d
                    else:
                        obj_poss_3d = None
                        
                    obj_poss = pos_l_objs
                    obj_masks = obj_masks_

                # ROI pool
                masks_xyxy = box_ops.box_cxcywh_to_xyxy(masks)
                if self.roi_align:
                    masks_xyxy_flat = masks_xyxy.float().reshape(B*S*M, 4)
                    box_batch_inds = torch.arange(B*S, device=masks_xyxy.device).repeat(M,1).transpose(1,0).flatten().unsqueeze(1) # first column needs to be batch index
                    masks_xyxy_flat = torch.cat([box_batch_inds, masks_xyxy_flat], dim=1)
                    output_size = (int(H_l/4), int(W_l/4))
                    feature_crop = roi_align(src.reshape(B*S,C_l,H_l,W_l), masks_xyxy_flat, output_size=output_size, spatial_scale=W_l)
                    pooled_obj_feat = F.adaptive_avg_pool2d(feature_crop, (1,1)).squeeze(-1).squeeze(-1).view(B,S,M,self.hidden_dim)
                elif self.roi_pool:
                    masks_xyxy_flat = masks_xyxy.float().reshape(B*S*M, 4)
                    box_batch_inds = torch.arange(B*S, device=masks_xyxy.device).repeat(M,1).transpose(1,0).flatten().unsqueeze(1) # first column needs to be batch index
                    masks_xyxy_flat = torch.cat([box_batch_inds, masks_xyxy_flat], dim=1)
                    output_size = (int(H_l/4), int(W_l/4))
                    feature_crop = roi_pool(src.reshape(B*S,C_l,H_l,W_l), masks_xyxy_flat, output_size=output_size, spatial_scale=W_l)
                    pooled_obj_feat = F.adaptive_avg_pool2d(feature_crop, (1,1)).squeeze(-1).squeeze(-1).view(B,S,M,self.hidden_dim)
                else:
                    assert(False)

                obj_feats[:,:,l] = pooled_obj_feat


                ########%%%%%%%% CURRENT IMAGE %%%%%%%%############
                pos_l_im = self.get_img_positional_encoding(
                                    B,S,H,W,H_l,W_l,
                                    depth=depth,
                                    pos_l=pos_l,
                                    pitch_adjustment=positions[:,-1,-1] if positions is not None else None  # adjust xyz by current pitch (since pitch is in absolute terms)
                                    )
                
                if self.do_decoder_2d3d:
                    poss_t_3d.append(pos_l_im[1]) # second positional enc is 3d
                    pos_l_im = pos_l_im[0] # first positional enc is 2d
                srcs_t.append(src[:,-1]) # last image is current image
                masks_t.append(mask[:,-1]) # last image is current image
                poss_t.append(pos_l_im)
            else:
                # Use masks NOT boxes - this is very inefficient
                srcs_t_ = torch.zeros((B, C_l, H_l, W_l)).to(self.device).to(src.dtype)
                masks_t_ = torch.zeros((B, H_l, W_l)).to(self.device).to(mask.dtype)
                poss_t_ = torch.zeros((B, C_pos_l, H_l, W_l)).to(self.device).to(pos_l.dtype)
                masks_interp = F.interpolate(masks.float().view(B*S, M, H, W), mode='bilinear', size=src.shape[-2:]).to(torch.bool)
                masks_interp = masks_interp.reshape(B,S,masks_interp.shape[1],H_l,W_l)

                ###### EXTRACT OBJECT AND FRAME FEATURES ########
                for b in range(B):
                    for s in range(S):

                        # obj_b = masks[b,s].squeeze(0)
                        obj_b = masks_interp[b,s].squeeze(0)
                        pos_l_b = pos_l[b,s].squeeze(0)
                        num_masks = int(torch.sum(check_zeros(masks[b,s].squeeze(0))).cpu().numpy())
                        src_b = src[b,s].squeeze(0)

                        obj_feat_, obj_pos_l_, obj_mask_ = self.extract_mask_features(
                                                            num_masks, 
                                                            obj_b,
                                                            masks,
                                                            src_b,
                                                            pos_l_b,
                                                            # flat_idx_batch,
                                                            b,
                                                            s,
                                                            frame_inds[s], # frame number index for frame embedding
                                                            )
                        obj_feats[b,s,l] = obj_feat_
                        obj_poss[b,s,l] = obj_pos_l_
                        obj_masks[b,s,l] = obj_mask_
                    
                    # only current time step, we do self-attention
                    # so we only save image features for time step t for transformer
                    srcs_t_[b] = src[b,-1] #[flat_idx_batch]
                    masks_t_[b] = mask[b,-1] #[flat_idx_batch]
                    poss_t_[b] = pos_l[b,-1] #[flat_idx_batch]
                srcs_t.append(srcs_t_)
                masks_t.append(masks_t_)
                poss_t.append(poss_t_)
                # assert mask is not None
        
        # NOTE: this is now handled in the level loop
        # obj_poss = obj_poss[:,:,0] # take first level to be pos encoding
        # obj_masks = obj_masks[:,:,0] # take first level masks (all are same)

        # let's concat the object features and learn a linear projection
        obj_feats = obj_feats.transpose(2,3).flatten(3,4)
        # project multiscale to correct size for query input
        obj_feats = self.multiscale_query_proj(obj_feats) # BxSxNxE

        if self.keep_one_object_instance:
            st()

        if memory is not None: 
            # parse memory indices and add actions to the features
            mem_feats = obj_feats[:,SH:SM+SH]
            mem_masks = obj_masks[:,SH:SM+SH]
            mem_poss = obj_poss[:,SH:SM+SH]

            if not self.only_intermediate_objs_memory:
                # add intermediate object tag to positional embeddings
                interm_pos = self.interm_embed[mem_interms.to(torch.long)]
                mem_poss += interm_pos

            if self.retrieve_knn_demos_subgoal:
                knn_inds = torch.arange(self.topk).to(self.device).unsqueeze(0).repeat(num_mem_frames,1).transpose(1,0).flatten()
                # add intermediate object tag to positional embeddings
                knn_pos = self.knn_embed[knn_inds].view(1, SM, 1, self.hidden_dim).expand(B, SM, M, self.hidden_dim)
                mem_poss += knn_pos

            if self.add_action_history:
                mem_actions = torch.cat(list(memory['actions'].values()), dim=1)
                # append action history tokens to object history tokens
                action_tokens, action_mask, action_pos = self.extract_action_history_tokens(
                    mem_actions,
                    mem_frame_inds
                    )
                mem_feats = torch.cat([mem_feats, action_tokens], dim=2)
                mem_masks = torch.cat([mem_masks, action_mask], dim=2)
                mem_poss = torch.cat([mem_poss, action_pos], dim=2)

            obj_feats = obj_feats[:,:SH]
            obj_masks = obj_masks[:,:SH]
            obj_poss = obj_poss[:,:SH]
        else:
            mem_feats = None
            mem_masks = None
            mem_poss = None

        if self.add_action_history:
            # append action history tokens to object history tokens
            action_tokens, action_mask, action_pos = self.extract_action_history_tokens(
                action_history, 
                history_frame_inds,
                1 # pad one because actions don't include current frame
                ) 
            obj_feats = torch.cat([obj_feats, action_tokens], dim=2)
            obj_masks = torch.cat([obj_masks, action_mask], dim=2)
            obj_poss = torch.cat([obj_poss, action_pos], dim=2)

        # print(obj_feats.shape)

        # query_embeds = None
        # if not self.two_stage:
        query_embeds = self.query_embed.weight
        if self.use_action_ghost_nodes:
            movement_pos = query_embeds[0:1].expand(self.num_movement_actions, query_embeds.shape[1]) # first is for positional encoding of the translation movements
            query_embeds = torch.cat([movement_pos, query_embeds[1:]]) #.unsqueeze(0).expand(B,len(self.action_distance_mapping), query_embeds.shape[1])
            
            # pitch is not relative so we need to adjust pitch to be absolute
            lookdown = torch.clamp(positions[:,-1,-1] + self.action_distance_mapping["LookDown"][-1], min=min(self.pitch_range), max=max(self.pitch_range))
            lookup = torch.clamp(positions[:,-1,-1] + self.action_distance_mapping["LookUp"][-1], min=min(self.pitch_range), max=max(self.pitch_range))

            if self.learned_3d_pos_enc:
                action_pos = torch.stack(list(self.action_distance_mapping.values()), dim=0).unsqueeze(0).expand(B, len(self.action_distance_mapping), 5).clone()
                action_pos[:,list(self.action_distance_mapping.keys()).index("LookDown"),-1] = lookdown
                action_pos[:,list(self.action_distance_mapping.keys()).index("LookUp"),-1] = lookup

                pos_l_objs_3d = self.pos_enc_3d(action_pos.float())
                action_pos_enc = pos_l_objs_3d.transpose(1,2)
            else:
                action_pos = torch.stack(list(self.action_distance_mapping.values()), dim=0).unsqueeze(0).expand(B, len(self.action_distance_mapping), 4).clone()
                action_pos[:,list(self.action_distance_mapping.keys()).index("LookDown"),-1] = lookdown
                action_pos[:,list(self.action_distance_mapping.keys()).index("LookUp"),-1] = lookup
                
                position_inds = (action_pos[:,:,:2] + self.max_distance_grid)/self.step_size
                rotation_ind = ((action_pos[:,:,2] + 360)/self.dt, (action_pos[:,:,3] + 360)/self.horizon_dt)
                rotation_inds = self.rot2dto1d_indices[rotation_ind[0].flatten().long(), rotation_ind[1].flatten().long()].reshape(B,len(self.action_distance_mapping))
                action_pos_enc = self.pos_enc_3d[position_inds[:,:,0].flatten().long(), position_inds[:,:,1].flatten().long(), rotation_inds.flatten()].to(device).reshape(B,len(self.action_distance_mapping),self.hidden_dim)
        else:
            action_pos_enc = None
        
        # time2 = time.time()
        # print("time prep:", time2-time1)

        # time1 = time.time()

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
                                                                                                        srcs_t, 
                                                                                                        masks_t, 
                                                                                                        poss_t, 
                                                                                                        query_embeds, 
                                                                                                        obj_srcs=obj_feats, 
                                                                                                        obj_masks=obj_masks,
                                                                                                        obj_pos_embeds=obj_poss,
                                                                                                        mem_srcs=mem_feats,
                                                                                                        mem_masks=mem_masks,
                                                                                                        mem_pos_embeds=mem_poss,
                                                                                                        text=text,
                                                                                                        B=B,
                                                                                                        S=S,
                                                                                                        action_pos_embeds=action_pos_enc,
                                                                                                        pos_embeds_3d=poss_t_3d, # img positional embed 3d (for )
                                                                                                        obj_poss_embeds_3d=obj_poss_3d, # obj/history positional embed 3d
                                                                                                        )


        if not self.do_decoder_2d3d: 
            # remove reference points for action queries
            if self.query_per_action:
                inter_references = inter_references[:,:,:-self.num_actions]
                init_reference = init_reference[:,:-self.num_actions]
            else:
                inter_references = inter_references[:,:,:-1]
                init_reference = init_reference[:,:-1]

        outputs_classes = []
        outputs_coords = []
        outputs_interms = []
        outputs_actions = []
        for lvl in range(hs.shape[0]):
            if self.do_deformable_atn_decoder:
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
            if self.query_per_action:
                hs_ = hs[lvl][:,:-self.num_actions] # last num_actions are action queries
                as_ = hs[lvl][:,-self.num_actions:] # last num_actions are action queries
            else:
                hs_ = hs[lvl][:,:-1] # last one is action query
                as_ = hs[lvl][:,-1:] # last one is action query
            outputs_class = self.class_embed[lvl](hs_)
            outputs_interm = self.intermediate_embed[lvl](hs_)
            outputs_action = self.action_embed[lvl](as_)
            tmp = self.bbox_embed[lvl](hs_)
            if self.do_deformable_atn_decoder:
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_interms.append(outputs_interm)
            outputs_actions.append(outputs_action)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_interms = torch.stack(outputs_interms)
        outputs_actions = torch.stack(outputs_actions)

        if self.query_per_action:
            outputs_actions = outputs_actions.transpose(3,2)

        #out_batches.reshape(2, 10).permute(1, 0).reshape(20)

        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)

        out = {
            'pred_logits': outputs_class[-1], 
            'pred_boxes': outputs_coord[-1],
            'pred_interms': outputs_interms[-1],
            'pred_actions': outputs_actions[-1],
            }
        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})
        if self.aux_loss:
            out_aux = {
                'pred_logits': outputs_class, 
                'pred_boxes': outputs_coord,
                'pred_interms': outputs_interms,
                'pred_actions': outputs_actions,
                }
            if self.with_vector:
                out_aux.update({'pred_vectors': outputs_vector})
            out['aux_outputs'] = self._set_aux_loss(out_aux)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {
                'pred_logits': enc_outputs_class, 
                'pred_boxes': enc_outputs_coord
                }

        # if torch.sum(torch.isnan(outputs_class))>0:
        #     st()       
        #     for m in range(len(masks)):
        #         print(torch.sum(masks[m]), m) 

        # time2 = time.time()
        # print("time trans:", time2-time1)

        return out

    def get_img_positional_encoding(
        self,
        B,S,H,W,H_l,W_l,
        depth=None,
        pos_l=None,
        pitch_adjustment=None,
    ):  
        '''
        Get 2D and/or 3D positional encodings for the image features
        pitch adjustment: pitch in degrees to correct point cloud
        '''

        if self.do_decoder_2d3d:
            pos_l_img_3d = self.extract_img_posenc_2D3D(
                                B,S,H,W,H_l,W_l,
                                depth=depth,
                                pos_l=pos_l,
                                pitch_adjustment=pitch_adjustment,
                                mode='3D'
                                ) 
            pos_l_img_2d = self.extract_img_posenc_2D3D(
                                B,S,H,W,H_l,W_l,
                                depth=depth,
                                pos_l=pos_l,
                                pitch_adjustment=pitch_adjustment,
                                mode='2D'
                                )  
            pos_l_img = (pos_l_img_2d, pos_l_img_3d)
        elif self.use_3d_img_pos_encodings:
            pos_l_img = self.extract_img_posenc_2D3D(
                                B,S,H,W,H_l,W_l,
                                depth=depth,
                                pos_l=pos_l,
                                pitch_adjustment=pitch_adjustment,
                                mode='3D'
                                ) 
        else:
            pos_l_img = self.extract_img_posenc_2D3D(
                                B,S,H,W,H_l,W_l,
                                depth=depth,
                                pos_l=pos_l,
                                pitch_adjustment=pitch_adjustment,
                                mode='2D'
                                ) 
        return pos_l_img

    def extract_img_posenc_2D3D(
        self,
        B,S,H,W,H_l,W_l,
        depth=None,
        pos_l=None,
        pitch_adjustment=None,
        mode='2D'
    ):
        '''
        Extract 2D or 3D positional encodings for the image features
        mode: '2D' or '3D' to get positional 3D from position + centroids or 2D encodings
        '''

        if mode=='3D':
            '''
            Use 3D positional encodings for the current image by unprojecting the depth map 
            and interpolating to get 3d position for each feature point
            '''
            with torch.no_grad():
                xyz = utils.geom.depth2pointcloud(depth.unsqueeze(1), self.pix_T_camX.expand(B,4,4))
                if pitch_adjustment is not None:
                    '''
                    Adjust for pitch (head tilt) so that it is relative to 0 pitch
                    '''
                    rx = torch.deg2rad(-pitch_adjustment) # pitch - positive is down, negative is up in aithor
                    ry = torch.zeros(B).to(self.device) # yaw
                    rz = torch.zeros(B).to(self.device) # roll  
                    rot = utils.geom.eul2rotm(rx, ry, rz)
                    rotm = utils.geom.eye_4x4(B).to(self.device).to(xyz.dtype)
                    rotm[:,0:3,0:3] = rot
                    xyz = utils.geom.apply_4x4(rotm, xyz)

                    # rx = np.radians(-pitch_adjustment.cpu().numpy()) #np.radians(rotation[1]) # yaw
                    # ry = np.zeros(B) #torch.deg2rad(torch.tensor([180]).to(torch.float).to(self.device)) #np.radians(rotation[1]) # yaw
                    # rz = np.zeros(B) # roll   #
                    # rot = torch.from_numpy(utils.geom.eul2rotm_py(rx, ry, rz)).to(self.device)
                    # rotm = utils.geom.eye_4x4(B).to(self.device).to(xyz.dtype)
                    # rotm[:,0:3,0:3] = rot
                    # xyz = utils.geom.apply_4x4(rotm, xyz)

                    # # visualize adjusted xyz
                    # b_plot = 0
                    # import matplotlib.pyplot as plt
                    # from mpl_toolkits.mplot3d import proj3d
                    # skip = 100   # Skip every n points
                    # fig = plt.figure(figsize=(8, 8))
                    # ax = fig.add_subplot(111, projection='3d')
                    # point_range = range(0, xyz.shape[1], skip)
                    # ax.scatter(
                    #     xyz[b_plot,point_range,0].cpu().numpy(), 
                    #     xyz[b_plot,point_range,1].cpu().numpy(), 
                    #     -xyz[b_plot,point_range,2].cpu().numpy(),
                    #     c=xyz[b_plot,point_range,2].cpu().numpy(), # height data for color
                    #     cmap='Spectral',
                    #     marker="x"
                    #     )
                    # # plt.colorbar()
                    # plt.savefig('data/images/test.png')
                    # plt.figure()
                    # plt.imshow(depth[b_plot].cpu().numpy())
                    # plt.savefig('data/images/test2.png')
                    # fig = plt.figure(figsize=(8, 8))
                    # ax = fig.add_subplot(111)
                    # ax.scatter(xyz[b_plot,:,0].cpu().numpy(), xyz[b_plot,:,1].cpu().numpy())
                    # plt.savefig('data/images/test3.png')

                xyz = xyz.reshape(B, H, W, 3).permute(0,3,1,2)
                xyz = F.interpolate(xyz, mode='bilinear', size=(H_l,W_l), align_corners=False)
                # fig = plt.figure(figsize=(8, 8))
                # ax = fig.add_subplot(111)
                # ax.scatter(xyz[b_plot,0].flatten().cpu().numpy(), xyz[b_plot,1].flatten().cpu().numpy())
                # plt.savefig('data/images/test4.png')
                if self.learned_3d_pos_enc:
                    xyz = xyz[:,[0,2,1],:,:].permute(0,2,3,1)
                    # append zeros since rotations are zero for all objs
                    pos_enc_input = torch.cat([xyz, torch.zeros(B,H_l,W_l,2).to(self.device)], dim=-1) 
                    pos_l_objs_3d = self.pos_enc_3d(pos_enc_input.flatten(1,2).float()).reshape(B,self.hidden_dim,H_l,W_l)
                    pos_l_im = pos_l_objs_3d
                else:
                    xyz = xyz[:,[0,1], :, :].permute(0,2,3,1) # throw away height info (not currently in positional encoding)
                    xyz = torch.round(xyz*(1/self.step_size))*self.step_size # round to step size of 3d pos encodings
                    rotation_ind_img = ((torch.zeros((B,H_l,W_l)) + 360)/self.dt, (torch.zeros((B,H_l,W_l)) + 360)/self.horizon_dt) # set to all zero rotation
                    rotation_inds_img = self.rot2dto1d_indices[rotation_ind_img[0].flatten().long(), rotation_ind_img[1].flatten().long()].reshape(B,H_l,W_l)
                    pos_l_objs_3d = self.pos_enc_3d[xyz[:,:,:,0].flatten().long(), xyz[:,:,:,1].flatten().long(), rotation_inds_img.flatten()].to(self.device).reshape(B,H_l,W_l,self.hidden_dim)
                    pos_l_objs_3d = pos_l_objs_3d.permute(0,3,1,2)
                    pos_l_im = pos_l_objs_3d
        elif mode=='2D':
            pos_l_im = pos_l[:,-1] # positional encoding only for current image
        else:
            assert(False) # wrong mode
        
        return pos_l_im

    def get_history_positional_encoding(
        self,
        B,
        S,
        M,
        obj_centroids=None,
        positions=None,
        masks=None,
        W_l=None,
        H_l=None,
        C_l=None,
        pos_l=None,
        frame_inds=None
    ):
        '''
        Get 2D and/or 3D positional encodings for the history features
        '''
        if self.do_decoder_2d3d:
            # 3d history positional encoding
            pos_l_objs_3d = self.extract_history_posenc_2D3D(
                                    B,S,M,
                                    obj_centroids=obj_centroids,
                                    positions=positions,
                                    masks=masks,
                                    W_l=W_l,H_l=H_l,C_l=C_l,
                                    pos_l=pos_l,
                                    frame_inds=frame_inds,
                                    mode='3D'
                                )
            # 2d history positional encoding
            pos_l_objs_2d = self.extract_history_posenc_2D3D(
                                    B,S,M,
                                    obj_centroids=obj_centroids,
                                    positions=positions,
                                    masks=masks,
                                    W_l=W_l,H_l=H_l,C_l=C_l,
                                    pos_l=pos_l,
                                    frame_inds=frame_inds,
                                    mode='2D'
                                )
            pos_l_objs = (pos_l_objs_2d, pos_l_objs_3d)

        elif self.use_3d_pos_enc:
            # 3d history positional encoding
            pos_l_objs = self.extract_history_posenc_2D3D(
                                    B,S,M,
                                    obj_centroids=obj_centroids,
                                    positions=positions,
                                    masks=masks,
                                    W_l=W_l,H_l=H_l,C_l=C_l,
                                    pos_l=pos_l,
                                    frame_inds=frame_inds,
                                    mode='3D'
                                )
        else:
            # 2d history positional encoding
            pos_l_objs = self.extract_history_posenc_2D3D(
                                    B,S,M,
                                    obj_centroids=obj_centroids,
                                    positions=positions,
                                    masks=masks,
                                    W_l=W_l,H_l=H_l,C_l=C_l,
                                    pos_l=pos_l,
                                    frame_inds=frame_inds,
                                    mode='2D'
                                )
        return pos_l_objs


    def extract_history_posenc_2D3D(
        self,
        B,
        S,
        M,
        obj_centroids=None,
        positions=None,
        masks=None,
        C_l=None,
        W_l=None,
        H_l=None,
        pos_l=None,
        frame_inds=None,
        mode='2D'
    ):
        '''
        mode: '2D' or '3D' to get positional 3D from position + centroids or 2D encodings
        '''
        
        if mode=='3D':
            with torch.no_grad():
                if self.use_3d_obj_centroids:
                    if self.learned_3d_pos_enc:
                        # append zeros since rotations are zero for all objs
                        pos_enc_input = torch.cat([obj_centroids, torch.zeros(B,S,M,2).to(self.device)], dim=-1) 
                        if self.add_whole_image_mask:
                            # append agent position for whole image features
                            pos_enc_input[:,:,0,:] = positions
                        pos_l_objs_3d = self.pos_enc_3d(pos_enc_input.flatten(1,2).float()).transpose(1,2).reshape(B,S,M,self.hidden_dim)
                    else:
                        # get obj 3d pos enc from centroid
                        obj_centroids_rounded = torch.round(obj_centroids*(1/self.step_size))*self.step_size
                        position_inds_obj_centroids = (obj_centroids_rounded + self.max_distance_grid)/self.step_size
                        rotation_ind_obj = ((torch.zeros((B,S,M)) + 360)/self.dt, (torch.zeros((B,S,M)) + 360)/self.horizon_dt)
                        rotation_inds_obj = self.rot2dto1d_indices[rotation_ind_obj[0].flatten().long(), rotation_ind_obj[1].flatten().long()].reshape(B,S,M)
                        pos_l_objs_3d = self.pos_enc_3d[position_inds_obj_centroids[:,:,:,0].flatten().long(), position_inds_obj_centroids[:,:,:,1].flatten().long(), rotation_inds_obj.flatten()].to(self.device).reshape(B,S,M,self.hidden_dim)

                        if self.add_whole_image_mask:
                            # get agent 3d pos enc
                            position_inds = (positions[:,:,:2] + self.max_distance_grid)/self.step_size
                            rotation_ind = ((positions[:,:,2] + 360)/self.dt, (positions[:,:,3] + 360)/self.horizon_dt)
                            rotation_inds = self.rot2dto1d_indices[rotation_ind[0].flatten().long(), rotation_ind[1].flatten().long()].reshape(B,S)
                            pos_l_agent_3d = self.pos_enc_3d[position_inds[:,:,0].flatten().long(), position_inds[:,:,1].flatten().long(), rotation_inds.flatten()].to(self.device).reshape(B,S,1,self.hidden_dim)
                            pos_l_objs_3d[:,:,0] = pos_l_agent_3d.squeeze(2) # make whole image mask (first index) the position of the agent

                else:
                    if self.learned_3d_pos_enc:
                        pos_l_objs_3d = self.pos_enc_3d(positions.float()).transpose(1,2).reshape(B,S,1,self.hidden_dim).expand(B,S,M,self.hidden_dim)
                    else:
                        position_inds = (positions[:,:,:2] + self.max_distance_grid)/self.step_size
                        rotation_ind = ((positions[:,:,2] + 360)/self.dt, (positions[:,:,3] + 360)/self.horizon_dt)
                        rotation_inds = self.rot2dto1d_indices[rotation_ind[0].flatten().long(), rotation_ind[1].flatten().long()].reshape(B,S)
                        pos_l_objs_3d = self.pos_enc_3d[position_inds[:,:,0].flatten().long(), position_inds[:,:,1].flatten().long(), rotation_inds.flatten()].to(self.device).reshape(B,S,1,self.hidden_dim).expand(B,S,M,self.hidden_dim)
                pos_l_objs = pos_l_objs_3d # add 3d positional embedding

        elif mode=='2D':
            # 2d object positional encoding
            masks_cx, masks_cy = masks[:,:,0]*W_l, masks[:,:,1]*W_l # mask is actually a box (cx, cy, lx, ly)
            masks_cx2, masks_cy2 = masks_cx.long().unsqueeze(-1).expand(B*S, M, self.hidden_dim).transpose(2,1), masks_cy.long().unsqueeze(-1).expand(B*S, M, self.hidden_dim).transpose(2,1)
            masks_x2 = masks_cx2*masks_cy2
            pos_l_objs = torch.gather(pos_l.reshape(B*S, C_l, H_l*W_l), 2, masks_x2).reshape(B,S, C_l, M).transpose(3,2)
            pos_l_objs_frame = self.frame_embed[frame_inds].reshape(1,S,1,self.hidden_dim).expand(B,S,M,self.hidden_dim)
            pos_l_objs += pos_l_objs_frame # add frame positional embedding
        else:
            assert(False) # wrong mode
        
        return pos_l_objs

    def extract_action_history_tokens(
        self,
        action_history,
        frame_inds,
        action_pad_frame=None
    ):  
        B,N = action_history.shape
        action_history = action_history.reshape(B*N)

        action_mask = action_history==-1

        invalid_actions = torch.where(action_mask) # history can be less than max history frames, these frames are tagged with -1
        valid_actions = torch.where(~action_mask) # history can be less than max history frames, these frames are tagged with -1
        
        action_tokens_ = self.action_token(action_history[valid_actions])
        action_tokens = torch.zeros([B*N,self.hidden_dim]).to(self.device).to(action_tokens_.dtype)
        action_tokens[valid_actions] = action_tokens_

        action_tokens = action_tokens.view(B,N,1,self.hidden_dim)
        action_pos = self.frame_embed[frame_inds].view(1, len(frame_inds), 1, self.hidden_dim).expand(B, len(frame_inds), 1, self.hidden_dim)
        action_mask = action_mask.view(B,N,1).to(torch.bool)

        if action_pad_frame is not None:
            action_tokens = torch.cat([action_tokens, torch.zeros(B, action_pad_frame, 1, self.hidden_dim).to(self.device)], dim=1)
            if not action_pos.shape[1]==N+action_pad_frame: # frame inds may already have this padding
                action_pos = torch.cat([action_pos, torch.zeros(B, action_pad_frame, 1, self.hidden_dim).to(self.device)], dim=1)
            action_mask = torch.cat([action_mask, torch.ones(B, action_pad_frame, 1).to(self.device).to(torch.bool)], dim=1) # padding should have attention mask

        return action_tokens, action_mask, action_pos

    def apply_backbone_sg(
        self,
        samples,
    ):
        '''
        Only save gradients for forward pass of current observation
        Note: this a bit slower as it does 2 forward passes
        '''
        B, S, C, H, W = samples.shape
        inds_g = torch.zeros(S).to(self.device).to(bool)
        inds_g[args.max_steps_per_subgoal-1] = True # inds to save gradients
        samples_g = samples[:,inds_g].contiguous().view(B*1, C, H, W).unbind(0)
        samples_sg = samples[:,~inds_g].contiguous().view(B*(S-1), C, H, W).unbind(0)
        if not isinstance(samples_g, NestedTensor):
            samples_g = nested_tensor_from_tensor_list(samples_g)
            samples_sg = nested_tensor_from_tensor_list(samples_sg)
        features_g, pos_g = self.backbone(samples_g) 
        with torch.no_grad():
            features_sg, pos_sg = self.backbone(samples_sg) 
        features = []
        pos = []
        for l in range(len(features_g)):
            feat_g = features_g[l]
            feat_sg = features_sg[l]
            p_g = pos_g[l]
            p_sg = pos_sg[l]
            src_g, mask_g = feat_g.decompose()
            src_sg, mask_sg = feat_sg.decompose()
            _,C_l,H_l,W_l = src_sg.shape

            # feats
            src_sg = src_sg.view(B,S-1,C_l,H_l,W_l)
            src_g = src_g.view(B,1,C_l,H_l,W_l)
            src = torch.zeros([B,S,C_l,H_l,W_l]).to(self.device).to(src_g.dtype)
            src[:,inds_g] = src_g
            src[:,~inds_g] = src_sg

            # mask
            mask_sg = mask_sg.view(B,S-1,H_l,W_l)
            mask_g = mask_g.view(B,1,H_l,W_l)
            mask = torch.zeros([B,S,H_l,W_l]).to(self.device).to(mask_g.dtype)
            mask[:,inds_g] = mask_g
            mask[:,~inds_g] = mask_sg

            # pos encoding
            p_sg = p_sg.view(B,S-1,self.hidden_dim,H_l,W_l)
            p_g = p_g.view(B,1,self.hidden_dim,H_l,W_l)
            pos_l = torch.zeros([B,S,self.hidden_dim,H_l,W_l]).to(self.device).to(p_g.dtype)
            pos_l[:,inds_g] = p_g
            pos_l[:,~inds_g] = p_sg

            src = src.view(B*S,C_l,H_l,W_l)
            mask = mask.view(B*S,H_l,W_l)
            features.append(NestedTensor(src, mask))
            pos_l = pos_l.view(B*S,self.hidden_dim,H_l,W_l)
            pos.append(pos_l)
        return features, pos

    def extract_mask_features(
        self, 
        num_masks, 
        obj_b,
        masks,
        src_b,
        pos_l_b,
        b,
        s,
        frame_num,
        ):
        '''
        num_masks: number of masks to iterate
        obj_b: masks downsampled to feature resolution
        masks: WxH full masks (not downsampled) - used if no downsampled values are found
        src_b: downsampled feature map for batch
        pos_l_b: positional embeddings for the feature map
        b: batch index
        frame_num: frame number index
        '''

        obj_feat_ = []
        obj_pos_l_ = []
        for n in range(num_masks):
            obj_b_n = obj_b[n]
            where_masks = torch.where(obj_b_n)

            # # pos encoding
            if len(where_masks[0])==0: 
                where_masks = torch.where(masks[b,s,n]) 
                # if no points in interpolated mask, take nearest point to median of mask
                y_m, x_m = torch.median(where_masks[0])*pos_l_b.shape[-1]//masks.shape[-1], torch.median(where_masks[1])*pos_l_b.shape[-1]//masks.shape[-1]
                where_masks = (torch.tensor([y_m], dtype=torch.int64), torch.tensor([x_m], dtype=torch.int64)) 
            else:
                y_m, x_m = torch.median(where_masks[0]), torch.median(where_masks[1])
            
            src_n = src_b[:,where_masks[0],where_masks[1]].mean(dim=1)
            obj_feat_.append(src_n)
            pos_obj = pos_l_b[:,y_m,x_m]
            obj_pos_l_.append(pos_obj)

        # pad obj features and pos encodings for max objects
        num_pad = self.max_objs - num_masks
        pad_feat = torch.zeros(num_pad, self.hidden_dim).to(self.device)
        pad_pos_l = torch.zeros(num_pad, self.hidden_dim).to(self.device)
        if len(obj_feat_)>0:
            obj_feat_ = torch.stack(obj_feat_)
            obj_feat_ = torch.cat([obj_feat_, pad_feat], dim=0)
            obj_pos_l_ = torch.stack(obj_pos_l_)
            frame_pos_embed = self.frame_embed[frame_num].view(1, -1).expand(obj_pos_l_.shape[0], self.hidden_dim)
            obj_pos_l_ = obj_pos_l_ + frame_pos_embed
            obj_pos_l_ = torch.cat([obj_pos_l_, pad_pos_l], dim=0)
        else:
            obj_feat_ = pad_feat 
            obj_pos_l_ = pad_pos_l 
        
        obj_mask_ = torch.ones(self.max_objs).to(torch.bool).to(self.device)
        obj_mask_[:num_masks] = False

        return obj_feat_, obj_pos_l_, obj_mask_

    @torch.jit.unused
    def _set_aux_loss(self, out_aux):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        out = []
        for l in range(out_aux[list(out_aux.keys())[0]].shape[0]-1):
            out_ = {}
            for k in out_aux.keys():
                out_[k] = out_aux[k][l]
            out.append(out_)
        return out

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vector):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_boxes': b, 'pred_vectors': c}
    #             for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_vector[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, 
                with_vector=False, 
                processor_dct=None, 
                vector_loss_coef=0.7, 
                no_vector_loss_norm=False,
                vector_start_stage=0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        self.vector_loss_coef = vector_loss_coef
        self.no_vector_loss_norm = no_vector_loss_norm
        self.vector_start_stage = vector_start_stage

        print(f'Training with {6-self.vector_start_stage} vector stages.')

        print(f"Training with vector_loss_coef {self.vector_loss_coef}.")

        if not self.no_vector_loss_norm:
            print('Training with vector_loss_norm.')

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # also allows for no objects in view
        target_classes_o = []
        for t, (_, J) in zip(targets, indices):
            if len(J)>0:
                target_classes_o.append(t["labels"][J])
        if len(target_classes_o)>0:
            target_classes_o = torch.cat(target_classes_o)
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_class_ce': loss_ce}

        if log:
            if len(target_classes_o)>0:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_intermediate(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        This is a binary loss
        """
        assert 'pred_interms' in outputs
        src_logits = outputs['pred_interms']

        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.full(src_logits.shape[:2], 2,
                                    dtype=torch.int64, device=src_logits.device)
        
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # also allows for no objects in view
        target_classes_o = []
        for t, (_, J) in zip(targets, indices):
            if len(J)>0:
                labels = t["labels"][J]
                interms = t["obj_targets"]
                if len(interms.shape)>1:
                    interms = interms.squeeze(0)
                target_classes_o_ = torch.zeros(len(labels), dtype=torch.long, device=src_logits.device) # 0 is not intermediate index
                for m in range(interms.shape[0]):
                    where_interm = torch.where(J.to(interms.device)==interms[m])[0]
                    target_classes_o_[where_interm] = 1 # 1 is intermediate index
                target_classes_o.append(target_classes_o_)
        if len(target_classes_o)>0:
            target_classes_o = torch.cat(target_classes_o)
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_interm_ce': loss_ce}

        if log:
            if len(target_classes_o)>0:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['intermediate_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_action(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_actions' in outputs
        src_logits = outputs['pred_actions'].squeeze(1)

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["expert_action"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.cat([t["expert_action"] for t in targets]) #.unsqueeze(1)
        # target_classes = torch.full(src_logits.shape[:1], src_logits.shape[2],
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1]],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device).cpu()
        # target_classes_onehot.scatter_(1, target_classes.cpu(), 1)

        loss_ce = F.cross_entropy(src_logits, target_classes) #, self.empty_weight)
        losses = {'loss_action_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['action_error'] = 100 - accuracy(src_logits, target_classes)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        # if len(idx[0])==0:
        #     losses = {}
        #     losses['loss_bbox'] = torch.tensor(0.0).to(device)
        #     losses['loss_giou'] = torch.tensor(0.0).to(device)

        src_boxes = outputs['pred_boxes'][idx]
        # target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # also allows for no objects in view
        target_boxes = []
        for t, (_, i) in zip(targets, indices):
            if len(i)>0:
                target_boxes.append(t['boxes'][i])

        if len(target_boxes) == 0:
            losses = {
                "loss_bbox": src_boxes.sum() * 0,
                "loss_giou": src_boxes.sum() * 0,
            }
            return losses

        target_boxes = torch.cat(target_boxes, dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_vectors" in outputs

        

        # time1 = time.time()

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_vectors"]
        src_boxes = outputs['pred_boxes']
        # TODO use valid to mask invalid areas due to padding in loss
        # target_boxes = torch.cat([box_ops.box_cxcywh_to_xyxy(t['boxes'][i]) for t, (_, i) in zip(targets, indices)], dim=0)
        # target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()

        src_vectors = src_masks[src_idx]
        
        # also allows for no objects in view
        target_boxes = []
        target_masks_ = []
        valid = []
        for t, (_, i) in zip(targets, indices):
            if len(i)>0:
                target_boxes.append(box_ops.box_cxcywh_to_xyxy(t['boxes'][i]))
            target_masks_.append(t["masks"])

        if len(target_boxes) == 0:
            losses = {
                "loss_vector": src_vectors.sum() * 0
            }
            return losses

        target_boxes = torch.cat(target_boxes, dim=0) #.to(device=src_masks.device)
        target_masks, valid = nested_tensor_from_tensor_list(target_masks_).decompose()
        target_masks = target_masks.to(src_masks) #.to(device=src_masks.device)
        src_boxes = src_boxes[src_idx]
        target_masks = target_masks[tgt_idx]
        
        # scale boxes to mask dimesnions
        N, mask_w, mask_h = target_masks.shape
        target_sizes = torch.as_tensor([mask_w, mask_h]).unsqueeze(0).repeat(N, 1).to(device=src_masks.device)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        target_boxes = target_boxes * scale_fct

        # for b in range(target_masks.shape[0]):
        #     # for m in range(target_masks.shape[1]):
        #     plt.figure(1); plt.clf()
        #     plt.imshow(target_masks[b].cpu().numpy())
        #     plt.savefig('images/test.png')
        #     st()

        # for b in range(target_masks.shape[0]):
        #     # for m in range(target_masks.shape[1]):
        #     plt.figure(1)
        #     plt.clf()
        #     mask = np.expand_dims(np.float32(target_masks[b].cpu().numpy()), axis=2).repeat(3,2)
        #     box = target_boxes[b] * mask.shape[0]
        #     mask = cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0, 1, 0), 1)
        #     plt.imshow(mask)
        #     plt.savefig('images/test.png')
        #     st()


        # crop gt_masks
        n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
        gt_masks = BitMasks(target_masks)
        gt_masks = gt_masks.crop_and_resize(target_boxes, gt_mask_len).float()
        target_masks = gt_masks

        # for b in range(target_masks.shape[0]):
        #     # for m in range(target_masks.shape[1]):
        #     plt.figure(1)
        #     plt.clf()
        #     mask = np.expand_dims(np.float32(target_masks[b].cpu().numpy()), axis=2).repeat(3,2)
        #     # box = target_boxes[b] * mask.shape[0]
        #     # mask = cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0, 1, 0), 1)
        #     plt.imshow(mask)
        #     plt.savefig('images/test.png')
        #     st()

        

        # perform dct transform
        target_vectors = []
        for i in range(target_masks.shape[0]):
            gt_mask_i = ((target_masks[i,:,:] >= 0.5)* 1).to(dtype=torch.uint8) 
            gt_mask_i = gt_mask_i.cpu().numpy().astype(np.float32)
            coeffs = cv2.dct(gt_mask_i)
            coeffs = torch.from_numpy(coeffs).flatten()
            coeffs = coeffs[torch.tensor(self.processor_dct.zigzag_table)]
            gt_label = coeffs.unsqueeze(0)
            target_vectors.append(gt_label)

        target_vectors = torch.cat(target_vectors, dim=0).to(device=src_vectors.device)
        losses = {}

        # time2 = time.time()
        # print("time loss masks:", time2-time1)
        
        if self.no_vector_loss_norm:
            losses['loss_vector'] = self.vector_loss_coef * F.l1_loss(src_vectors, target_vectors, reduction='none').sum() / num_boxes
        else:
            losses['loss_vector'] = self.vector_loss_coef * F.l1_loss(src_vectors, target_vectors, reduction='mean')
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx.long(), src_idx.long()
        # return indices[0]

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx.long(), tgt_idx.long()
        # return indices[1]

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'intermediate': self.loss_intermediate,
            'action': self.loss_action,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks' and i < self.vector_start_stage:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss in ['labels', 'intermediate', 'action']:
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', 'intermediate', 'action']:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss in ['labels', 'intermediate', 'action']:
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, processor_dct=None):
        super().__init__()
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, outputs, target_sizes, do_masks=True, return_features=False, features=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, out_vector, out_interms, out_actions = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_vectors'], outputs['pred_interms'], outputs['pred_actions'] #, outputs['batch_inds']
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        if True:
            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 50, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

            # if args.do_predict_oop:
            # out_logits2 = outputs['pred_logits2']
            prob_interms = out_interms.sigmoid()
            topk_boxes_interms = topk_boxes.unsqueeze(2).repeat(1,1,out_interms.shape[2])
            scores_interms = torch.gather(prob_interms, 1, topk_boxes_interms)
            labels_interms = torch.argmax(scores_interms, dim=2)
            scores_interms = torch.max(scores_interms, dim=2).values

            prob_actions = out_actions.squeeze(1).softmax(1)
            labels_action = torch.argmax(prob_actions, dim=1)

            if self.processor_dct is not None:
                n_keep = self.processor_dct.n_keep
                vectors = torch.gather(out_vector, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, n_keep))
        else:
            prob = out_logits.sigmoid()
            # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 25, dim=1)

            scores = prob.max(dim=2).values
            # topk_boxes = topk_indexes // out_logits.shape[2]
            labels = prob.max(dim=2).indices
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
            if self.processor_dct is not None:
                n_keep = self.processor_dct.n_keep
                vectors = out_vector #torch.gather(out_vector, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, n_keep))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        if self.processor_dct is not None and do_masks:
            masks = []
            n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
            b, r, c = vectors.shape
            for bi in range(b):
                outputs_masks_per_image = []
                for ri in range(r):
                    # here visual for training
                    idct = np.zeros((gt_mask_len ** 2))
                    idct[:n_keep] = vectors[bi,ri].cpu().numpy()
                    idct = self.processor_dct.inverse_zigzag(idct, gt_mask_len, gt_mask_len)
                    re_mask = cv2.idct(idct)
                    max_v = np.max(re_mask)
                    min_v = np.min(re_mask)
                    re_mask = np.where(re_mask>(max_v+min_v) / 2., 1, 0)
                    re_mask = torch.from_numpy(re_mask)[None].float()
                    outputs_masks_per_image.append(re_mask)
                outputs_masks_per_image = torch.cat(outputs_masks_per_image, dim=0).to(out_vector.device)
                # here padding local mask to global mask
                outputs_masks_per_image = retry_if_cuda_oom(paste_masks_in_image)(
                    outputs_masks_per_image,  # N, 1, M, M
                    boxes[bi],
                    (img_h[bi], img_w[bi]),
                    threshold=0.5,
                )
                outputs_masks_per_image = outputs_masks_per_image.unsqueeze(1).cpu()
                masks.append(outputs_masks_per_image)
            masks = torch.stack(masks)

        # N = scores.shape[1]
        # _, N, C, H, W = masks.shape
        # # reshape to (B, S-1, ...)
        # scores = scores.reshape(B, S-1, N)
        # labels = labels.reshape(B, S-1, N)
        # boxes = boxes.reshape(B, S-1, N, 4)
        # masks = masks.reshape(B, S-1, N, C, H, W)

        if return_features and features is not None:
            features_keep = torch.gather(features, 1, topk_boxes.unsqueeze(-1).repeat(1,1,features.shape[-1]))


        if self.processor_dct is None or not do_masks:
            results1 = [{'scores': s, 'labels': l, 'boxes': b, 'scores_interm':i, 'labels_interm':e, 'labels_action':a} for s, l, b, i, e, a in zip(scores, labels, boxes, scores_interms, labels_interms, labels_action)]
            # if args.do_predict_oop:
            #     results2 = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores2, labels2, boxes)]
        else:
            results1 = [{'scores': s, 'labels': l, 'boxes': b, 'masks': m, 'scores_interm':i, 'labels_interm':e, 'labels_action':a} for s, l, b, m, i, e, a in zip(scores, labels, boxes, masks, scores_interms, labels_interms, labels_action)]
            # if args.do_predict_oop:
            #     results2 = [{'scores': s, 'labels': l, 'boxes': b, 'masks':m} for s, l, b, m in zip(scores2, labels2, boxes, masks)]

        results = {'pred1':results1}
        # if args.do_predict_oop:
        #     results['pred2'] = results2

        return results


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5, processor_dct=None):
        super().__init__()
        self.threshold = threshold
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(num_classes, num_actions, actions2idx):
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     num_classes = 250
    device = torch.device(args.device)

    if 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args) 
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer() if not args.checkpoint else build_cp_deforamble_transformer()
    if args.with_vector:
        processor_dct = ProcessorDCT(args.n_keep, args.gt_mask_len)
    model = SOLQ(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_actions=num_actions,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        with_vector=args.with_vector, 
        processor_dct=processor_dct if args.with_vector else None,
        vector_hidden_dim=args.vector_hidden_dim,
        actions2idx=actions2idx
    )

    matcher = build_matcher(args)
    weight_dict = {
        'loss_class_ce': args.cls_loss_coef, 
        'loss_bbox': args.bbox_loss_coef,
        'loss_interm_ce': args.interm_loss_coef,
        'loss_action_ce': args.action_loss_coef,
        }
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_vector"] = 1
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        print("Doing mask loss!")
        losses += ["masks"]
    if args.do_intermediate_loss:
        losses += ["intermediate"]
    if args.do_action_loss:
        losses += ["action"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, 
                                                                        with_vector=args.with_vector, 
                                                                        processor_dct=processor_dct if args.with_vector else None,
                                                                        vector_loss_coef=args.vector_loss_coef,
                                                                        no_vector_loss_norm=args.no_vector_loss_norm,
                                                                        vector_start_stage=args.vector_start_stage)
    criterion.to(device)
    # postprocessors = {'bbox': PostProcess(processor_dct=processor_dct if (args.with_vector and args.eval) else None)}
    postprocessors = {'bbox': PostProcess(processor_dct=processor_dct if (args.with_vector) else None)}

    if args.masks: # and args.eval:
        postprocessors['segm'] = PostProcessSegm(processor_dct=processor_dct if args.with_vector else None)
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
