import time

import numpy as np
import torch.nn as nn
import torch
from ..model_utils import model_nms_utils

from .anchor_head_template import AnchorHeadTemplate
from pcdet.models.model_utils.basic_block_2d import build_block
from pcdet.ops.iou3d_nms import iou3d_nms_utils

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.feature_map_stride = model_cfg.ANCHOR_GENERATOR_CONFIG[0]['feature_map_stride']

        if self.model_cfg.get('VOXEL_SIZE', None):
            self.voxel_size = model_cfg.VOXEL_SIZE

        # build pre block
        if self.model_cfg.get('PRE_BLOCK', None):
            pre_block = []

            block_types = self.model_cfg.PRE_BLOCK.BLOCK_TYPE
            num_filters = self.model_cfg.PRE_BLOCK.NUM_FILTERS
            layer_strides = self.model_cfg.PRE_BLOCK.LAYER_STRIDES
            kernel_sizes = self.model_cfg.PRE_BLOCK.KERNEL_SIZES
            paddings = self.model_cfg.PRE_BLOCK.PADDINGS
            in_channels = input_channels

            for i in range(len(num_filters)):
                pre_block.extend(build_block(
                    block_types[i], in_channels, num_filters[i], kernel_size=kernel_sizes[i],
                    stride=layer_strides[i], padding=paddings[i], bias=False
                ))
                in_channels = num_filters[i]
            self.pre_block = nn.Sequential(*pre_block)

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_IOU', None) is not None and model_cfg.USE_IOU == True:
            # self.conv_iou = nn.Conv2d(
            #     input_channels,
            #     self.num_anchors_per_location * self.num_class,
            #     kernel_size=1
            # )
            # # TODO: zhanghn 2024-4-22
            # self.conv_iou = nn.Conv2d(
            #     input_channels,
            #     self.num_anchors_per_location * 2,
            #     kernel_size=1
            # )
            # TODO: zhanghn 2024-4-30
            self.conv_iou = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * 1,
                kernel_size=1
            )
        else:
            self.conv_iou = None

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()


    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        in_feature_2d = spatial_features_2d

        if self.model_cfg.get('VOXEL_SIZE', None):
            output_size = [self.grid_size[0], self.grid_size[1]]
            in_feature_2d = nn.functional.interpolate(
                in_feature_2d, output_size, mode='bilinear', align_corners=False
            )

        if hasattr(self, 'pre_block'):
            in_feature_2d = self.pre_block(in_feature_2d)
            data_dict['spatial_features_2d_preblock'] = in_feature_2d

        cls_preds = self.conv_cls(in_feature_2d)
        box_preds = self.conv_box(in_feature_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(in_feature_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.conv_iou is not None:
            iou_cls_preds = self.conv_iou(in_feature_2d)
            iou_cls_preds = iou_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['iou_cls_preds'] = iou_cls_preds
        else:
            iou_cls_preds = None

        if self.training:
            target_boxes = data_dict['gt_boxes']


            # visualization code
            # num_gt_boxes = target_boxes.shape[1]

            # label assign kd
            num_target_boxes_list = [0 for _ in range(data_dict['batch_size'])]
            weight_anchor_list = [None for _ in range(data_dict['batch_size'])]
            if self.kd_head is not None and not self.is_teacher and self.model_cfg.get('LABEL_ASSIGN_KD', None):
                
                # import ipdb; ipdb.set_trace(context=20)
                # from pcdet.datasets.dataset import DatasetTemplate
                # DatasetTemplate.__vis_open3d__(
                #     data_dict['points'][:, 1:].cpu().numpy(), target_boxes[0, :, :7].cpu().numpy(),
                #     data_dict['decoded_pred_tea'][0]['pred_boxes'][:, :7].cpu().numpy()
                # )

                if not self.is_teacher:
                    if  'NMS_CONFIG' in self.model_cfg:
                        # TODO: add by zhanghn, 2024-5-13
                        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                            batch_size=data_dict['batch_size'],
                            cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
                        )

                        # cost_thre = np.zeros((data_dict['batch_size'], self.num_class))
                        cost_thre = []
                        for index in range(data_dict['batch_size']):
                            cls_preds_tmp, label_preds_tmp = torch.max(batch_cls_preds[index].sigmoid(), dim=-1)
                            selected, _ = model_nms_utils.class_agnostic_nms(
                                box_scores=cls_preds_tmp, box_preds=batch_box_preds[index],
                                nms_config=self.model_cfg.NMS_CONFIG,
                                score_thresh=self.model_cfg.NMS_CONFIG.SCORE_THRESH
                            )
                            label_preds_tmp = label_preds_tmp[selected]
                            cur_pred_tmp = batch_box_preds[index][selected]
                            cur_cls_tmp = cls_preds_tmp[selected]

                            valid_mask = data_dict['gt_boxes'][index][:, 3] > 0
                            valid_gt_boxes = data_dict['gt_boxes'][index][valid_mask]
                            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(
                                cur_pred_tmp[:, :7], valid_gt_boxes[:, :7]
                            )
                            # ious, _ = torch.max(iou_matrix, dim=1)
                            # cur_cost = cur_cls_tmp.pow(0.5) * ious.pow(0.5)
                            # for cls_i in range(self.num_class):
                            #     if (label_preds_tmp==cls_i).sum() > 0:
                            #         cost_thre[index, cls_i] = cur_cost[label_preds_tmp==cls_i].mean()
                        
                            distance = (cur_pred_tmp[:, None, 0:3] - valid_gt_boxes[None, :, 0:3]).norm(dim=-1)  # (N, M)
                            # print(distance.shape)
                            # print(cur_cls_tmp.shape)
                            # exit()
                            TOPK = 9
                            _, topk_idxs = distance.topk(TOPK, dim=0, largest=False)  # (K, M)
                            candidate_ious = iou_matrix[topk_idxs, torch.arange(valid_gt_boxes.shape[0])]  # (K, M)

                            cur_gt_label = valid_gt_boxes[:, -1][None, :].repeat(TOPK, 1) - 1
                            cur_pred_label = label_preds_tmp[topk_idxs]
                            cur_pred_cls = cur_cls_tmp[topk_idxs]
                            mask_label = cur_gt_label == cur_pred_label
                            cost = cur_pred_cls.pow(0.5) * candidate_ious.pow(0.5)
                            cost *= mask_label

                            # calcualte mean
                            cost_mean_per_gt = cost.sum(dim=0) / torch.clamp(mask_label.sum(dim=0), 1)
                            # calcualte std
                            cost_std_per_gt2 = ((cost - cost_mean_per_gt[None, :].repeat(TOPK, 1)).pow(2) * mask_label).sum(dim=0)
                            cost_std_per_gt = (cost_std_per_gt2 / torch.clamp((mask_label.sum(dim=0) - 1), 1)).pow(0.5)
                            cost_thresh_per_gt = cost_mean_per_gt + cost_std_per_gt + 1e-6
                            # print(iou_thresh_per_gt)
                            cost_thre.append(cost_thresh_per_gt)
                        
                        data_dict['cost_thre'] = cost_thre
                    else:
                       data_dict['cost_thre'] = None
                

                # TODO: change by zhanghn, 2024-5-3
                if self.model_cfg.LABEL_ASSIGN_KD.USE_STAGE_ONE:   # use stage 1 for label assignment
                    target_boxes, num_target_boxes_list, weight_anchor_list = self.kd_head.parse_teacher_pred_to_targets(
                        kd_cfg=self.model_cfg.LABEL_ASSIGN_KD, pred_boxes_tea=data_dict['decoded_pred_tea'][-1],
                        gt_boxes=target_boxes
                    )
                else:
                    # target_boxes, num_target_boxes_list = self.kd_head.parse_teacher_pred_to_targets(
                    #     kd_cfg=self.model_cfg.LABEL_ASSIGN_KD, pred_boxes_tea=data_dict['decoded_pred_tea'],
                    #     gt_boxes=target_boxes, cost_thre = data_dict['cost_thre']
                    # )
                    target_boxes, num_target_boxes_list, weight_anchor_list = self.kd_head.parse_teacher_pred_to_targets(
                        kd_cfg=self.model_cfg.LABEL_ASSIGN_KD, pred_boxes_tea=data_dict['decoded_pred_tea'],
                        gt_boxes=target_boxes, cost_thre = data_dict['cost_thre']
                    )

                # import ipdb; ipdb.set_trace(context=20)
                # from pcdet.datasets.dataset import DatasetTemplate
                # num_target_boxes = int(target_boxes.shape[1] - num_gt_boxes)
                # DatasetTemplate.__vis_open3d__(
                #     data_dict['points'][:, 1:].cpu().numpy(), target_boxes[0, num_target_boxes:, :7].cpu().numpy(),
                #     target_boxes[0, :num_target_boxes, :7].cpu().numpy()
                # )
            # exit()

            

            # print(target_boxes.shape)
            # print(target_boxes[:,:,-1])
            # exit()

            
            if  self.model_cfg.TARGET_ASSIGNER_CONFIG.NAME == 'AxisAlignedTargetAssigner':
                targets_dict = self.assign_targets(
                    gt_boxes=target_boxes
                )
            else:
                # TODO: add by zhanghn, 2024-5-6
                if None in weight_anchor_list:
                    targets_dict = self.assign_targets_iou(
                        gt_boxes=target_boxes, num_target_boxes_list=num_target_boxes_list
                    )
                else:
                    targets_dict = self.assign_targets_iou_weight(
                            gt_boxes=target_boxes, num_target_boxes_list=num_target_boxes_list,
                            weight_anchor_list = weight_anchor_list
                        )
                    # print(targets_dict['box_cls_labels'][0][(targets_dict['box_cls_labels'][0]>0)].shape)
                    # print(targets_dict['tea_weight'][0][(targets_dict['tea_weight'][0]<1) * (targets_dict['tea_weight'][0]!=0)].shape)
                    # print(targets_dict['tea_weight'].shape)
                    # exit()


            self.forward_ret_dict.update(targets_dict)

            if not self.is_teacher:
                data_dict['cls_preds'] = self.forward_ret_dict['cls_preds']
                data_dict['box_preds'] = self.forward_ret_dict['box_preds']
                data_dict['iou_preds'] = iou_cls_preds

                # # TODO: add by zhanghn, 2024-5-13
                # batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                #     batch_size=data_dict['batch_size'],
                #     cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
                # )

                # for index in range(data_dict['batch_size']):
                #     cls_preds_tmp, label_preds_tmp = torch.max(batch_cls_preds[index].sigmoid(), dim=-1)
                #     selected, selected_scores = model_nms_utils.class_agnostic_nms(
                #         box_scores=cls_preds_tmp, box_preds=batch_box_preds[index],
                #         nms_config=self.model_cfg.NMS_CONFIG,
                #         score_thresh=self.model_cfg.NMS_CONFIG.SCORE_THRESH
                #     )
                #     label_preds_tmp = label_preds_tmp[selected]
                #     print('************')
                #     print('aa', cls_preds_tmp[selected][label_preds_tmp==0].mean())
                #     print('bb', cls_preds_tmp[selected][label_preds_tmp==1].mean())
                #     print('cc', cls_preds_tmp[selected][label_preds_tmp==2].mean())
                #     print(batch_box_preds[index][selected].shape)
                #     print('____________')
                # # exit()


        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

            if self.is_teacher:
                data_dict['batch_cls_preds_tea_stage1'] = batch_cls_preds
                data_dict['batch_box_preds_tea_stage1'] = batch_box_preds

            data_dict['batch_iou_preds'] = iou_cls_preds

            # print('a', box_preds.shape)
            # print('b', batch_box_preds.shape)
            # exit()

            if self.is_teacher:
                data_dict['batch_cls_preds_tea_densehead'] = batch_cls_preds
                data_dict['batch_box_preds_tea_densehead'] = batch_box_preds
                data_dict['cls_preds_normalized_tea_densehead'] = False

            data_dict['cls_preds_vina'] = cls_preds

        return data_dict
