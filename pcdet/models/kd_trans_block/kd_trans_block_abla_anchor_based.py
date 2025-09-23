import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.spconv_utils import spconv
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ...utils import common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms, class_agnostic_nms_minor
from ...ops.iou3d_nms import iou3d_nms_utils


class KDPointTrans_ablation(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    # @torch.no_grad()
    def proposal_layer_tea2stu(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        # if batch_dict.get('rois', None) is not None:
        #     return batch_dict

        batch_size = batch_dict['batch_size']
        batch_box_preds_tea = batch_dict['batch_box_preds_tea_densehead']
        batch_cls_preds_tea = batch_dict['batch_cls_preds_tea_densehead']
        rois = batch_box_preds_tea.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds_tea.shape[-1]))
        roi_scores = batch_box_preds_tea.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds_tea.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        batch_box_preds_stu = batch_dict['batch_box_preds']
        batch_cls_preds_stu = batch_dict['batch_cls_preds']
        rois_stu = batch_box_preds_stu.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds_stu.shape[-1]))
        roi_scores_stu = batch_box_preds_stu.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels_stu = batch_box_preds_stu.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        dense_cls_preds_stu = batch_dict['cls_preds'].view(batch_size, -1, batch_cls_preds_stu.shape[-1])
        cls_select = dense_cls_preds_stu.new_zeros(
            (batch_size, nms_config.NMS_POST_MAXSIZE, batch_cls_preds_stu.shape[-1]))

        ''' stu-to-tea'''
        cls_select_reverse = dense_cls_preds_stu.new_zeros(
            (batch_size, nms_config.NMS_POST_MAXSIZE, batch_cls_preds_stu.shape[-1]))
        rois_reverse = batch_box_preds_tea.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE,
                                                      batch_box_preds_tea.shape[-1]))

        select_mask = batch_box_preds_tea.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds_tea.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            ''' tea-to-stu'''
            box_preds = batch_box_preds_tea[batch_mask]
            cls_preds = batch_cls_preds_tea[batch_mask]
            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            ''' stu-to-tea '''
            box_preds_stu = batch_box_preds_stu[batch_mask]
            cls_preds_stu = batch_cls_preds_stu[batch_mask]
            cur_roi_scores_stu, cur_roi_labels_stu = torch.max(cls_preds_stu, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config,
                    score_thresh=nms_config.SCORE_THRESH_MINOR,
                )

                # TODO: add for "student ---> select teacher"
                # selected_minor, selected_scores_minor = class_agnostic_nms_minor(
                #     box_scores=cur_roi_scores_stu, box_preds=box_preds_stu, nms_config=nms_config,
                #     score_thresh_minor=nms_config.SCORE_THRESH_MINOR,
                # )

            ''' tea-to-stu'''
            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

            # TODO: student
            rois_stu[index, :len(selected), :] = box_preds_stu[selected]
            roi_scores_stu[index, :len(selected)] = cur_roi_scores_stu[selected]
            roi_labels_stu[index, :len(selected)] = cur_roi_labels_stu[selected]

            cur_cls_preds = dense_cls_preds_stu[batch_mask]
            cls_select[index, :len(selected), :] = cur_cls_preds[selected]

            select_mask[index, :len(selected)] = 1.0


            ''' stu-to-tea '''
            # rois_reverse[index, :len(selected_minor), :] = box_preds[selected_minor]
            # cls_select_reverse[index, :len(selected_minor), :] = cur_cls_preds[selected_minor]
            # select_mask[index, :len(selected_minor)] = 1.0


        batch_dict['rois_tea'] = rois
        batch_dict['roi_scores_tea'] = roi_scores
        batch_dict['roi_labels_tea'] = roi_labels + 1
        batch_dict['has_class_labels_tea'] = True if batch_cls_preds_tea.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)

        # TODO: student
        batch_dict['rois_stu'] = rois_stu
        batch_dict['roi_scores_stu'] = roi_scores_stu
        batch_dict['roi_labels_stu'] = roi_labels_stu + 1
        batch_dict['has_class_labels_stu'] = True if batch_cls_preds_stu.shape[-1] > 1 else False

        batch_dict['cls_preds_selected'] = cls_select
        batch_dict['select_mask'] = select_mask
        # # TODO: ''' stu-to-tea '''
        # batch_dict['rois_reverse'] = rois_reverse
        # batch_dict['cls_preds_selected_reverse'] = cls_select_reverse
        # batch_dict['select_mask'] = select_mask

        return batch_dict


    def proposal_layer_stu2tea(self, batch_dict, nms_config):

        batch_size = batch_dict['batch_size']
        batch_box_preds_tea = batch_dict['batch_box_preds_tea_densehead']
        batch_cls_preds_tea = batch_dict['batch_cls_preds_tea_densehead']

        batch_box_preds_stu = batch_dict['batch_box_preds']
        batch_cls_preds_stu = batch_dict['batch_cls_preds']

        dense_cls_preds_stu = batch_dict['cls_preds'].view(batch_size, -1, batch_cls_preds_stu.shape[-1])

        ''' stu-to-tea'''
        cls_select_reverse = dense_cls_preds_stu.new_zeros(
            (batch_size, nms_config.NMS_POST_MAXSIZE, batch_cls_preds_stu.shape[-1]))
        rois_reverse = batch_box_preds_tea.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE,
                                                      batch_box_preds_tea.shape[-1]))

        select_mask = batch_box_preds_tea.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds_tea.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_box_preds_tea[batch_mask]
            box_preds_stu = batch_box_preds_stu[batch_mask]
            cls_preds_stu = batch_cls_preds_stu[batch_mask]
            cur_roi_scores_stu, cur_roi_labels_stu = torch.max(cls_preds_stu, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                # TODO: add for "student ---> select teacher"
                selected_minor, selected_scores_minor = class_agnostic_nms_minor(
                    box_scores=cur_roi_scores_stu, box_preds=box_preds_stu, nms_config=nms_config,
                    score_thresh_minor=nms_config.SCORE_THRESH_MINOR,
                )

            ''' stu-to-tea '''
            cur_cls_preds = dense_cls_preds_stu[batch_mask]
            rois_reverse[index, :len(selected_minor), :] = box_preds[selected_minor]
            cls_select_reverse[index, :len(selected_minor), :] = cur_cls_preds[selected_minor]
            select_mask[index, :len(selected_minor)] = 1.0

        # TODO: ''' stu-to-tea '''
        batch_dict['rois_new_tea'] = rois_reverse
        batch_dict['cls_preds_selected_kd'] = cls_select_reverse
        batch_dict['select_mask'] = select_mask

        return batch_dict


    def forward(self, batch_dict):

        if self.model_cfg.MODE == 'Stu2Tea':
            _ = self.proposal_layer_stu2tea(batch_dict, nms_config=self.model_cfg.NMS_CONFIG)

            batch_dict['re_run_tea_flag'] = True
            roi_tmp = batch_dict['rois'].detach()
            batch_dict['rois'] = batch_dict['rois_new_tea'].detach()
            teacher_model = batch_dict['teacher_model']
            teacher_model.roi_head(batch_dict)
            batch_dict['rois'] = roi_tmp.detach()
            batch_dict['re_run_tea_flag'] = None

            select_obj_mask = batch_dict['select_mask'].view(-1, 1)

            batch_dict['cls_preds_selected_kd'] = batch_dict['cls_preds_selected_kd'].view(-1, batch_dict['cls_preds_selected_kd'].shape[-1])
            cls_pred_selected_reverse_max, _ = batch_dict['cls_preds_selected_kd'].max(dim=1, keepdim=True)

            batch_dict['cls_preds_kd_stu'] = cls_pred_selected_reverse_max * select_obj_mask
            batch_dict['cls_preds_kd_tea'] = batch_dict['rcnn_cls_stu_to_tea']* select_obj_mask

        elif self.model_cfg.MODE == 'Tea2Stu':

            _ = self.proposal_layer_tea2stu(batch_dict, nms_config=self.model_cfg.NMS_CONFIG)

            select_obj_mask = batch_dict['select_mask'].view(-1, 1)
            cls_pred_selected_max, _ = batch_dict['cls_preds_selected'].view(-1, batch_dict['cls_preds_selected'].shape[-1]).max(dim=1, keepdim=True)

            batch_dict['cls_preds_kd_stu'] = cls_pred_selected_max * select_obj_mask
            batch_dict['cls_preds_kd_tea'] = batch_dict['rcnn_cls_tea'] * select_obj_mask


        elif self.model_cfg.MODE == 'Dual':

            _ = self.proposal_layer_tea2stu(batch_dict, nms_config=self.model_cfg.NMS_CONFIG)

            select_obj_mask = batch_dict['select_mask'].view(-1, 1)
            cls_pred_selected_max, _ = batch_dict['cls_preds_selected'].view(-1, batch_dict['cls_preds_selected'].shape[
                -1]).max(dim=1, keepdim=True)

            cur_stu_fea = cls_pred_selected_max * select_obj_mask
            cur_tea_fea = batch_dict['rcnn_cls_tea'] * select_obj_mask
            cur_mask = select_obj_mask

            _ = self.proposal_layer_stu2tea(batch_dict, nms_config=self.model_cfg.NMS_CONFIG)

            batch_dict['re_run_tea_flag'] = True
            roi_tmp = batch_dict['rois'].detach()
            batch_dict['rois'] = batch_dict['rois_new_tea'].detach()
            teacher_model = batch_dict['teacher_model']
            teacher_model.roi_head(batch_dict)
            batch_dict['rois'] = roi_tmp.detach()
            batch_dict['re_run_tea_flag'] = None

            select_obj_mask = batch_dict['select_mask'].view(-1, 1)

            batch_dict['cls_preds_selected_kd'] = batch_dict['cls_preds_selected_kd'].view(-1, batch_dict[
                'cls_preds_selected_kd'].shape[-1])
            cls_pred_selected_reverse_max, _ = batch_dict['cls_preds_selected_kd'].max(dim=1, keepdim=True)


            cur_stu_fea = torch.cat((cur_stu_fea, cls_pred_selected_reverse_max * select_obj_mask), dim=0)
            cur_tea_fea = torch.cat((cur_tea_fea, batch_dict['rcnn_cls_stu_to_tea'] * select_obj_mask), dim=0)
            cur_mask = torch.cat((cur_mask, select_obj_mask), dim=0)

            batch_dict['cls_preds_kd_stu'] = cur_stu_fea
            batch_dict['cls_preds_kd_tea'] = cur_tea_fea
            batch_dict['select_mask'] = cur_mask



        return batch_dict
