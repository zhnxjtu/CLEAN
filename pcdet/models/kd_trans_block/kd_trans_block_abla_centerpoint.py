import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ...utils.spconv_utils import spconv
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ...utils import common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms, class_agnostic_nms_minor
from .target_assigned_teacher.proposal_taget_layer_teacher import ProposalTargetLayer
from ..model_utils import centernet_utils


class KDPointTrans_ablation_cp(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range


    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, K=40):
        batch, num_class, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_classes = (topk_ind // K).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_classes, topk_ys, topk_xs

    def decode_bbox_from_heatmap(self, heatmap, rot_cos, rot_sin, center, center_z, dim,
                                 point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, K=100,
                                 circle_nms=False, score_thresh=None, post_center_limit_range=None):
        batch_size, num_class, _, _ = heatmap.size()

        # if circle_nms:
        #     # TODO: not checked yet
        #     assert False, 'not checked yet'
        #     heatmap = _nms(heatmap)

        scores, inds, class_ids, ys, xs = self._topk(heatmap, K=K)
        center = self._transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
        rot_sin = self._transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
        rot_cos = self._transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
        center_z = self._transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
        dim = self._transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)

        angle = torch.atan2(rot_sin, rot_cos)
        xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

        xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
        ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]
        if vel is not None:
            vel = self._transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
            box_part_list.append(vel)

        final_box_preds = torch.cat((box_part_list), dim=-1)
        final_scores = scores.view(batch_size, K)
        final_class_ids = class_ids.view(batch_size, K)

        assert post_center_limit_range is not None
        mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

        if score_thresh is not None:
            mask &= (final_scores > score_thresh)

        ret_pred_dicts = []
        for k in range(batch_size):
            cur_mask = mask[k]
            cur_boxes = final_box_preds[k, cur_mask]
            cur_scores = final_scores[k, cur_mask]
            cur_labels = final_class_ids[k, cur_mask]
            cur_index = inds[k, cur_mask]

            # if circle_nms:
            #     assert False, 'not checked yet'
            #     centers = cur_boxes[:, [0, 1]]
            #     boxes = torch.cat((centers, scores.view(-1, 1)), dim=1)
            #     keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=nms_post_max_size)
            #
            #     cur_boxes = cur_boxes[keep]
            #     cur_scores = cur_scores[keep]
            #     cur_labels = cur_labels[keep]

            ret_pred_dicts.append({
                'pred_boxes': cur_boxes,
                'pred_scores': cur_scores,
                'pred_labels': cur_labels,
                'index': cur_index,
            })
        return ret_pred_dicts

    def decode_bbox_from_heatmap_tea(self, heatmap, rot_cos, rot_sin, center, center_z, dim,
                                     rot_cos_tea, rot_sin_tea, center_tea, center_z_tea, dim_tea,
                                 point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, vel_tea=None, K=100,
                                 circle_nms=False, score_thresh=None, post_center_limit_range=None):
        batch_size, num_class, _, _ = heatmap.size()

        scores, inds, class_ids, ys, xs = self._topk(heatmap, K=K)
        center = self._transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
        rot_sin = self._transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
        rot_cos = self._transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
        center_z = self._transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
        dim = self._transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)

        angle = torch.atan2(rot_sin, rot_cos)
        xs_tmp = xs
        ys_tmp = ys
        xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

        xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
        ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]
        if vel is not None:
            vel = self._transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
            box_part_list.append(vel)

        final_box_preds = torch.cat((box_part_list), dim=-1)
        final_scores = scores.view(batch_size, K)
        final_class_ids = class_ids.view(batch_size, K)

        assert post_center_limit_range is not None
        mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

        if score_thresh is not None:
            mask &= (final_scores > score_thresh)


        #TODO: # FOR TEACHER
        center_tea = self._transpose_and_gather_feat(center_tea, inds).view(batch_size, K, 2)
        rot_sin_tea = self._transpose_and_gather_feat(rot_sin_tea, inds).view(batch_size, K, 1)
        rot_cos_tea = self._transpose_and_gather_feat(rot_cos_tea, inds).view(batch_size, K, 1)
        center_z_tea = self._transpose_and_gather_feat(center_z_tea, inds).view(batch_size, K, 1)
        dim_tea = self._transpose_and_gather_feat(dim_tea, inds).view(batch_size, K, 3)

        angle_tea = torch.atan2(rot_sin_tea, rot_cos_tea)
        xs_tea = xs_tmp.view(batch_size, K, 1) + center_tea[:, :, 0:1]
        ys_tea = ys_tmp.view(batch_size, K, 1) + center_tea[:, :, 1:2]

        xs_tea = xs_tea * feature_map_stride * voxel_size[0] + point_cloud_range[0]
        ys_tea = ys_tea * feature_map_stride * voxel_size[1] + point_cloud_range[1]


        box_part_list_tea = [xs_tea, ys_tea, center_z_tea, dim_tea, angle_tea]
        if vel_tea is not None:
            vel_tea = self._transpose_and_gather_feat(vel_tea, inds).view(batch_size, K, 2)
            box_part_list_tea.append(vel_tea)

        final_box_preds_tea = torch.cat((box_part_list_tea), dim=-1)
        # print('zz4', torch.isnan(final_box_preds_tea).any() or torch.isinf(final_box_preds_tea).any())
        # print('aa', final_box_preds_tea.shape)

        mask_tea = (final_box_preds_tea[..., :3] >= post_center_limit_range[:3]).all(2)
        mask_tea &= (final_box_preds_tea[..., :3] <= post_center_limit_range[3:]).all(2)
        # scores, inds, class_ids, ys, xs = self._topk(heatmap, K=K)
        if score_thresh is not None:
            mask_tea &= (final_scores > score_thresh)

        mask &= mask_tea
        # print(final_scores.shape)
        # print(score_thresh)
        # print(mask.sum())
        # exit()

        ret_pred_dicts = []
        ret_pred_dicts_tea = []
        for k in range(batch_size):
            cur_mask = mask[k]
            cur_boxes = final_box_preds[k, cur_mask]
            cur_scores = final_scores[k, cur_mask]
            cur_labels = final_class_ids[k, cur_mask]
            cur_index = inds[k, cur_mask]

            ret_pred_dicts.append({
                'pred_boxes': cur_boxes,
                'pred_scores': cur_scores,
                'pred_labels': cur_labels,
                'index': cur_index,
            })

            cur_boxes_tea = final_box_preds_tea[k, cur_mask]
            if self.model_cfg.MODE == 'Tea2Stu' or self.model_cfg.MODE == 'Dual':
                ret_pred_dicts_tea.append({
                    'pred_boxes': cur_boxes_tea,
                    'pred_scores': cur_scores,
                    'index': cur_index,
                })
            else:
                ret_pred_dicts_tea.append({
                        'pred_boxes': cur_boxes_tea,
                    })

        return ret_pred_dicts, ret_pred_dicts_tea

    def proposal_layer(self, batch_dict, nms_config):
        post_process_cfg = nms_config.POST_CONFIG
        post_center_limit_range = torch.tensor(self.point_cloud_range).cuda().float()

        pred_dicts = batch_dict['pred_dicts_stu']
        pred_dicts_before = batch_dict['pred_dicts_stu']
        batch_size = batch_dict['batch_size']

        # feature_number = batch_dict['batch_box_preds_tea_densehead'].shape[-1]
        batch_box_preds_tea = batch_dict['batch_box_preds_tea_densehead'].view(batch_size, 188, 188, 6, -1)
        batch_cls_preds_tea = torch.sigmoid(batch_dict['batch_cls_preds_tea_densehead'].view(batch_size, 188, 188, 6, -1))
        batch_cls_preds_tea_index = batch_cls_preds_tea.max(-1)[0].max(-1,keepdim=True)[1].unsqueeze(-1).repeat(1,1,1,1,7)
        batch_box_preds_tea = batch_box_preds_tea.gather(dim=3, index = batch_cls_preds_tea_index).squeeze()


        # ret_dict = [{
        #     'pred_boxes': [],
        #     'pred_scores': [],
        #     'pred_labels': [],
        # } for k in range(batch_size)]

        for idx, pred_dict in enumerate(pred_dicts):

            if idx > 0:
                assert False

            batch_hm = pred_dict['hm'].sigmoid()
            # print(batch_box_preds_tea.shape)
            # exit()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in post_process_cfg.HEAD_ORDER else None

            final_pred_dicts = self.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=post_process_cfg.FEATURE_MAP_STRIDE,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )


            rois_reverse = batch_box_preds_tea.new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, 7))
            cls_select_reverse = batch_box_preds_tea.new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, pred_dict['hm'].shape[1]))
            mask = batch_box_preds_tea.new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, 1))
            for k, final_dict in enumerate(final_pred_dicts):
                # final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                # print(final_dict['pred_scores'].sort())
                # exit()
                selected, selected_scores = class_agnostic_nms_minor(
                    box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                    nms_config=nms_config.TRAIN,
                    score_thresh_minor=nms_config.TRAIN.SCORE_THRESH_MINOR,
                )

                # final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                # final_dict['pred_scores'] = selected_scores
                # final_dict['index'] = final_dict['index'][selected]
                # final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                # ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                # ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                # ret_dict[k]['index'].append(final_dict['index'])
                # ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])


                cur_box_preds_tea = batch_box_preds_tea[k]
                # cur_cls_preds_tea = batch_cls_preds_tea[k]
                select_index = final_dict['index'][selected]


                # print(cur_box_preds_tea.shape)
                with torch.no_grad():
                    rois_reverse[k, :len(selected), :] = cur_box_preds_tea[select_index//188, select_index%188, :]
                    mask[k, :len(selected), :] = 1.0
                # exit()
                cls_select_reverse[k, :len(selected), :] = pred_dicts_before[idx]['hm'][k, :, select_index//188, select_index%188].t()

        batch_dict['rois_reverse'] = rois_reverse
        batch_dict['cls_preds_selected_reverse'] = cls_select_reverse
        batch_dict['mask_use_item'] = mask

        return batch_dict


    def proposal_layer_cp_tkd(self, batch_dict, nms_config):
        post_process_cfg = nms_config.POST_CONFIG
        post_center_limit_range = torch.tensor(self.point_cloud_range).cuda().float()

        pred_dicts = batch_dict['pred_dicts_stu']
        pred_dicts_before = batch_dict['pred_dicts_stu']
        batch_size = batch_dict['batch_size']

        # feature_number = batch_dict['batch_box_preds_tea_densehead'].shape[-1]
        pred_dicts_tea = batch_dict['pred_dicts_tea']

        # batch_box_preds_tea = batch_dict['batch_box_preds_tea_densehead'].view(batch_size, 200, 176, 6, -1)
        # batch_cls_preds_tea = torch.sigmoid(batch_dict['batch_cls_preds_tea_densehead'].view(batch_size, 200, 176, 6, -1))
        # batch_cls_preds_tea_index = batch_cls_preds_tea.max(-1)[0].max(-1,keepdim=True)[1].unsqueeze(-1).repeat(1,1,1,1,7)
        # batch_box_preds_tea = batch_box_preds_tea.gather(dim=3, index = batch_cls_preds_tea_index).squeeze()


        # ret_dict = [{
        #     'pred_boxes': [],
        #     'pred_scores': [],
        #     'pred_labels': [],
        # } for k in range(batch_size)]

        for idx, pred_dict in enumerate(pred_dicts):

            if idx > 0:
                assert False


            # TODO: STUDENT DECODE
            # batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in post_process_cfg.HEAD_ORDER else None

            # TODO: TEACHER DECODE
            batch_hm_tea = pred_dicts_tea[idx]['hm'].sigmoid()
            batch_center_tea = pred_dicts_tea[idx]['center']
            batch_center_z_tea = pred_dicts_tea[idx]['center_z']
            batch_dim_tea = pred_dicts_tea[idx]['dim'].exp()
            batch_rot_cos_tea = pred_dicts_tea[idx]['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin_tea = pred_dicts_tea[idx]['rot'][:, 1].unsqueeze(dim=1)
            batch_vel_tea = pred_dicts_tea[idx]['vel'] if 'vel' in post_process_cfg.HEAD_ORDER else None

            # print('f1', torch.isnan(batch_center_tea).any().any or torch.isinf(batch_center_tea).any())
            # print('f2', torch.isnan(batch_center_z_tea).any() or torch.isinf(batch_center_z_tea).any())
            # print('f3', torch.isnan(batch_dim_tea).any() or torch.isinf(batch_dim_tea).any())
            # print('f4', torch.isnan(batch_rot_cos_tea).any() or torch.isinf(batch_rot_cos_tea).any())
            # print('f5', torch.isnan(batch_rot_sin_tea).any() or torch.isinf(batch_rot_sin_tea).any())

            final_pred_dicts, final_pred_dicts_tea = self.decode_bbox_from_heatmap_tea(
                heatmap=batch_hm_tea, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                rot_cos_tea=batch_rot_cos_tea, rot_sin_tea=batch_rot_sin_tea,
                center_tea=batch_center_tea, center_z_tea=batch_center_z_tea, dim_tea=batch_dim_tea, vel_tea=batch_vel_tea,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=post_process_cfg.FEATURE_MAP_STRIDE,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )
            # print('zz3', torch.isnan(final_pred_dicts_tea[0]['pred_boxes']).any() or torch.isinf(final_pred_dicts_tea[0]['pred_boxes']).any())

            # for bs in range(batch_size):
            #     print('a', final_pred_dicts[bs]['pred_boxes'].shape)
            #     print('b', final_pred_dicts_tea[bs]['pred_boxes'].shape)
            # exit()
            # print(batch_hm_tea.shape)  # bs, 3, 188, 188


            rois_reverse = final_pred_dicts_tea[0]['pred_boxes'].new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, final_pred_dicts_tea[0]['pred_boxes'].shape[-1]))
            cls_select_reverse = pred_dicts_before[idx]['hm'].new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, pred_dict['hm'].shape[1]))
            mask = pred_dicts_before[idx]['hm'].new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, 1))
            # select_index_to_get_tf_mask = batch_hm.new_ones(batch_hm.shape) * -1
            width = batch_hm_tea.shape[-1]
            for k, final_dict in enumerate(final_pred_dicts_tea):
                if nms_config.TRAIN.get('SCORE_THRESH_MINOR'):
                    selected, selected_scores = class_agnostic_nms_minor(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=nms_config.TRAIN,
                        score_thresh_minor=nms_config.TRAIN.SCORE_THRESH_MINOR,
                    )
                else:
                    selected, selected_scores = class_agnostic_nms_minor(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=nms_config.TRAIN,
                    )

                # print('a',selected)
                with torch.no_grad():
                    # select_index_to_get_tf_mask[k, :, select_index//width, select_index%width] = 1.0
                    mask[k, :len(selected), :] = 1.0
                    rois_reverse[k, :len(selected), :] = final_pred_dicts_tea[k]['pred_boxes'][selected]

                select_index = final_dict['index'][selected]
                cls_select_reverse[k, :len(selected), :] = pred_dicts_before[idx]['hm'][k, :, select_index//width, select_index%width].t()

        batch_dict['rois_reverse'] = rois_reverse
        # print('zz1', torch.isnan(rois_reverse).any() or torch.isinf(rois_reverse).any())
        batch_dict['cls_preds_selected_reverse'] = cls_select_reverse
        batch_dict['mask_use_item'] = mask

        return batch_dict

    def proposal_layer_cp(self, batch_dict, nms_config):
        post_process_cfg = nms_config.POST_CONFIG
        post_center_limit_range = torch.tensor(self.point_cloud_range).cuda().float()

        pred_dicts = batch_dict['pred_dicts_stu']
        pred_dicts_before = batch_dict['pred_dicts_stu']
        batch_size = batch_dict['batch_size']

        # feature_number = batch_dict['batch_box_preds_tea_densehead'].shape[-1]
        pred_dicts_tea = batch_dict['pred_dicts_tea']

        # batch_box_preds_tea = batch_dict['batch_box_preds_tea_densehead'].view(batch_size, 200, 176, 6, -1)
        # batch_cls_preds_tea = torch.sigmoid(batch_dict['batch_cls_preds_tea_densehead'].view(batch_size, 200, 176, 6, -1))
        # batch_cls_preds_tea_index = batch_cls_preds_tea.max(-1)[0].max(-1,keepdim=True)[1].unsqueeze(-1).repeat(1,1,1,1,7)
        # batch_box_preds_tea = batch_box_preds_tea.gather(dim=3, index = batch_cls_preds_tea_index).squeeze()


        # ret_dict = [{
        #     'pred_boxes': [],
        #     'pred_scores': [],
        #     'pred_labels': [],
        # } for k in range(batch_size)]

        for idx, pred_dict in enumerate(pred_dicts):

            if idx > 0:
                assert False


            # TODO: STUDENT DECODE
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in post_process_cfg.HEAD_ORDER else None

            # TODO: TEACHER DECODE
            # batch_hm_tea = pred_dicts_tea[idx]['hm'].sigmoid()
            batch_center_tea = pred_dicts_tea[idx]['center']
            batch_center_z_tea = pred_dicts_tea[idx]['center_z']
            batch_dim_tea = pred_dicts_tea[idx]['dim'].exp()
            batch_rot_cos_tea = pred_dicts_tea[idx]['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin_tea = pred_dicts_tea[idx]['rot'][:, 1].unsqueeze(dim=1)
            batch_vel_tea = pred_dicts_tea[idx]['vel'] if 'vel' in post_process_cfg.HEAD_ORDER else None

            # print('f1', torch.isnan(batch_center_tea).any().any or torch.isinf(batch_center_tea).any())
            # print('f2', torch.isnan(batch_center_z_tea).any() or torch.isinf(batch_center_z_tea).any())
            # print('f3', torch.isnan(batch_dim_tea).any() or torch.isinf(batch_dim_tea).any())
            # print('f4', torch.isnan(batch_rot_cos_tea).any() or torch.isinf(batch_rot_cos_tea).any())
            # print('f5', torch.isnan(batch_rot_sin_tea).any() or torch.isinf(batch_rot_sin_tea).any())

            final_pred_dicts, final_pred_dicts_tea = self.decode_bbox_from_heatmap_tea(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                rot_cos_tea=batch_rot_cos_tea, rot_sin_tea=batch_rot_sin_tea,
                center_tea=batch_center_tea, center_z_tea=batch_center_z_tea, dim_tea=batch_dim_tea, vel_tea=batch_vel_tea,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=post_process_cfg.FEATURE_MAP_STRIDE,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )
            # print('zz3', torch.isnan(final_pred_dicts_tea[0]['pred_boxes']).any() or torch.isinf(final_pred_dicts_tea[0]['pred_boxes']).any())

            # for bs in range(batch_size):
            #     print('a', final_pred_dicts[bs]['pred_boxes'].shape)
            #     print('b', final_pred_dicts_tea[bs]['pred_boxes'].shape)
            # exit()
            # print(batch_hm_tea.shape)  # bs, 3, 188, 188


            rois_reverse = final_pred_dicts_tea[0]['pred_boxes'].new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, final_pred_dicts_tea[0]['pred_boxes'].shape[-1]))
            cls_select_reverse = pred_dicts_before[idx]['hm'].new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, pred_dict['hm'].shape[1]))
            mask = pred_dicts_before[idx]['hm'].new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, 1))
            # select_index_to_get_tf_mask = batch_hm.new_ones(batch_hm.shape) * -1
            width = batch_hm.shape[-1]
            for k, final_dict in enumerate(final_pred_dicts):
                if nms_config.TRAIN.get('SCORE_THRESH_MINOR'):
                    selected, selected_scores = class_agnostic_nms_minor(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=nms_config.TRAIN,
                        score_thresh_minor=nms_config.TRAIN.SCORE_THRESH_MINOR,
                    )
                else:
                    selected, selected_scores = class_agnostic_nms_minor(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=nms_config.TRAIN,
                    )

                # print('b', selected)
                with torch.no_grad():
                    # select_index_to_get_tf_mask[k, :, select_index//width, select_index%width] = 1.0
                    mask[k, :len(selected), :] = 1.0
                    rois_reverse[k, :len(selected), :] = final_pred_dicts_tea[k]['pred_boxes'][selected]

                select_index = final_dict['index'][selected]
                cls_select_reverse[k, :len(selected), :] = pred_dicts_before[idx]['hm'][k, :, select_index//width, select_index%width].t()

        batch_dict['rois_reverse'] = rois_reverse
        # print('zz1', torch.isnan(rois_reverse).any() or torch.isinf(rois_reverse).any())
        batch_dict['cls_preds_selected_reverse'] = cls_select_reverse
        batch_dict['mask_use_item'] = mask

        return batch_dict

    def forward(self, batch_dict):

        # for u,v in batch_dict.items():
        #     print(u)
        #
        #
        # stu_preds_dict = batch_dict['final_box_dicts_stu_kd']
        # print(len(stu_preds_dict[0]))
        # print(batch_dict['batch_cls_preds_tea_densehead'].shape)

        # exit()
        if batch_dict.get('batch_cls_preds_tea_densehead'):
            _ = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.NMS_CONFIG
            )
        else:
            if self.model_cfg.MODE == 'Dual':
                _ = self.proposal_layer_cp_tkd(
                    batch_dict, nms_config=self.model_cfg.NMS_CONFIG
                )

                with torch.no_grad():
                    batch_dict['re_run_tea_flag'] = True
                    # roi_tmp = batch_dict['rois'].detach()
                    fea_num = batch_dict['rois'].shape[-1]
                    batch_dict['rois'] = batch_dict['rois_reverse'].detach().view(batch_dict['batch_size'], -1, fea_num)

                    if batch_dict['teacher_model'].__class__.__name__ == 'PVRCNNPlusPlus':
                        # print('a')
                        # exit()
                        batch_dict['point_coords'] = batch_dict['point_coords_tea']
                        batch_dict['point_feature'] = batch_dict['point_features_tea']
                        batch_dict['point_cls_scores'] = batch_dict['point_cls_scores_tea']

                    # print('zz2',torch.isnan(batch_dict['rois']).any() or torch.isinf(batch_dict['rois']).any())

                    teacher_model = batch_dict['teacher_model']
                    teacher_model.eval()
                    teacher_model.roi_head(batch_dict)
                    # batch_dict['rois'] = roi_tmp.detach()
                    batch_dict['re_run_tea_flag'] = None

                    # select_obj_mask = batch_dict['mask_use_item'].view(-1, 1)
                    batch_dict['cls_preds_selected_reverse_tea'] = batch_dict['rcnn_cls_stu_to_tea']
                    # print(batch_dict['cls_preds_selected_reverse_tea'].sum())

                cls_pred_selected_reverse_max = batch_dict['cls_preds_selected_reverse'].max(dim=-1)[0].view(-1, 1)
                batch_dict['cls_preds_selected_reverse_stu'] = cls_pred_selected_reverse_max

                ask_use_item_tmp = batch_dict['mask_use_item'].view(-1, 1)



                _ = self.proposal_layer_cp(
                    batch_dict, nms_config=self.model_cfg.NMS_CONFIG
                )

                with torch.no_grad():
                    batch_dict['re_run_tea_flag'] = True
                    # roi_tmp = batch_dict['rois'].detach()
                    fea_num = batch_dict['rois'].shape[-1]
                    batch_dict['rois'] = batch_dict['rois_reverse'].detach().view(batch_dict['batch_size'], -1, fea_num)

                    if batch_dict['teacher_model'].__class__.__name__ == 'PVRCNNPlusPlus':
                        # print('a')
                        # exit()
                        batch_dict['point_coords'] = batch_dict['point_coords_tea']
                        batch_dict['point_feature'] = batch_dict['point_features_tea']
                        batch_dict['point_cls_scores'] = batch_dict['point_cls_scores_tea']

                    # print('zz2',torch.isnan(batch_dict['rois']).any() or torch.isinf(batch_dict['rois']).any())

                    teacher_model = batch_dict['teacher_model']
                    teacher_model.eval()
                    teacher_model.roi_head(batch_dict)
                    # batch_dict['rois'] = roi_tmp.detach()
                    batch_dict['re_run_tea_flag'] = None

                    # select_obj_mask = batch_dict['mask_use_item'].view(-1, 1)
                    batch_dict['cls_preds_selected_reverse_tea'] = torch.cat((batch_dict['cls_preds_selected_reverse_tea'],
                                                                              batch_dict['rcnn_cls_stu_to_tea']),
                                                                             dim=0)
                    # print(batch_dict['cls_preds_selected_reverse_tea'].sum())

                cls_pred_selected_reverse_max = batch_dict['cls_preds_selected_reverse'].max(dim=-1)[0].view(-1, 1)
                batch_dict['cls_preds_selected_reverse_stu'] = torch.cat((batch_dict['cls_preds_selected_reverse_stu'],
                                                                          cls_pred_selected_reverse_max),
                                                                         dim=0)
                batch_dict['mask_use_item'] = torch.cat((ask_use_item_tmp, batch_dict['mask_use_item'].view(-1, 1)), dim=1)


            else:

                if self.model_cfg.MODE == 'Tea2Stu':
                    _ = self.proposal_layer_cp_tkd(
                        batch_dict, nms_config=self.model_cfg.NMS_CONFIG
                    )
                else:
                    _ = self.proposal_layer_cp(
                        batch_dict, nms_config=self.model_cfg.NMS_CONFIG
                    )

                with torch.no_grad():
                    batch_dict['re_run_tea_flag'] = True
                    # roi_tmp = batch_dict['rois'].detach()
                    fea_num = batch_dict['rois'].shape[-1]
                    batch_dict['rois'] = batch_dict['rois_reverse'].detach().view(batch_dict['batch_size'], -1, fea_num)

                    if batch_dict['teacher_model'].__class__.__name__ == 'PVRCNNPlusPlus':
                        # print('a')
                        # exit()
                        batch_dict['point_coords'] = batch_dict['point_coords_tea']
                        batch_dict['point_feature'] = batch_dict['point_features_tea']
                        batch_dict['point_cls_scores'] = batch_dict['point_cls_scores_tea']


                    # print('zz2',torch.isnan(batch_dict['rois']).any() or torch.isinf(batch_dict['rois']).any())

                    teacher_model = batch_dict['teacher_model']
                    teacher_model.eval()
                    teacher_model.roi_head(batch_dict)
                    # batch_dict['rois'] = roi_tmp.detach()
                    batch_dict['re_run_tea_flag'] = None

                    select_obj_mask = batch_dict['mask_use_item'].view(-1, 1)
                    # for i in range(batch_dict['batch_size']):
                    #     print('b', batch_dict['mask_use_item'][i].sum())
                    batch_dict['cls_preds_selected_reverse_tea'] = batch_dict['rcnn_cls_stu_to_tea'] * select_obj_mask
                    # print(batch_dict['cls_preds_selected_reverse_tea'].sum())


                cls_pred_selected_reverse_max = batch_dict['cls_preds_selected_reverse'].max(dim=-1)[0].view(-1, 1)
                batch_dict['cls_preds_selected_reverse_stu'] = cls_pred_selected_reverse_max * select_obj_mask


                if self.model_cfg.NMS_CONFIG.get('USE_REG_MASK', False):
                    batch_dict['box_preds_selected_reverse_tea'] = batch_dict['batch_box_preds_tea'].view(-1, batch_dict['batch_box_preds_tea'].shape[-1]) * batch_dict['mask_reg'].view(-1, 1)
                    batch_dict['box_preds_selected_reverse_stu'] = batch_dict['raw_box_stu'].view(-1, batch_dict['batch_box_preds_tea'].shape[-1]) * batch_dict['mask_reg'].view(-1, 1)



        return batch_dict
