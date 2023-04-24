import os
import cv2
import math
import numpy as np
import argparse
from utils import *

def get_pred_room_corners(room_lines):
    room_corners = []
    for lines in room_lines:
        corners = []
        for c in lines:
            if c[0] not in corners:
                corners.append(c[0])
        room_corners.append(np.array(corners))
    return room_corners

def get_gt_room_corners(room_data):
    room_corners = []
    for item in room_data:
        corners = []
        for c in item['room_corners']:
            if c not in corners:
                corners.append(c)
        room_corners.append(corners)
    return room_corners

def get_gt_angle(room_corners):
    room_angles = []
    for i, corners in enumerate(room_corners):
        corners_num = len(corners)
        angles = []
        for j, point in enumerate(corners):
            id = j
            pre_id = j - 1 if j > 0 else corners_num - 1
            next_id = (j + 1) % corners_num
            l0 = np.array(corners[pre_id]) - np.array(corners[id])
            l1 = np.array(corners[next_id]) - np.array(corners[id])
            angle = math.acos(l0.dot(l1) / (np.linalg.norm(l0) * np.linalg.norm(l1))) / 3.1415926 * 180
            angles.append(angle)
        room_angles.append(angles)
    return room_angles

def get_corner_edge_map(edges):
    all_corners = []
    corner_to_line_id = {}
    line_to_corner_id = {}
    for edge_id, edge in enumerate(edges):
        line_to_corner_id[edge_id] = []
        for i in range(2):
            if edge[i] not in all_corners:
                corner_to_line_id[len(all_corners)] = [edge_id]
                line_to_corner_id[edge_id].append(len(all_corners))
                all_corners.append(edge[i])
            else:
                pos = all_corners.index(edge[i])
                corner_to_line_id[pos].append(edge_id)
                line_to_corner_id[edge_id].append(pos)
    return corner_to_line_id, line_to_corner_id, all_corners

def get_all_pred_rooms_mask(all_room_edges, img_size):
    all_room_masks = []
    for room_edges in all_room_edges:
        room_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        room_corners = []
        for edges in room_edges:
            room_corners.append(edges[0])
            room_corners.append(edges[1])
        cv2.drawContours(room_mask, [np.array(room_corners)], -1, 255, -1)
        all_room_masks.append(room_mask)
    return all_room_masks

def evaluate_rooms(room_pred, room_gt, IOU_threshold=0.5):
    num_room_gt = len(room_gt)
    num_room_pred = len(room_pred)
    num_room_matches = 0
    pred_room_visit = [False] * num_room_pred
    room_match_index = [-1] * num_room_pred
    for gt_id, gt_room in enumerate(room_gt):
        for pred_id, pred_room in enumerate(room_pred):
            if pred_room_visit[pred_id]:
                continue
            IOU = float(np.logical_and(pred_room, gt_room['mask']).sum()) / np.logical_or(pred_room, gt_room['mask']).sum()
            if IOU > IOU_threshold:
                pred_room_visit[pred_id] = True
                room_match_index[pred_id] = gt_id
                num_room_matches += 1
                break
    return (num_room_matches, num_room_gt, num_room_pred, room_match_index)

def evaluate_corners(room_corner_pred, room_corners_gt, room_match_index, img_size, distance_threshold=10):
    num_gt = sum([len(corners) for corners in room_corners_gt])
    num_pred = sum([len(corners) for corners in room_corner_pred])
    num_matches = 0
    scale = int(img_size / 256)
    distance_threshold *= scale
    for pred_id, pred_corners in enumerate(room_corner_pred):
        visit = [0] * len(pred_corners)
        gt_id = room_match_index[pred_id]
        if gt_id == -1:
            continue
        gt_corners = room_corners_gt[gt_id]
        visit_gt = [0] * len(gt_corners)
        diff = np.linalg.norm(np.expand_dims(gt_corners, 1) - np.expand_dims(pred_corners, 0), axis=2)
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                if diff[i][j] < distance_threshold:
                    if visit[j] == 1:
                        continue
                    if visit_gt[i] == 1:
                        continue
                    visit[j] = 1
                    visit_gt[i] = 1
                    num_matches += 1
    return np.array([num_matches, num_gt, num_pred])

def evaluate_angle(room_corner_pred, room_corners_gt, pred_angle, gt_angle, room_match_index, img_size, distance_threshold=10, angle_threshold=5):
    num_gt = sum([len(corners) for corners in room_corners_gt])
    num_pred = sum([len(corners) for corners in room_corner_pred])
    num_matches = 0
    scale = int(img_size / 256)
    distance_threshold *= scale
    for pred_id, pred_corners in enumerate(room_corner_pred):
        visit_pred = [0] * len(pred_corners)
        gt_id = room_match_index[pred_id]
        if gt_id == -1:
            continue
        gt_corners = room_corners_gt[gt_id]
        visit_gt = [0] * len(gt_corners)
        diff = np.linalg.norm(np.expand_dims(gt_corners, 1) - np.expand_dims(pred_corners, 0), axis=2)
        angle_diff = np.expand_dims(gt_angle[gt_id], 1)- np.expand_dims(pred_angle[pred_id], 0)
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                if diff[i][j] < distance_threshold and abs(angle_diff[i][j]) < angle_threshold:
                    if visit_pred[j] == 1:
                        continue
                    if visit_gt[i] == 1:
                        continue
                    visit_pred[j] = 1
                    visit_gt[i] = 1
                    num_matches += 1
    return np.array([num_matches, num_gt, num_pred])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_result", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    avg_corner_precision = 0.0
    avg_corner_recall = 0.0
    avg_angle_precision = 0.0
    avg_angle_recall = 0.0
    avg_room_precision = 0.0
    avg_room_recall = 0.0
    count = 0

    for scene_id, file_name in enumerate(sorted(os.listdir(args.predict_result))):
        if not file_name.endswith('npz'):
            continue
        if '_layer' in file_name:
            scene_name = file_name.split('_layer')[0]
            layer_name = file_name.split(scene_name + '_')[1].split('.')[0]
        else:
            scene_name = file_name.split('.npz')[0]
            layer_name = "layer-00"
        if args.dataset == 'cyberverse':
            gt_file = 'data/%s/evaluation_groundtruth/%s'%(args.dataset, file_name.replace('npz', 'npy'))
        else:
            gt_file = 'data/%s/evaluation_groundtruth/%s.npy'%(args.dataset, scene_name)
        arrangement_path = 'data/%s/arrangement_graph/%s/%s/arr.obj'%(args.dataset, scene_name, layer_name)
        arrobj = load_tri_obj(arrangement_path)
        gt_room_data = np.load(gt_file, allow_pickle=True).tolist()
        scene_box = gt_room_data['scene_box']
        img_size = gt_room_data['img_size']
        print("evaluate ", scene_name, layer_name)

        #### get arrnet room info ####
        pred_result = np.load(os.path.join(args.predict_result, file_name))
        face_label_pred = pred_result['predFaceLabel']
        wall_label_pred = pred_result['predWallLabel']
        arrnet_lines, arrnet_room_lines, arrnet_corners = get_room_boundarys_for_evaluate(arrobj, face_label_pred, wall_label_pred, scene_box, img_size)

        #### evaluate rooms ####
        pred_room_masks = get_all_pred_rooms_mask(arrnet_room_lines, img_size)
        match_rooms, gt_rooms, pred_rooms, room_match_index = evaluate_rooms(pred_room_masks, gt_room_data['room_instances_annot'], 0.5)
        avg_room_precision += min(match_rooms / pred_rooms, 1.0)
        avg_room_recall += match_rooms / gt_rooms

        #### evaluate corners ####
        pred_room_corners = get_pred_room_corners(arrnet_room_lines)
        gt_room_corners = get_gt_room_corners(gt_room_data['room_instances_annot'])
        match_corners, gt_corners, pred_corners = \
            evaluate_corners(pred_room_corners, gt_room_corners, room_match_index, img_size)
        avg_corner_precision += min(match_corners / pred_corners, 1.0)
        avg_corner_recall += match_corners / gt_corners

        #### evaluate angle ####
        gt_angle = get_gt_angle(gt_room_corners)
        pred_angle = get_gt_angle(pred_room_corners)
        match_angles, gt_angles, pred_angles = \
            evaluate_angle(pred_room_corners, gt_room_corners, pred_angle, gt_angle, room_match_index, img_size)
        avg_angle_precision += min(match_angles / pred_angles, 1.0)
        avg_angle_recall += match_angles / gt_angles

        count += 1

    print('avg Room precision and recall', avg_room_precision / count, avg_room_recall / count)
    print('avg Corner precision and recall', avg_corner_precision / count, avg_corner_recall / count)
    print('avg Angle precision and recall', avg_angle_precision / count, avg_angle_recall / count)
