import numpy as np
import math
import random

def load_tri_obj(tri_obj_file, with_inst=0, with_height=False):
    token = -10000
    lines = [l.strip() for l in open(tri_obj_file)]
    coords = []
    triangles = {token:[]}
    for line in lines:
        words = [w for w in line.split(' ') if w != '']
        if len(words) == 0:
            continue
        if words[0] == '#':
            token = int(words[1])
            triangles[token] = []
        if words[0] == 'v':
            if with_height:
                coords.append([float(words[1]), float(words[2]), float(words[3])])
            else:
                coords.append([float(words[1]), float(words[2])])
        elif words[0] == 'f':
            triangles[token].append([int(words[1]) - 1, int(words[2]) - 1, int(words[3]) - 1])
    for tr in triangles:
        triangles[tr] = np.array(triangles[tr], dtype='int32')
    if with_inst == 0:
        triangles = triangles[-10000]
    return {'V': np.array(coords, dtype='float64'),
            'F': triangles}

def convert_tri_to_lines(faces):
    pts = faces['V']
    segments = []
    tris = faces['F']
    for tri in tris:
        segments.append([tri[0], tri[1]])
        segments.append([tri[1], tri[2]])
        segments.append([tri[2], tri[0]])
    segments = np.array(segments, dtype='int32')
    return pts, segments

def get_twin_lines(segments):
    twin_list = [-1] * len(segments)
    twin_line_dict = {}
    for i, l in enumerate(np.sort(segments)):
        tl = tuple(l)
        if tl not in twin_line_dict:
            twin_line_dict[tl] = [i]
        else:
            twin_line_dict[tl].append(i)
    for i, l in enumerate(np.sort(segments)):
        tl = tuple(l)
        if tl in twin_line_dict:
            tmp_array = np.array(twin_line_dict[tl])
            if len(tmp_array[np.where(tmp_array != i)]) > 0:
                twin_list[i] = tmp_array[np.where(tmp_array != i)][0]
    return twin_list

def get_line_face_connections(arrobj, floor_label):
    tris = arrobj['F']
    line_count = 0
    line_to_face = {}
    face_to_line = {}
    for i, tri in enumerate(tris):
        if floor_label[i] == 0:
            line_count += 3
            continue
        face_to_line[i] = []
        for j in range(3):
            face_to_line[i].append(line_count)
            line_to_face[line_count] = i
            line_count += 1
    return face_to_line, line_to_face

def get_pixel_coord(pt, scene_box, im_size):
    u = np.floor((pt[0] - scene_box[0]) * 1.0 / scene_box[2] * im_size).astype(np.int32)
    v = np.floor((pt[1] - scene_box[1]) * 1.0 / scene_box[3] * im_size).astype(np.int32)
    return (u, v)

def convert_to_pixel_coords(pts, segments, room_boundarys, room_corners, scene_box, im_size):
    all_room_corners_in_order = []
    all_room_lines = []
    all_corners = []
    point_dict = {}
    for room_id, ls in enumerate(room_boundarys):
        if len(room_corners[room_id]) == 0:
            continue
        if len(ls) < 4:
            continue
        point_angle = {}
        for l in ls:
            for i in range(2):
                id = segments[l][i]
                if id in point_angle:
                    point_angle[id].append(segments[l][1 - i])
                else:
                    point_angle[id] = [segments[l][1 - i]]
        corners_in_order = []
        room_lines = []
        p_start = list(room_corners[room_id].keys())[0]
        p_start = sorted(room_corners[room_id].keys(), key=lambda x:pts[x][0])[0]
        p = -1
        p_prev = -1
        p_prev_corner = p_start
        while (p != p_start):
            if p == -1:
                p_prev = p_start
                p = point_angle[p_start][0]
            else:
                p_tmp = p_prev
                p_prev = p
                p = point_angle[p][1 - point_angle[p].index(p_tmp)]
            if p in room_corners[room_id]:
                line_p0 = get_pixel_coord(pts[p_prev_corner], scene_box, im_size)
                line_p1 = get_pixel_coord(pts[p], scene_box, im_size)
                room_lines.append((line_p0, line_p1))
                corners_in_order.append(p)
                all_corners.append(line_p0)
                all_corners.append(line_p1)
                if p_prev_corner not in point_dict:
                    point_dict[p_prev_corner] = line_p0
                if p not in point_dict:
                    point_dict[p] = line_p1
                p_prev_corner = p
        all_room_lines.append(room_lines)
        all_room_corners_in_order.append(corners_in_order)
    return all_room_lines, all_room_corners_in_order, point_dict, all_corners

def get_room_corners(room_boundarys, twin_lines, pts, segments):
    room_corners = []
    for ls in room_boundarys:
        corners = {}
        point_angle = {}
        for l in ls:
            for i in range(2):
                id = segments[l][i]
                if id in point_angle:
                    point_angle[id].append(segments[l][1 - i])
                else:
                    point_angle[id] = [segments[l][1 - i]]
        for p, another_p in point_angle.items():
            if len(another_p) != 2:
                continue
            l0 = pts[another_p[0]] - pts[p]
            l1 = pts[another_p[1]] - pts[p]
            len0 = np.linalg.norm(l0)
            len1 = np.linalg.norm(l1)
            if len0 == 0 or len1 == 0:
                continue
            angle = abs(np.dot(l0, l1) / (len0 * len1))
            if angle > math.cos(10 / 180.0 * 3.141592654):
                continue
            corners[p] = another_p
        room_corners.append(corners)
    return room_corners

def DFS(f, face_to_line, line_to_face, wall_label, twin_lines, visited, pts, segments):
    boundary_lines = []
    stack = []
    stack.append(f)
    visited.append(f)
    cur_region_face = []
    while (len(stack) > 0):
        fv = stack.pop()
        cur_region_face.append(fv)
        lines = face_to_line[fv]
        lines += [twin_lines[i] for i in lines]
        new_lines = list(set(lines))
        while -1 in new_lines:
            new_lines.remove(-1)
        for l in new_lines:
            if wall_label[l] != 0:
                if twin_lines[l] not in boundary_lines:
                    boundary_lines.append(l)
                continue
            if l not in line_to_face:
                continue
            fvv = line_to_face[l]
            if fvv not in visited:
                stack.append(fvv)
                visited.append(fvv)
    return cur_region_face, list(set(boundary_lines))

def get_origin_room_boundarys(arrobj, floor_label, wall_label, twin_lines):
    pts, segments = convert_tri_to_lines(arrobj)
    face_to_line, line_to_face = get_line_face_connections(arrobj, floor_label)
    group_boundary_lines = []
    visit = []
    for i, (f, ls) in enumerate(face_to_line.items()):
        if f not in visit:
            cur_region_faces, boundary_lines = DFS(f, face_to_line, line_to_face, wall_label, twin_lines, visit, pts, segments)
            if len(boundary_lines) > 0:
                group_boundary_lines.append(boundary_lines)
    return group_boundary_lines

def filter_boundarys(room_boundarys, pts, segments):
    filter_room_boundarys = []
    for i, ls in enumerate(room_boundarys):
        point_degree = {}
        for l in ls:
            for j in range(2):
                id = segments[l][j]
                if id in point_degree:
                    point_degree[id].append(l)
                else:
                    point_degree[id] = [l]
        valid_loop = True
        for k, v in point_degree.items():
            if len(v) < 2:
                print("Error loop!")
                valid_loop = False
                break
        if valid_loop:
            filter_room_boundarys.append(ls)
    return filter_room_boundarys

def get_room_boundarys(arrobj, pts, segments, floor_label, wall_label):
    twin_lines = get_twin_lines(segments)
    origin_room_boundary_lines = get_origin_room_boundarys(arrobj, floor_label, wall_label, twin_lines)
    final_room_boundarys = filter_boundarys(origin_room_boundary_lines, pts, segments)
    room_corners = get_room_corners(final_room_boundarys, twin_lines, pts, segments)
    return room_corners, final_room_boundarys

def get_room_boundarys_for_evaluate(arrobj, floor_label, wall_label, scene_box, im_size):
    pts, segments = convert_tri_to_lines(arrobj)
    room_corners, final_room_boundarys = get_room_boundarys(arrobj, pts, segments, floor_label, wall_label)
    all_room_lines, _, _, all_corners = convert_to_pixel_coords(pts, segments, final_room_boundarys, room_corners, scene_box, im_size)
    all_lines = [i for item in all_room_lines for i in item]
    return all_lines, all_room_lines, list(set(all_corners))
