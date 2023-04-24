import os
import glob
import dgl
import random
from ctypes import *
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class GraphDataset(Dataset):
    def __init__(self, root, usage='train', scale=1.0, dataset='floorsp'):
        self.root = root
        self.usage = usage
        self.scale = scale
        self.dataset = dataset
        if dataset == 'floorsp' or dataset == 'structured3d':
            self.filePaths = glob.glob(root + '/*/*/data.npz')
            self.filePaths.sort()
            if usage == 'train':
                lines = [l.strip() for l in open('data/%s/train_scenes.txt'%(dataset))]
            elif usage == 'val':
                lines = [l.strip() for l in open('data/%s/val_scenes.txt' % (dataset))]
            top = 0
            for i in range(len(self.filePaths)):
                token_id = -3
                f = self.filePaths[i].split('/')[token_id]
                if f in lines:
                    self.filePaths[top] = self.filePaths[i]
                    top += 1
            self.filePaths = self.filePaths[:top]
        elif dataset == 'cyberverse':
            if usage == 'train':
                self.filePaths = [l.strip() for l in open('data/cyberverse/train_scenes.txt')]
            else:
                self.filePaths = [l.strip() for l in open('data/cyberverse/val_scenes.txt')]
        else:
            print("Unsupported datset name.")
        self.data = [np.load(f, allow_pickle=True) for f in self.filePaths]
        self.idx = [i for i in range(0, len(self.data), 1)]
        self.data_len = len(self.idx)

    def CreateGraph(self, index, r, t1, t2):
        info = self.data[index]
        V = info['V']
        F = info['F']
        hb = info['FBound']
        hf = info['FFeat']

        V = V * self.scale
        F = np.concatenate([F, np.array([[0, 0, 0]])], axis=0)
        hb = np.concatenate([hb, np.array([0])], axis=0)
        hf = np.concatenate([hf, np.array([0])], axis=0)
        hc = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
        Segpair = info['ExtendHalfedges']
        if self.usage == 'train':
            imgX = hc[:, 0] * np.cos(r) - hc[:, 1] * np.sin(r) + t1
            imgY = hc[:, 0] * np.sin(r) + hc[:, 1] * np.cos(r) + t2
            hc[:, 0] = imgX
            hc[:, 1] = imgY
        diff0 = V[F[:, 1]] - V[F[:, 0]]
        diff1 = V[F[:, 2]] - V[F[:, 0]]
        norm = np.abs(np.cross(diff0, diff1, axis=1))
        edge = info['E2HE']
        edge[edge < 0] = (F.shape[0] - 1) * 3
        edgeId = np.array([i for i in range(edge.shape[0])])
        dedge2edge = np.zeros((F.shape[0] * 3), dtype='int32')
        dedge2edge[edge[:, 0]] = edgeId
        dedge2edge[edge[:, 1]] = edgeId

        edgeLink = np.unique(dedge2edge[Segpair], axis=0)
        u = np.concatenate([edgeLink[:, 0], edgeLink[:, 1] + edge.shape[0]], axis=0)
        v = np.concatenate([edgeLink[:, 1], edgeLink[:, 0] + edge.shape[0]], axis=0)
        ge = dgl.graph((u, v), num_nodes=edge.shape[0] * 2)

        edge1 = np.zeros((edge.shape[0] * 2, 2), dtype='int32')
        edge1[:edge.shape[0]] = edge
        edge1[edge.shape[0]:, 0] = edge[:, 1]
        edge1[edge.shape[0]:, 1] = edge[:, 0]

        he = edge
        edge = torch.from_numpy(edge1 // 3).long()
        g = dgl.graph((edge[:, 0], edge[:, 1]), num_nodes=F.shape[0])
        occ = info['HEFeat'][he[:, 0]]
        v0 = he[:, 0]
        v1 = he[:, 0] // 3 * 3 + (he[:, 0] + 1) % 3
        idx = F.reshape(-1)
        c0 = (V[idx[v0]] + V[idx[v1]]) * 0.5
        c0 = np.concatenate([c0, c0], axis=0)
        occ = np.concatenate([occ, occ], axis=0)

        g.edata['x'] = torch.from_numpy(np.concatenate([occ, c0], axis=1)).float()
        l = V[idx[v0]] - V[idx[v1]]
        l = np.concatenate([l, l], axis=0)
        l = np.linalg.norm(l, axis=1)
        g.edata['len'] = torch.from_numpy(l).float()
        g.ndata['x'] = torch.from_numpy(np.concatenate([norm.reshape(-1, 1), hb.reshape(-1, 1), hf.reshape(-1, 1), hc.reshape(-1, 2)], axis=1)).float()
        wall_label = info['wall_gt_label']
        wall_label = wall_label[he[:, 0]]
        wall_label = np.concatenate([wall_label, wall_label], axis=0)
        gt_label = info['floor_gt_label']
        gt_label = np.concatenate([gt_label, np.array([0])], axis=0)
        g.edata['y'] = torch.from_numpy(wall_label > 0).long()
        p0 = np.concatenate([V[idx[v0]], V[idx[v0]]], axis=0)
        p1 = np.concatenate([V[idx[v1]], V[idx[v1]]], axis=0)
        g.edata['p0'] = torch.from_numpy(p0)
        g.edata['p1'] = torch.from_numpy(p1)
        eidx = np.concatenate([idx[v0].reshape(-1, 1), idx[v1].reshape(-1, 1)], axis=1)
        eidx = np.concatenate([eidx, eidx], axis=0)
        g.edata['eIdx'] = torch.from_numpy(eidx).long()
        g.ndata['y'] = torch.from_numpy(gt_label).long()
        g.ndata['p0'] = torch.from_numpy(V[F[:, 0]])
        g.ndata['p1'] = torch.from_numpy(V[F[:, 1]])
        g.ndata['p2'] = torch.from_numpy(V[F[:, 2]])
        g.ndata['fIdx'] = torch.from_numpy(F[:, :]).long()
        g.ndata['area'] = torch.from_numpy(norm).float()
        return g, ge, torch.from_numpy(V).float()

    def __getitem__(self, index):
        r = np.random.rand(1) * np.pi * 2
        t1 = np.random.rand(1) - 0.5
        t2 = np.random.rand(1) - 0.5
        g, edgeLink, v = self.CreateGraph(index, r, t1, t2)
        if self.dataset == 'floorsp' or self.dataset == 'structured3d':
            token = self.filePaths[index].split('/')[-3]
        elif self.dataset == 'cyberverse':
            scene_token = self.filePaths[index].split('/')[-3]
            layer_token = self.filePaths[index].split('/')[-2]
            token = scene_token + '_' + layer_token
        return {'g': g, 'e': edgeLink, 'v': v, 'n': token}

    def __len__(self):
        return self.data_len

def Collate(b):
    return {'g': dgl.batch([b[i]['g'] for i in range(len(b))]),
            'e': dgl.batch([b[i]['e'] for i in range(len(b))]),
            'v': torch.cat([b[i]['v'] for i in range(len(b))]),
            'n': [b[i]['n'] for i in range(len(b))]}
