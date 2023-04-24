import os
import cv2
import argparse
import dgl
import numpy as np
from ctypes import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import GraphDataset, Collate
from model import *

parser = argparse.ArgumentParser(description='ArrangementNet')
parser.add_argument('--logdir', type=str, default='log')
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--ratioE', type=float, default=0.5)
parser.add_argument('--ratioF', type=float, default=1)
parser.add_argument('--increaseScore', type=float, default=0)
parser.add_argument('--dataset', type=str, default='floorsp')
args = parser.parse_args()

if args.dataset == 'floorsp':
    train_dataset = GraphDataset(usage='train', root='data/floorsp/arrangement_graph/', scale=1)
    val_dataset = GraphDataset(usage='val', root='data/floorsp/arrangement_graph/', scale=1)
elif args.dataset == 'structured3d':
    train_dataset = GraphDataset(usage='train', root='data/structured3d/arrangement_graph/', scale=1e-1, dataset='structured3d')
    val_dataset = GraphDataset(usage='val', root='data/structured3d/arrangement_graph/', scale=1e-1, dataset='structured3d')
elif args.dataset == 'cyberverse':
    train_dataset = GraphDataset(usage='train', root='data/cyberverse/arrangement_graph/', scale=1e-2, dataset='cyberverse')
    val_dataset = GraphDataset(usage='val', root='data/cyberverse/arrangement_graph/', scale=1e-2, dataset='cyberverse')
else:
    print("Unsupported dataset")
    exit(0)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=Collate)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=Collate)

if args.dataset == 'cyberverse':
    model = UVNetGraphEncoder(input_dim=5, input_edge_dim=7, output_dim=2, hidden_dim=32, num_layers=5)
else:
    model = UVNetGraphEncoder(input_dim=5, input_edge_dim=7, output_dim=2, hidden_dim=32, num_layers=6)
model.cuda()

if args.resume != "":
    model.load_state_dict(torch.load(args.resume))
if args.eval == 1:
    if not os.path.exists('eval/%s'%(args.dataset)):
        os.makedirs('eval/%s'%(args.dataset))
    if not os.path.exists('visual/wall'):
        os.makedirs('visual/wall')
    if not os.path.exists('visual/floor'):
        os.makedirs('visual/floor')
else:
    writer = SummaryWriter(logdir=args.logdir + '/summary')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def SaveFloor(sample_batched, pred_label, gt, it):
    global args
    p0 = sample_batched['g'].ndata['p0']
    p1 = sample_batched['g'].ndata['p1']
    p2 = sample_batched['g'].ndata['p2']
    amplifier = 1.0
    if args.dataset == 'cyberverse':
        amplifier = 20.0
    p0 = p0.data.cpu().numpy() * amplifier
    p1 = p1.data.cpu().numpy() * amplifier
    p2 = p2.data.cpu().numpy() * amplifier
    gt = gt.data.cpu().numpy()
    pred = pred_label.data.cpu().numpy()
    ps = np.concatenate([p0, p1, p2], axis=0)
    xMin = np.min(ps[:, 0])
    xMax = np.max(ps[:, 0])
    yMin = np.min(ps[:, 1])
    yMax = np.max(ps[:, 1])
    xMin -= 1
    yMin -= 1
    xMax += 1
    yMax += 1

    resolution = 2e-2
    h = int((yMax - yMin) / resolution)
    w = int((xMax - xMin) / resolution)
    img = np.zeros((h, w), dtype='uint8')
    img1 = np.zeros((h, w), dtype='uint8')
    for i in range(gt.shape[0]):
        bgr = int(gt[i] * 255)
        triangle = np.zeros((3, 2))
        triangle[0] = p0[i]
        triangle[1] = p1[i]
        triangle[2] = p2[i]
        triangle[:, 0] = ((triangle[:, 0] - xMin) / resolution)
        triangle[:, 1] = ((triangle[:, 1] - yMin) / resolution)
        triangle = triangle.reshape(1, 3, 2).astype('int32')
        v = int(gt[i] * 255)
        bgr = [v]
        triangle = triangle.astype('int32')
        cv2.fillConvexPoly(img, triangle, bgr)
        if pred[i] == 1:
            v = 255
        elif pred[i] == 2:
            v = 128
        else:
            v = 0
        bgr = [v]
        cv2.fillConvexPoly(img1, triangle, bgr)
    filename = 'visual/floor/%02d-gt.png'%(it)
    cv2.imwrite(filename, img)
    filename = 'visual/floor/%02d-pred.png'%(it)
    cv2.imwrite(filename, img1)

def SaveWall(sample_batched, pred_label_wall, gt_wall, it):
    global args
    p0 = sample_batched['g'].ndata['p0']
    p1 = sample_batched['g'].ndata['p1']
    p2 = sample_batched['g'].ndata['p2']
    amplifier = 1.0
    if args.dataset == 'cyberverse':
        amplifier = 20.0
    p0 = p0.data.cpu().numpy() * amplifier
    p1 = p1.data.cpu().numpy() * amplifier
    p2 = p2.data.cpu().numpy() * amplifier
    ps = np.concatenate([p0, p1, p2], axis=0)
    xMin = np.min(ps[:, 0])
    xMax = np.max(ps[:, 0])
    yMin = np.min(ps[:, 1])
    yMax = np.max(ps[:, 1])
    xMin -= 1
    yMin -= 1
    xMax += 1
    yMax += 1

    resolution = 2e-2
    h = int((yMax - yMin) / resolution)
    w = int((xMax - xMin) / resolution)
    img_line = np.zeros((h, w, 3), dtype='uint8')
    img_line1 = np.zeros((h, w, 3), dtype='uint8')
    p0 = sample_batched['g'].edata['p0'].data.cpu().numpy()
    p1 = sample_batched['g'].edata['p1'].data.cpu().numpy()
    p0 = p0 * amplifier
    p1 = p1 * amplifier
    pred_label_wall = pred_label_wall.data.cpu().numpy()
    gt_wall = gt_wall.data.cpu().numpy()
    for i in range(gt_wall.shape[0]):
        q0 = (int((p0[i, 0] - xMin) / resolution), int((p0[i, 1] - yMin) / resolution))
        q1 = (int((p1[i, 0] - xMin) / resolution), int((p1[i, 1] - yMin) / resolution))
        if gt_wall[i] == 1:
            cv2.line(img_line, q0, q1, (255, 255, 255), 1)
        if pred_label_wall[i] == 1:
            cv2.line(img_line1, q0, q1, (255, 255, 255), 1)
        elif pred_label_wall[i] == 2:
            cv2.line(img_line1, q0, q1, (64, 64, 64), 1)
    cv2.imwrite('visual/wall/%02d-gt.png'%(it), img_line)
    cv2.imwrite('visual/wall/%02d-pred.png'%(it), img_line1)

best_acc = 0
best_acc_wall = 0
train_best_acc = 0
best_acc_all = 0
softmax = nn.Softmax(dim=1)
GraphCut = cdll.LoadLibrary('src/gco/build/libgraphcut.so')

def graphcut(nScore, eScore, srcIdx, dstIdx, fIdx, eIdx, srcEdgeIdx, dstEdgeIdx, V):
    cutN = np.zeros((nScore.shape[0]), dtype='int32')
    cutE = np.zeros((eScore.shape[0]), dtype='int32')
    HE2E = np.zeros((nScore.shape[0] * 3), dtype='int32')
    GraphCut.process(c_int(nScore.shape[0]), c_int(eScore.shape[0]), c_int(V.shape[0]), c_int(srcEdgeIdx.shape[0]),
                     c_void_p(nScore.ctypes.data), c_void_p(eScore.ctypes.data),
                     c_void_p(srcIdx.ctypes.data), c_void_p(dstIdx.ctypes.data),
                     c_void_p(cutN.ctypes.data), c_void_p(cutE.ctypes.data), c_void_p(HE2E.ctypes.data),
                     c_void_p(fIdx.ctypes.data),
                     c_void_p(eIdx.ctypes.data),
                     c_void_p(srcEdgeIdx.ctypes.data),
                     c_void_p(dstEdgeIdx.ctypes.data),
                     c_void_p(V.ctypes.data),
                     c_double(args.ratioF),
                     c_double(args.ratioE),
                     c_double(args.increaseScore))
    return cutN, cutE, HE2E

def EvalEpoch(epoch):
    global best_acc, best_acc_wall, best_acc_all, args
    model.eval()
    accs = []
    accs_wall = []
    loss_wall_arr = []
    for it, sample_batched in enumerate(val_loader):
        sample_batched = {'g': sample_batched['g'].to('cuda:0'), 'e': sample_batched['e'].to('cuda:0'),
                          'v': sample_batched['v'].to('cuda:0'), 'n': sample_batched['n']}
        pred_n, pred_e = model(sample_batched['g'], sample_batched['e'])
        gt = sample_batched['g'].ndata['y']
        gt_wall = sample_batched['g'].edata['y']
        area = sample_batched['g'].ndata['area']
        area[:] = 1
        loss = F.cross_entropy(pred_n, gt, reduction='none')
        loss = torch.sum(loss * area)
        pred_label = pred_n[:, 1] > pred_n[:, 0]
        pred_label_wall = pred_e[:, 1] > pred_e[:, 0]
        if args.eval == 1:
            pred_e_score = softmax(pred_e)[:, 1].data.cpu().numpy().astype('float64')
            pred_n_score = softmax(pred_n)[:, 1].data.cpu().numpy().astype('float64')
            src, dst = sample_batched['g'].edges()
            src = src.data.cpu().numpy().astype('int32')
            dst = dst.data.cpu().numpy().astype('int32')
            scale = 1.0
            if args.dataset == 'cyberverse':
                scale = 100.0
            elif args.dataset == 'structured3d':
                scale = 10.0
            cutN, cutE, HE2E = graphcut(pred_n_score, pred_e_score, src, dst,
                                        sample_batched['g'].ndata['fIdx'].cpu().numpy().astype('int32'),
                                        sample_batched['g'].edata['eIdx'].cpu().numpy().astype('int32'),
                                        sample_batched['e'].edges()[0].cpu().numpy().astype('int32'),
                                        sample_batched['e'].edges()[1].cpu().numpy().astype('int32'),
                                        sample_batched['v'].cpu().numpy().astype('float64') * scale)
            pred_label = torch.from_numpy(cutN).cuda()
            pred_label_wall = torch.from_numpy(cutE).cuda()
            np.savez_compressed('eval/%s/%s.npz'%(args.dataset, sample_batched['n'][0]),
                               V=sample_batched['v'].cpu().numpy().astype('float64') * 100,
                               F=sample_batched['g'].ndata['fIdx'].cpu().numpy().astype('int32'),
                               gtFaceLabel=gt.cpu().numpy().astype('int32'),
                               predFaceLabel=cutN,
                               gtWallLabel=gt_wall.cpu().numpy().astype('int32')[HE2E],
                               predWallLabel=cutE[HE2E])
        loss_wall = F.cross_entropy(pred_e, gt_wall, reduction='none')
        lens = sample_batched['g'].edata['len']
        lens[:] = 1
        loss_wall = torch.sum(loss_wall * lens)
        true_positive = torch.sum((pred_label > 0) * gt * area) / torch.sum(area)
        false_positive = torch.sum((pred_label > 0) * (gt == 0) * area) / torch.sum(area)
        false_negative = torch.sum((pred_label == 0) * gt * area) / torch.sum(area)
        acc = (true_positive / (true_positive + false_positive + false_negative)).item()

        tp_wall = torch.sum((pred_label_wall > 0) * gt_wall * lens) / torch.sum(lens)
        fp_wall = torch.sum((pred_label_wall > 0) * (gt_wall == 0) * lens) / torch.sum(lens)
        fn_wall = torch.sum((pred_label_wall == 0) * gt_wall * lens) / torch.sum(lens)
        acc_wall = (tp_wall / (tp_wall + fp_wall + fn_wall)).item()
        accs.append(acc)
        accs_wall.append(acc_wall)
        loss_wall_arr.append(loss_wall.item())
        if args.eval == 1:
            SaveFloor(sample_batched, pred_label, gt, it)
            SaveWall(sample_batched, pred_label_wall, gt_wall, it)
        del sample_batched, pred_n, pred_e, gt, gt_wall, area, loss, pred_label_wall, pred_label, lens, loss_wall
        torch.cuda.empty_cache()
    acc = np.mean(accs)
    acc_wall = np.mean(accs_wall)
    if acc > best_acc:
        best_acc = acc
        if args.eval == 0:
            torch.save(model.state_dict(), 'best-floor.ckpt')
    if acc_wall > best_acc_wall:
        best_acc_wall = acc_wall
        if args.eval == 0:
            torch.save(model.state_dict(), 'best-wall.ckpt')
    if best_acc_all < acc + acc_wall:
        best_acc_all = acc + acc_wall
        if args.eval == 0:
            torch.save(model.state_dict(), 'best-floor-wall.ckpt')
    print('iter = %d, eval bestacc = '%(epoch), best_acc, best_acc_wall, np.mean(loss_wall_arr), best_acc_all * 0.5)
    if args.eval == 0:
        torch.save(model.state_dict(), 'current.ckpt')
    else:
        exit(0)

def TrainEpoch(epoch):
    global train_best_acc
    loss_arr = []
    loss_wall_arr = []
    accs = []
    model.train()
    for it, sample_batched in enumerate(train_loader):
        sample_batched = {'g': sample_batched['g'].to('cuda:0'), 'e': sample_batched['e'].to('cuda:0')}
        optimizer.zero_grad()
        pred_n, pred_e = model(sample_batched['g'], sample_batched['e'])
        gt = sample_batched['g'].ndata['y']
        gt_wall = sample_batched['g'].edata['y']
        loss_wall = F.cross_entropy(pred_e, gt_wall, reduction='none')
        loss = F.cross_entropy(pred_n, gt, reduction='none')
        area = sample_batched['g'].ndata['area']
        lens = sample_batched['g'].edata['len']
        area[:] = 1
        lens[:] = 1
        loss = torch.sum(loss * area) / torch.sum(area)
        loss_wall = torch.sum(loss_wall * lens) / torch.sum(lens)
        l = loss + loss_wall
        l.backward()
        loss_arr.append(loss.item())
        loss_wall_arr.append(loss_wall.item())
        optimizer.step()

        pred_label = pred_n[:, 1] > pred_n[:, 0]
        true_positive = torch.sum((pred_label > 0) * gt * area) / torch.sum(area)
        false_positive = torch.sum((pred_label > 0) * (gt == 0) * area) / torch.sum(area)
        false_negative = torch.sum((pred_label == 0) * gt * area) / torch.sum(area)
        acc = (true_positive / (true_positive + false_positive + false_negative)).item()
        accs.append(acc)
        del sample_batched, pred_n, pred_e, gt, gt_wall, area, lens, loss, l, pred_label
        torch.cuda.empty_cache()
    acc = np.mean(accs)
    if acc > train_best_acc:
        train_best_acc = acc
    print('iter = %d, train bestacc = '%(epoch), train_best_acc)
    print('loss = ', np.mean(loss_arr), np.mean(loss_wall_arr))

for epoch in range(300):
    EvalEpoch(epoch)
    torch.cuda.empty_cache()
    TrainEpoch(epoch)
    torch.cuda.empty_cache()
print('final bestacc = ', best_acc)



