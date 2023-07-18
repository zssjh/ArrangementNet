# ArrangementNet
### Official implementation of the paper ArrangementNet: Learning Scene Arrangements for Vectorized Indoor Scene Modeling. (SIGGRAPH 2023)

We provide three datasets at `data/`, which are consistent with the paper. **cyberverse** includes 54 large-scale scenes, **floorsp** is obtained from https://github.com/woodfrog/floor-sp, **structured3d** is obtained from https://github.com/bertjiazheng/Structured3D. We have already generated the arrangement graph based on three datasets and saved at `data/*/arrangement_graph`, which can be directly used as input to the network. We provide the evaluation groundtruth at `data/*/evaluation_groundtruth`.

#### 1. Environment Setup

This repo was developed and tested with **Python3.7**, and you should install the following version of dgl and torch.

>dgl==0.6.1
>
>torch==1.8.0


#### 2. complie gco

We rectify the network prediction by graphcut, so you should compile gco to get `libgraphcut.so` at `src/gco/build`.

```shell
cd src/gco
mkdi build && cd build
cmake ..
make -j8
```


#### 3. train
```shell
python src/train.py --dataset cyberverse
python src/train.py --dataset floorsp
python src/train.py --dataset structured3d
```

#### 4. inference


You can replicate the results in the paper using the pretrained model under `checkpoint/`.
```shell
python src/train.py --eval 1 --dataset cyberverse --resume checkpoint/cyberverse.ckpt
python src/train.py --eval 1 --dataset floorsp --resume checkpoint/floorsp.ckpt
python src/train.py --eval 1 --dataset structured3d --resume checkpoint/structured3d.ckpt
```

#### 5. evaluate

The inference results will be saved at `eval/`, then you can evaluate the results using the following command to get the precision and recall of Room/Corner/Angle.
```shell
python evaluation/evaluations.py --predict_result eval/cyberverse --dataset cyberverse
python evaluation/evaluations.py --predict_result eval/floorsp --dataset floorsp
python evaluation/evaluations.py --predict_result eval/structured3d --dataset structured3d
```

For example, you will get evaluation result of cyberverse dataset like this

>avg Room precision and recall 0.8804061910311912 0.8134793800970271
>
>avg Corner precision and recall 0.8220827182273324 0.8274320067970539
>
>avg Angle precision and recall 0.8004808248122176 0.8051524746935288

