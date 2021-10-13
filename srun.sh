#!/bin/bash
pip install mmcv-full==1.3.14
pip install git+file:///netscratch/minouei/versicherung/src/mmseg21
python -u tools/train.py configs/0ver/segformer_mit-b2_512x512_160k_ver.py  --launcher=slurm
