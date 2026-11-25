# HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction

> **이 폴더:** AV2 + 클래스(`actor_type`) HiVT 코드.  
> 워크스페이스 루트: [`../README.md`](../README.md) · 구조 [`../STRUCTURE.md`](../STRUCTURE.md)  
> Sparse4D 연결: [`../integration/`](../integration/) · 시간 동기화 [`../integration/docs/TIME_SYNC.md`](../integration/docs/TIME_SYNC.md)  
> Sparse4D 가중치: [`../Sparse4D/checkpoints/`](../Sparse4D/checkpoints/)

This repository contains the official implementation of [HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf) published in CVPR 2022.

![](assets/overview.png)

## Gettting Started

1\. Clone this repository:
```
git clone https://github.com/ZikangZhou/HiVT.git
cd HiVT
```

2\. **이 워크스페이스(Sparse4D 연결) 최초 설치는 루트 [`../README.md`](../README.md)를 따르세요.**  
   아래는 원 논문 HiVT(AV1) 참고용이며, 현재 연결 환경(torch 2.4 + mmcv 등)과 다릅니다.

```
# (참고·비권장 for this workspace)
conda create -n HiVT python=3.8
conda activate HiVT
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch-geometric==1.7.2 -c rusty1s -c conda-forge
conda install pytorch-lightning==1.5.2 -c conda-forge
```

3\. Argoverse 1 / `argoverse-api`, vendored `av2-api`, and local AV2 `datasets/train|val|test`
   were removed (no HiVT retrain planned). Use pip `av2` if needed.
   Connection tests use `integration/fixtures` + Sparse4D.

## Training

AV2 only. Example:

```
python train.py --root /path/to/av2_dataset_root/ --embed_dim 128
```

**Note**: When running the training script for the first time, it will take several hours to preprocess the data (~3.5 hours on my machine). Training on an RTX 2080 Ti GPU takes 35-40 minutes per epoch.

During training, the checkpoints will be saved in `lightning_logs/` automatically. To monitor the training process:
```
tensorboard --logdir lightning_logs/
```

## Evaluation

To evaluate the prediction performance:
```
python eval.py --root /path/to/dataset_root/ --batch_size 32 --ckpt_path /path/to/your_checkpoint.ckpt
```

## Pretrained Models

We provide the pretrained HiVT-64 and HiVT-128 in [checkpoints/](checkpoints). You can evaluate the pretrained models using the aforementioned evaluation command, or have a look at the training process via TensorBoard:
```
tensorboard --logdir checkpoints/
```

## Results

### Quantitative Results

For this repository, the expected performance on Argoverse 1.1 validation set is:

| Models | minADE | minFDE | MR |
| :--- | :---: | :---: | :---: |
| HiVT-64 | 0.69 | 1.03 | 0.10 |
| HiVT-128 | 0.66 | 0.97 | 0.09 |

### Qualitative Results

![](assets/visualization.png)

## Citation

If you found this repository useful, please consider citing our work:

```
@inproceedings{zhou2022hivt,
  title={HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction},
  author={Zhou, Zikang and Ye, Luyao and Wang, Jianping and Wu, Kui and Lu, Kejie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## License

This repository is licensed under [Apache 2.0](LICENSE).

