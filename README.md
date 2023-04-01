## Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners

![Alt Text](https://vis-www.cs.umass.edu/mod-squad/materials/annimation.mp4)



This is a PyTorch/GPU implementation of the paper [Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners](https://arxiv.org/abs/2212.08066):
```
@article{chen2022modsquad,
            title={Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners},
            author={Zitian Chen and Yikang Shen and Mingyu Ding and Zhenfang Chen and Hengshuang Zhao and Erik Learned-Miller and Chuang Gan},
            journal={CVPR},
            year={2023}
}
```

* The implementation was based on Pytorch==1.10.2 and test on powerpc server.



### Prepare

Dataset: [Taskonomy](https://github.com/StanfordVL/taskonomy)

An example of the download from tiny subset

```
omnitools.download class_object class_scene depth_euclidean depth_zbuffer edge_occlusion edge_texture keypoints2d keypoints3d nonfixated_matches normal points principal_curvature reshading rgb segment_semantic segment_unsup2d segment_unsup25d --components taskonomy --subset tiny --dest ./taskonomy_tiny/   --connections_total 40 --agree --name [your name] --email [your email]
```



Environment: timm==0.3.2 pytorch==1.10.2

Install MoE module:

```
cd parallel_linear
pip3 install .
```

### Train

```
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 44875 main_mt.py \
        --batch_size 6 \
        --epochs 100 \
        --input_size 224 \
        --blr 4e-4 --weight_decay 0.05 \
        --warmup_epochs 10 \
        --model mtvit_taskgate_att_mlp_base_MI_twice \
        --drop_path 0.1 \
        --scaleup \
        --exp-name scaleup_mtvit_taskgate_att_mlp_base_MI_twice \
```

