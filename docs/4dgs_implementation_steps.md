# 4DGS集成实施步骤指南

## 📋 概述

本文档提供了4DGS神经网络模块集成到World Model项目的详细实施步骤。按照本指南，您可以逐步完成4DGS模块的集成工作。

## 🎯 实施目标

- ✅ 将4DGS模块集成到MMDetection3D框架
- ✅ 保持4DGS原有功能完整性
- ✅ 提供统一的配置和训练接口
- ✅ 确保系统稳定性和可维护性

## 📅 实施时间表

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|----------|------|
| 阶段1 | 基础集成 | 2-3天 | 🔄 进行中 |
| 阶段2 | 深度集成 | 1-2周 | ⏳ 待开始 |
| 阶段3 | 优化扩展 | 1周 | ⏳ 待开始 |

## 🚀 阶段1：基础集成

### 步骤1.1：创建目录结构

首先创建4DGS集成所需的目录结构：

```bash
# 在项目根目录执行
mkdir -p projects/mmdet3d_plugin/4DGS/models
mkdir -p projects/mmdet3d_plugin/4DGS/datasets
mkdir -p projects/mmdet3d_plugin/4DGS/adapters
mkdir -p projects/mmdet3d_plugin/4DGS/configs
mkdir -p projects/configs/4DGS
mkdir -p examples/4dgs
mkdir -p tests/test_4dgs
```

### 步骤1.2：更新模块注册

编辑 `projects/mmdet3d_plugin/__init__.py`，添加4DGS模块导入：

```python
# 在文件末尾添加以下内容
from .4DGS import *

# 如果需要显式导入特定组件
from .4DGS.models import FourDGSModel
from .4DGS.datasets import FourDGSDataset
```

### 步骤1.3：创建模型包装器

创建 `projects/mmdet3d_plugin/4DGS/models/__init__.py`：

```python
"""
4DGS模型模块初始化
"""
from .fourdgs_model import FourDGSModel

__all__ = ['FourDGSModel']
```

创建 `projects/mmdet3d_plugin/4DGS/models/fourdgs_model.py`：

```python
"""
4DGS模型包装器，用于集成到MMDet3D框架
"""
import torch
import torch.nn as nn
from mmdet3d.models import DETECTORS, BaseDetector
from mmdet3d.core import bbox3d2result
from mmcv.runner import auto_fp16

# 导入4DGS原始组件
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim


@DETECTORS.register_module()
class FourDGSModel(BaseDetector):
    """
    4DGS模型的MMDet3D包装器
    
    这个类将4DGS的训练和推理逻辑包装成MMDet3D兼容的接口，
    使得4DGS可以无缝集成到MMDet3D的训练和推理流程中。
    
    Args:
        model_params (dict): 4DGS模型参数配置
        optimization_params (dict): 优化参数配置  
        pipeline_params (dict): 渲染管道参数配置
        train_cfg (dict): 训练配置
        test_cfg (dict): 测试配置
        pretrained (str): 预训练模型路径
    """
    
    def __init__(self,
                 model_params=None,
                 optimization_params=None,
                 pipeline_params=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FourDGSModel, self).__init__()
        
        # 保存配置参数
        self.model_params = model_params or {}
        self.optimization_params = optimization_params or {}
        self.pipeline_params = pipeline_params or {}
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 初始化4DGS核心组件
        self.gaussians = GaussianModel(self.model_params.get('sh_degree', 3))
        self.scene = None
        self.background = torch.tensor([1, 1, 1] if self.model_params.get('white_background', False) else [0, 0, 0], dtype=torch.float32, device="cuda")
        
        # 训练状态
        self.iteration = 0
        
    def init_weights(self, pretrained=None):
        """
        初始化模型权重
        
        Args:
            pretrained (str): 预训练模型路径
        """
        if pretrained is not None:
            # 加载预训练的4DGS模型
            self.load_4dgs_checkpoint(pretrained)
            
    def load_4dgs_checkpoint(self, checkpoint_path):
        """
        加载4DGS检查点
        
        Args:
            checkpoint_path (str): 检查点文件路径
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.gaussians.restore(checkpoint, self.optimization_params)
            
    def extract_feat(self, imgs):
        """
        特征提取（4DGS不需要传统的特征提取）
        
        Args:
            imgs: 输入图像
            
        Returns:
            imgs: 直接返回输入图像
        """
        return imgs
        
    @auto_fp16(apply_to=('imgs',))
    def forward_train(self,
                      imgs,
                      img_metas,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      **kwargs):
        """
        训练前向传播
        
        Args:
            imgs (torch.Tensor): 输入图像 [B, C, H, W]
            img_metas (list[dict]): 图像元信息
            gt_bboxes_3d: 3D边界框标注（4DGS不使用）
            gt_labels_3d: 3D标签（4DGS不使用）
            
        Returns:
            dict: 损失字典
        """
        losses = dict()
        
        # 这里需要根据实际的4DGS训练数据格式进行适配
        # 目前返回模拟损失，实际实现需要：
        # 1. 从img_metas中提取相机参数
        # 2. 调用4DGS渲染器进行渲染
        # 3. 计算L1和SSIM损失
        
        batch_size = imgs.shape[0]
        device = imgs.device
        
        # 模拟损失计算（实际实现时需要替换）
        total_l1_loss = 0.0
        total_ssim_loss = 0.0
        
        for i in range(batch_size):
            # 这里应该调用4DGS的渲染和损失计算逻辑
            # render_result = self.render_image(img_metas[i])
            # l1_loss_val = l1_loss(render_result, imgs[i])
            # ssim_loss_val = 1.0 - ssim(render_result, imgs[i])
            
            # 临时模拟损失
            l1_loss_val = torch.tensor(0.1, device=device, requires_grad=True)
            ssim_loss_val = torch.tensor(0.05, device=device, requires_grad=True)
            
            total_l1_loss += l1_loss_val
            total_ssim_loss += ssim_loss_val
        
        # 计算平均损失
        losses['loss_l1'] = total_l1_loss / batch_size
        losses['loss_ssim'] = total_ssim_loss / batch_size
        
        # 总损失
        lambda_dssim = self.optimization_params.get('lambda_dssim', 0.2)
        losses['loss_total'] = (1.0 - lambda_dssim) * losses['loss_l1'] + lambda_dssim * losses['loss_ssim']
        
        return losses
        
    def simple_test(self, imgs, img_metas, **kwargs):
        """
        简单测试（推理）
        
        Args:
            imgs (torch.Tensor): 输入图像
            img_metas (list[dict]): 图像元信息
            
        Returns:
            list: 推理结果
        """
        results = []
        
        for i, img_meta in enumerate(img_metas):
            # 4DGS推理逻辑
            # 这里应该调用4DGS的渲染器生成新视角图像
            result = dict(
                img_meta=img_meta,
                # rendered_image=rendered_result,  # 实际渲染结果
                # depth_map=depth_result,          # 深度图
                # gaussian_count=self.gaussians.get_xyz.shape[0]  # 高斯点数量
            )
            results.append(result)
            
        return results
        
    def aug_test(self, imgs, img_metas, **kwargs):
        """
        增强测试
        
        Args:
            imgs: 输入图像
            img_metas: 图像元信息
            
        Returns:
            list: 测试结果
        """
        return self.simple_test(imgs, img_metas, **kwargs)
        
    def render_image(self, camera_info):
        """
        渲染单张图像
        
        Args:
            camera_info (dict): 相机信息
            
        Returns:
            torch.Tensor: 渲染结果
        """
        # 这里实现4DGS的渲染逻辑
        # 需要从camera_info中提取相机参数，调用render函数
        pass
        
    def show_result(self, *args, **kwargs):
        """显示结果"""
        pass
        
    def forward_dummy(self, imgs):
        """
        用于计算FLOPs的虚拟前向传播
        
        Args:
            imgs: 输入图像
            
        Returns:
            tuple: 输出结果
        """
        return self.simple_test(imgs, [{}] * imgs.shape[0])
```

### 步骤1.4：创建基础配置文件

创建 `projects/configs/4DGS/4dgs_base.py`：

```python
"""
4DGS基础配置文件
"""
_base_ = ['../_base_/default_runtime.py']

# 插件配置
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# 4DGS模型配置
model = dict(
    type='FourDGSModel',
    
    # 4DGS模型参数
    model_params=dict(
        sh_degree=3,                    # 球谐函数度数
        source_path="",                 # 数据源路径
        model_path="",                  # 模型保存路径
        images="images",                # 图像文件夹名称
        resolution=-1,                  # 图像分辨率（-1表示原始分辨率）
        white_background=False,         # 是否使用白色背景
        data_device="cuda",             # 数据设备
        eval=False                      # 是否为评估模式
    ),
    
    # 优化参数
    optimization_params=dict(
        iterations=30000,               # 训练迭代次数
        position_lr_init=0.00016,       # 位置学习率初始值
        position_lr_final=0.0000016,    # 位置学习率最终值
        position_lr_delay_mult=0.01,    # 位置学习率延迟倍数
        position_lr_max_steps=30000,    # 位置学习率最大步数
        feature_lr=0.0025,              # 特征学习率
        opacity_lr=0.05,                # 透明度学习率
        scaling_lr=0.005,               # 缩放学习率
        rotation_lr=0.001,              # 旋转学习率
        percent_dense=0.01,             # 密化百分比
        lambda_dssim=0.2,               # DSSIM损失权重
        densification_interval=100,     # 密化间隔
        opacity_reset_interval=3000,    # 透明度重置间隔
        densify_from_iter=500,          # 开始密化的迭代次数
        densify_until_iter=15000,       # 结束密化的迭代次数
        densify_grad_threshold=0.0002   # 密化梯度阈值
    ),
    
    # 渲染管道参数
    pipeline_params=dict(
        convert_SHs_python=False,       # 是否使用Python转换球谐函数
        compute_cov3D_python=False,     # 是否使用Python计算3D协方差
        debug=False                     # 是否开启调试模式
    ),
    
    # 训练配置
    train_cfg=dict(),
    
    # 测试配置
    test_cfg=dict()
)

# 数据集配置
dataset_type = 'FourDGSDataset'
data_root = 'data/4dgs/'

# 数据管道
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['img'])
         ])
]

# 数据配置
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True
    )
)

# 优化器配置
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率调度
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20000, 25000]
)

# 运行时配置
runner = dict(type='EpochBasedRunner', max_epochs=100)

# 检查点配置
checkpoint_config = dict(interval=1)

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# 评估配置
evaluation = dict(interval=1, metric='mAP')

# 分布式配置
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/4dgs_base'
load_from = None
resume_from = None
workflow = [('train', 1)]
```

### 步骤1.5：创建数据集适配器

创建 `projects/mmdet3d_plugin/4DGS/datasets/__init__.py`：

```python
"""
4DGS数据集模块初始化
"""
from .fourdgs_dataset import FourDGSDataset

__all__ = ['FourDGSDataset']
```

创建 `projects/mmdet3d_plugin/4DGS/datasets/fourdgs_dataset.py`：

```python
"""
4DGS数据集适配器
"""
import numpy as np
import torch
from mmdet3d.datasets import DATASETS, Custom3DDataset
from mmdet3d.core.bbox import get_box_type


@DATASETS.register_module()
class FourDGSDataset(Custom3DDataset):
    """
    4DGS数据集包装器，用于适配MMDet3D数据管道
    
    这个类将4DGS的数据格式适配到MMDet3D的标准数据接口，
    使得4DGS可以使用MMDet3D的数据加载和预处理功能。
    
    Args:
        data_root (str): 数据根目录
        ann_file (str): 标注文件路径
        pipeline (list): 数据处理管道
        classes (list): 类别列表
        modality (dict): 模态配置
        box_type_3d (str): 3D边界框类型
        filter_empty_gt (bool): 是否过滤空标注
        test_mode (bool): 是否为测试模式
    """
    
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        
        # 初始化类别信息
        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        
        # 加载数据信息
        self.data_infos = self.load_annotations(self.ann_file)
        
        # 过滤空标注
        if not test_mode and filter_empty_gt:
            self.data_infos = self.filter_empty_gt_data(self.data_infos)
            
        # 初始化数据管道
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
            
        # 设置标志位
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
        
    def load_annotations(self, ann_file):
        """
        加载标注文件
        
        Args:
            ann_file (str): 标注文件路径
            
        Returns:
            list: 数据信息列表
        """
        # 这里需要根据实际的4DGS数据格式进行实现
        # 目前返回模拟数据
        data_infos = []
        
        # 示例数据结构
        for i in range(100):  # 假设有100个样本
            info = dict(
                sample_idx=i,
                img_filename=f'image_{i:06d}.jpg',
                img_shape=(800, 800, 3),
                lidar_path=f'lidar_{i:06d}.bin',
                sweeps=[],
                cams=dict(),
                lidar2ego_translation=[0.0, 0.0, 0.0],
                lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
                ego2global_translation=[0.0, 0.0, 0.0],
                ego2global_rotation=[1.0, 0.0, 0.0, 0.0],
                timestamp=1000000 + i,
                gt_boxes_3d=np.array([]),  # 3D边界框
                gt_labels_3d=np.array([]), # 3D标签
                gt_names=[]                # 类别名称
            )
            data_infos.append(info)
            
        return data_infos
        
    def filter_empty_gt_data(self, data_infos):
        """
        过滤空标注数据
        
        Args:
            data_infos (list): 原始数据信息
            
        Returns:
            list: 过滤后的数据信息
        """
        filtered_infos = []
        for info in data_infos:
            if len(info['gt_boxes_3d']) > 0:
                filtered_infos.append(info)
        return filtered_infos
        
    def get_data_info(self, index):
        """
        获取指定索引的数据信息
        
        Args:
            index (int): 数据索引
            
        Returns:
            dict: 数据信息
        """
        info = self.data_infos[index]
        
        # 构建输入数据字典
        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=info.get('lidar_path', None),
            sweeps=info['sweeps'],
            timestamp=info['timestamp'],
            img_filename=info['img_filename'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )
        
        # 添加相机信息
        if 'cams' in info:
            input_dict['cams'] = info['cams']
            
        # 添加标注信息
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            
        return input_dict
        
    def get_ann_info(self, index):
        """
        获取标注信息
        
        Args:
            index (int): 数据索引
            
        Returns:
            dict: 标注信息
        """
        info = self.data_infos[index]
        
        # 构建标注字典
        annos = dict(
            gt_bboxes_3d=info['gt_boxes_3d'],
            gt_labels_3d=info['gt_labels_3d'],
            gt_names=info['gt_names']
        )
        
        return annos
        
    def prepare_train_data(self, index):
        """
        准备训练数据
        
        Args:
            index (int): 数据索引
            
        Returns:
            dict: 处理后的训练数据
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
            
        # 应用数据处理管道
        example = self.pipeline(input_dict)
        
        return example
        
    def prepare_test_data(self, index):
        """
        准备测试数据
        
        Args:
            index (int): 数据索引
            
        Returns:
            dict: 处理后的测试数据
        """
        input_dict = self.get_data_info(index)
        
        # 应用数据处理管道
        example = self.pipeline(input_dict)
        
        return example
        
    def format_results(self, results, jsonfile_prefix=None):
        """
        格式化结果
        
        Args:
            results (list): 预测结果
            jsonfile_prefix (str): JSON文件前缀
            
        Returns:
            dict: 格式化后的结果
        """
        # 实现结果格式化逻辑
        formatted_results = dict()
        
        return formatted_results
        
    def evaluate(self, results, metric='mAP', logger=None, **kwargs):
        """
        评估结果
        
        Args:
            results (list): 预测结果
            metric (str): 评估指标
            logger: 日志记录器
            
        Returns:
            dict: 评估结果
        """
        # 实现评估逻辑
        eval_results = dict()
        
        if metric == 'mAP':
            # 计算mAP
            eval_results['mAP'] = 0.0
            
        return eval_results
```

### 步骤1.6：更新4DGS模块初始化

编辑 `projects/mmdet3d_plugin/4DGS/__init__.py`：

```python
"""
4DGS模块初始化文件
"""
from .models import *
from .datasets import *

# 保持原有导入
from .gaussian_renderer import *
from .scene import *
from .scripts import *
from .utils import *

# 新增MMDet3D兼容组件
__all__ = [
    'GaussianRenderer', 'Scene', 'Scripts', 'Utils',
    'FourDGSModel', 'FourDGSDataset'
]
```

### 步骤1.7：创建训练脚本

创建 `examples/4dgs/train_4dgs.py`：

```python
#!/usr/bin/env python3
"""
4DGS训练示例脚本

使用方法:
python examples/4dgs/train_4dgs.py projects/configs/4DGS/4dgs_base.py --work-dir work_dirs/4dgs_base
"""

import argparse
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mmcv import Config
from mmdet3d.apis import train_detector
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train 4DGS model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
        
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        
    cfg.gpus = args.gpus
    
    # 创建工作目录
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 初始化日志记录器
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    # 记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    
    # 记录配置信息
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # 设置随机种子
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
        
    # 构建模型
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    
    # 开始训练
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=(not args.no_validate),
        timestamp=timestamp
    )


if __name__ == '__main__':
    main()
```

### 步骤1.8：创建测试脚本

创建 `examples/4dgs/test_4dgs.py`：

```python
#!/usr/bin/env python3
"""
4DGS测试示例脚本

使用方法:
python examples/4dgs/test_4dgs.py projects/configs/4DGS/4dgs_base.py work_dirs/4dgs_base/latest.pth --eval mAP
"""

import argparse
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mmcv import Config
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test 4DGS model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--eval', type=str, nargs='+', help='evaluation metrics')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 构建数据集
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )
    
    # 构建模型
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    
    # 加载检查点
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 设置类别信息
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
        
    # 包装模型
    model = MMDataParallel(model, device_ids=[0])
    
    # 开始测试
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    
    # 保存结果
    if args.out:
        print(f'Writing results to {args.out}')
        mmcv.dump(outputs, args.out)
        
    # 评估结果
    if args.eval:
        eval_kwargs = {}
        eval_results = dataset.evaluate(outputs, args.eval, **eval_kwargs)
        for k, v in eval_results.items():
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
```

## 🧪 阶段1验证

### 验证步骤1：检查模块注册

```bash
# 在项目根目录执行
python -c "
from projects.mmdet3d_plugin.4DGS.models import FourDGSModel
from projects.mmdet3d_plugin.4DGS.datasets import FourDGSDataset
print('4DGS模块注册成功!')
"
```

### 验证步骤2：测试配置加载

```bash
# 测试配置文件加载
python -c "
from mmcv import Config
cfg = Config.fromfile('projects/configs/4DGS/4dgs_base.py')
print('配置加载成功!')
print(f'模型类型: {cfg.model.type}')
"
```

### 验证步骤3：测试模型构建

```bash
# 测试模型构建
python -c "
from mmcv import Config
from mmdet3d.models import build_model
cfg = Config.fromfile('projects/configs/4DGS/4dgs_base.py')
model = build_model(cfg.model)
print('模型构建成功!')
print(f'模型类型: {type(model)}')
"
```

## 📝 阶段1总结

完成阶段1后，您应该已经：

- ✅ 创建了完整的目录结构
- ✅ 实现了基础的模型包装器
- ✅ 创建了数据集适配器
- ✅ 配置了基础的训练和测试流程
- ✅ 验证了模块注册和配置加载

## 🔄 下一步

阶段1完成后，您可以继续进行：

1. **阶段2：深度集成** - 实现完整的数据适配和训练逻辑
2. **阶段3：优化扩展** - 性能优化和功能扩展
3. **测试验证** - 全面的功能和性能测试

## 🚨 注意事项

1. **依赖检查**：确保所有必要的依赖已正确安装
2. **路径配置**：检查所有文件路径是否正确
3. **权限设置**：确保有足够的文件读写权限
4. **GPU内存**：监控GPU内存使用情况
5. **版本兼容**：确保各组件版本兼容

## 📞 获取帮助

如果在实施过程中遇到问题，请参考：
- [故障排除文档](4dgs_troubleshooting.md)
- [API参考手册](4dgs_api_reference.md)
- [技术分析报告](4dgs_technical_analysis.md)

---

*本实施指南将随着项目进展持续更新，请关注最新版本。*