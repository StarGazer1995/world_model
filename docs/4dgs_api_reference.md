# 4DGS API参考手册

## 📖 概述

本文档提供了4DGS模块集成到World Model项目后的完整API参考，包括所有类、方法、函数和配置选项的详细说明。

## 📚 目录

- [核心模型API](#核心模型api)
- [数据集API](#数据集api)
- [配置API](#配置api)
- [工具函数API](#工具函数api)
- [渲染器API](#渲染器api)
- [场景管理API](#场景管理api)
- [训练API](#训练api)
- [推理API](#推理api)

---

## 🧠 核心模型API

### FourDGSModel

4DGS模型的MMDet3D包装器，提供与MMDetection3D框架的完整集成。

```python
class FourDGSModel(BaseDetector):
    """
    4DGS模型的MMDet3D包装器
    
    Args:
        model_params (dict): 4DGS模型参数配置
        optimization_params (dict): 优化参数配置  
        pipeline_params (dict): 渲染管道参数配置
        train_cfg (dict): 训练配置
        test_cfg (dict): 测试配置
        pretrained (str): 预训练模型路径
    """
```

#### 构造函数参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `model_params` | dict | None | 4DGS模型参数配置 |
| `optimization_params` | dict | None | 优化参数配置 |
| `pipeline_params` | dict | None | 渲染管道参数配置 |
| `train_cfg` | dict | None | 训练配置 |
| `test_cfg` | dict | None | 测试配置 |
| `pretrained` | str | None | 预训练模型路径 |

#### 主要方法

##### `__init__(self, model_params=None, optimization_params=None, pipeline_params=None, train_cfg=None, test_cfg=None, pretrained=None)`

**功能**: 初始化4DGS模型

**参数**:
- `model_params` (dict): 模型参数配置
- `optimization_params` (dict): 优化参数配置
- `pipeline_params` (dict): 渲染管道参数配置
- `train_cfg` (dict): 训练配置
- `test_cfg` (dict): 测试配置
- `pretrained` (str): 预训练模型路径

**返回**: None

**示例**:
```python
model = FourDGSModel(
    model_params={'sh_degree': 3, 'white_background': False},
    optimization_params={'iterations': 30000, 'position_lr_init': 0.00016},
    pipeline_params={'convert_SHs_python': False}
)
```

##### `init_weights(self, pretrained=None)`

**功能**: 初始化模型权重

**参数**:
- `pretrained` (str): 预训练模型路径

**返回**: None

**示例**:
```python
model.init_weights(pretrained='path/to/checkpoint.pth')
```

##### `load_4dgs_checkpoint(self, checkpoint_path)`

**功能**: 加载4DGS检查点

**参数**:
- `checkpoint_path` (str): 检查点文件路径

**返回**: None

**示例**:
```python
model.load_4dgs_checkpoint('work_dirs/4dgs/checkpoint.pth')
```

##### `extract_feat(self, imgs)`

**功能**: 特征提取（4DGS直接返回输入图像）

**参数**:
- `imgs` (torch.Tensor): 输入图像张量 [B, C, H, W]

**返回**: torch.Tensor - 输入图像张量

**示例**:
```python
features = model.extract_feat(input_images)
```

##### `forward_train(self, imgs, img_metas, gt_bboxes_3d=None, gt_labels_3d=None, **kwargs)`

**功能**: 训练前向传播

**参数**:
- `imgs` (torch.Tensor): 输入图像 [B, C, H, W]
- `img_metas` (list[dict]): 图像元信息
- `gt_bboxes_3d` (torch.Tensor): 3D边界框标注（可选）
- `gt_labels_3d` (torch.Tensor): 3D标签（可选）
- `**kwargs`: 其他参数

**返回**: dict - 损失字典

**示例**:
```python
losses = model.forward_train(
    imgs=input_images,
    img_metas=image_metas,
    gt_bboxes_3d=gt_boxes,
    gt_labels_3d=gt_labels
)
```

##### `simple_test(self, imgs, img_metas, **kwargs)`

**功能**: 简单测试（推理）

**参数**:
- `imgs` (torch.Tensor): 输入图像
- `img_metas` (list[dict]): 图像元信息
- `**kwargs`: 其他参数

**返回**: list - 推理结果列表

**示例**:
```python
results = model.simple_test(input_images, image_metas)
```

##### `render_image(self, camera_info)`

**功能**: 渲染单张图像

**参数**:
- `camera_info` (dict): 相机信息字典

**返回**: torch.Tensor - 渲染结果

**示例**:
```python
rendered_image = model.render_image({
    'camera_center': [0, 0, 0],
    'camera_rotation': [1, 0, 0, 0],
    'fov': 60.0
})
```

---

## 📊 数据集API

### FourDGSDataset

4DGS数据集适配器，用于将4DGS数据格式适配到MMDet3D标准接口。

```python
class FourDGSDataset(Custom3DDataset):
    """
    4DGS数据集包装器
    
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
```

#### 构造函数参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `data_root` | str | - | 数据根目录 |
| `ann_file` | str | - | 标注文件路径 |
| `pipeline` | list | None | 数据处理管道 |
| `classes` | list | None | 类别列表 |
| `modality` | dict | None | 模态配置 |
| `box_type_3d` | str | 'LiDAR' | 3D边界框类型 |
| `filter_empty_gt` | bool | True | 是否过滤空标注 |
| `test_mode` | bool | False | 是否为测试模式 |

#### 主要方法

##### `load_annotations(self, ann_file)`

**功能**: 加载标注文件

**参数**:
- `ann_file` (str): 标注文件路径

**返回**: list - 数据信息列表

**示例**:
```python
data_infos = dataset.load_annotations('annotations/train.json')
```

##### `get_data_info(self, index)`

**功能**: 获取指定索引的数据信息

**参数**:
- `index` (int): 数据索引

**返回**: dict - 数据信息字典

**示例**:
```python
data_info = dataset.get_data_info(0)
```

##### `get_ann_info(self, index)`

**功能**: 获取标注信息

**参数**:
- `index` (int): 数据索引

**返回**: dict - 标注信息字典

**示例**:
```python
ann_info = dataset.get_ann_info(0)
```

##### `prepare_train_data(self, index)`

**功能**: 准备训练数据

**参数**:
- `index` (int): 数据索引

**返回**: dict - 处理后的训练数据

**示例**:
```python
train_data = dataset.prepare_train_data(0)
```

##### `prepare_test_data(self, index)`

**功能**: 准备测试数据

**参数**:
- `index` (int): 数据索引

**返回**: dict - 处理后的测试数据

**示例**:
```python
test_data = dataset.prepare_test_data(0)
```

##### `evaluate(self, results, metric='mAP', logger=None, **kwargs)`

**功能**: 评估结果

**参数**:
- `results` (list): 预测结果
- `metric` (str): 评估指标
- `logger`: 日志记录器
- `**kwargs`: 其他参数

**返回**: dict - 评估结果字典

**示例**:
```python
eval_results = dataset.evaluate(predictions, metric='mAP')
```

---

## ⚙️ 配置API

### 模型配置参数

#### model_params

4DGS模型核心参数配置。

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `sh_degree` | int | 3 | 球谐函数度数 |
| `source_path` | str | "" | 数据源路径 |
| `model_path` | str | "" | 模型保存路径 |
| `images` | str | "images" | 图像文件夹名称 |
| `resolution` | int | -1 | 图像分辨率（-1表示原始分辨率） |
| `white_background` | bool | False | 是否使用白色背景 |
| `data_device` | str | "cuda" | 数据设备 |
| `eval` | bool | False | 是否为评估模式 |

**示例配置**:
```python
model_params = dict(
    sh_degree=3,
    source_path="data/4dgs/scene1",
    model_path="work_dirs/4dgs/models",
    images="images",
    resolution=800,
    white_background=False,
    data_device="cuda",
    eval=False
)
```

#### optimization_params

优化器参数配置。

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `iterations` | int | 30000 | 训练迭代次数 |
| `position_lr_init` | float | 0.00016 | 位置学习率初始值 |
| `position_lr_final` | float | 0.0000016 | 位置学习率最终值 |
| `position_lr_delay_mult` | float | 0.01 | 位置学习率延迟倍数 |
| `position_lr_max_steps` | int | 30000 | 位置学习率最大步数 |
| `feature_lr` | float | 0.0025 | 特征学习率 |
| `opacity_lr` | float | 0.05 | 透明度学习率 |
| `scaling_lr` | float | 0.005 | 缩放学习率 |
| `rotation_lr` | float | 0.001 | 旋转学习率 |
| `percent_dense` | float | 0.01 | 密化百分比 |
| `lambda_dssim` | float | 0.2 | DSSIM损失权重 |
| `densification_interval` | int | 100 | 密化间隔 |
| `opacity_reset_interval` | int | 3000 | 透明度重置间隔 |
| `densify_from_iter` | int | 500 | 开始密化的迭代次数 |
| `densify_until_iter` | int | 15000 | 结束密化的迭代次数 |
| `densify_grad_threshold` | float | 0.0002 | 密化梯度阈值 |

**示例配置**:
```python
optimization_params = dict(
    iterations=30000,
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    feature_lr=0.0025,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    lambda_dssim=0.2,
    densification_interval=100,
    densify_from_iter=500,
    densify_until_iter=15000
)
```

#### pipeline_params

渲染管道参数配置。

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `convert_SHs_python` | bool | False | 是否使用Python转换球谐函数 |
| `compute_cov3D_python` | bool | False | 是否使用Python计算3D协方差 |
| `debug` | bool | False | 是否开启调试模式 |

**示例配置**:
```python
pipeline_params = dict(
    convert_SHs_python=False,
    compute_cov3D_python=False,
    debug=False
)
```

---

## 🛠️ 工具函数API

### 数据转换工具

#### `convert_mmdet3d_to_4dgs(data_dict)`

**功能**: 将MMDet3D数据格式转换为4DGS格式

**参数**:
- `data_dict` (dict): MMDet3D格式的数据字典

**返回**: dict - 4DGS格式的数据字典

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.utils import convert_mmdet3d_to_4dgs

mmdet3d_data = {
    'img': torch.tensor(...),
    'img_metas': [{'filename': 'image.jpg', 'camera_intrinsics': [...]}]
}
fourdgs_data = convert_mmdet3d_to_4dgs(mmdet3d_data)
```

#### `convert_4dgs_to_mmdet3d(data_dict)`

**功能**: 将4DGS数据格式转换为MMDet3D格式

**参数**:
- `data_dict` (dict): 4DGS格式的数据字典

**返回**: dict - MMDet3D格式的数据字典

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.utils import convert_4dgs_to_mmdet3d

fourdgs_data = {
    'image': torch.tensor(...),
    'camera': {...}
}
mmdet3d_data = convert_4dgs_to_mmdet3d(fourdgs_data)
```

### 配置合并工具

#### `merge_hparams(base_config, override_config)`

**功能**: 合并配置参数

**参数**:
- `base_config` (dict): 基础配置
- `override_config` (dict): 覆盖配置

**返回**: dict - 合并后的配置

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.utils import merge_hparams

base_cfg = {'iterations': 30000, 'lr': 0.001}
override_cfg = {'lr': 0.002, 'batch_size': 4}
merged_cfg = merge_hparams(base_cfg, override_cfg)
# 结果: {'iterations': 30000, 'lr': 0.002, 'batch_size': 4}
```

---

## 🎨 渲染器API

### GaussianRenderer

高斯渲染器，负责将3D高斯点云渲染为2D图像。

```python
class GaussianRenderer:
    """
    高斯渲染器
    
    Args:
        raster_settings: 光栅化设置
    """
```

#### 主要方法

##### `render(viewpoint_camera, pc, pipe, bg_color)`

**功能**: 渲染图像

**参数**:
- `viewpoint_camera`: 视点相机对象
- `pc`: 点云对象
- `pipe`: 渲染管道设置
- `bg_color`: 背景颜色

**返回**: dict - 渲染结果字典

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.gaussian_renderer import render

render_result = render(
    viewpoint_camera=camera,
    pc=gaussians,
    pipe=pipeline_params,
    bg_color=background_color
)
```

---

## 🏗️ 场景管理API

### Scene

场景管理类，负责管理3D场景数据和相机信息。

```python
class Scene:
    """
    场景管理类
    
    Args:
        args: 场景参数
        gaussians: 高斯模型
        load_iteration: 加载的迭代次数
        shuffle: 是否打乱数据
        resolution_scales: 分辨率缩放比例
    """
```

#### 主要方法

##### `__init__(self, args, gaussians, load_iteration=-1, shuffle=True, resolution_scales=[1.0])`

**功能**: 初始化场景

**参数**:
- `args`: 场景参数
- `gaussians`: 高斯模型对象
- `load_iteration` (int): 加载的迭代次数
- `shuffle` (bool): 是否打乱数据
- `resolution_scales` (list): 分辨率缩放比例

**返回**: None

##### `getTrainCameras(self, scale=1.0)`

**功能**: 获取训练相机列表

**参数**:
- `scale` (float): 缩放比例

**返回**: list - 训练相机列表

**示例**:
```python
train_cameras = scene.getTrainCameras(scale=1.0)
```

##### `getTestCameras(self, scale=1.0)`

**功能**: 获取测试相机列表

**参数**:
- `scale` (float): 缩放比例

**返回**: list - 测试相机列表

**示例**:
```python
test_cameras = scene.getTestCameras(scale=1.0)
```

### GaussianModel

高斯模型类，管理3D高斯点的属性和操作。

```python
class GaussianModel:
    """
    高斯模型类
    
    Args:
        sh_degree: 球谐函数度数
    """
```

#### 主要属性

| 属性名 | 类型 | 描述 |
|--------|------|------|
| `_xyz` | torch.Tensor | 3D位置坐标 |
| `_features_dc` | torch.Tensor | 直流特征 |
| `_features_rest` | torch.Tensor | 其余特征 |
| `_scaling` | torch.Tensor | 缩放参数 |
| `_rotation` | torch.Tensor | 旋转参数 |
| `_opacity` | torch.Tensor | 透明度参数 |

#### 主要方法

##### `create_from_pcd(self, pcd, spatial_lr_scale)`

**功能**: 从点云创建高斯模型

**参数**:
- `pcd`: 点云数据
- `spatial_lr_scale` (float): 空间学习率缩放

**返回**: None

**示例**:
```python
gaussians.create_from_pcd(point_cloud, spatial_lr_scale=1.0)
```

##### `training_setup(self, training_args)`

**功能**: 设置训练参数

**参数**:
- `training_args`: 训练参数

**返回**: None

**示例**:
```python
gaussians.training_setup(optimization_params)
```

##### `save_ply(self, path)`

**功能**: 保存为PLY文件

**参数**:
- `path` (str): 保存路径

**返回**: None

**示例**:
```python
gaussians.save_ply('output/gaussians.ply')
```

##### `load_ply(self, path)`

**功能**: 从PLY文件加载

**参数**:
- `path` (str): 文件路径

**返回**: None

**示例**:
```python
gaussians.load_ply('input/gaussians.ply')
```

---

## 🚀 训练API

### 训练函数

#### `train_4dgs_model(config_path, work_dir=None, resume_from=None)`

**功能**: 训练4DGS模型

**参数**:
- `config_path` (str): 配置文件路径
- `work_dir` (str): 工作目录
- `resume_from` (str): 恢复训练的检查点路径

**返回**: None

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.apis import train_4dgs_model

train_4dgs_model(
    config_path='projects/configs/4DGS/4dgs_base.py',
    work_dir='work_dirs/4dgs_experiment',
    resume_from='work_dirs/4dgs_experiment/latest.pth'
)
```

#### `build_4dgs_model(config)`

**功能**: 构建4DGS模型

**参数**:
- `config` (dict): 模型配置

**返回**: FourDGSModel - 4DGS模型实例

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.apis import build_4dgs_model

model = build_4dgs_model(config.model)
```

---

## 🔍 推理API

### 推理函数

#### `inference_4dgs_model(model, data_loader, show=False, out_dir=None)`

**功能**: 4DGS模型推理

**参数**:
- `model`: 4DGS模型实例
- `data_loader`: 数据加载器
- `show` (bool): 是否显示结果
- `out_dir` (str): 输出目录

**返回**: list - 推理结果列表

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.apis import inference_4dgs_model

results = inference_4dgs_model(
    model=trained_model,
    data_loader=test_loader,
    show=True,
    out_dir='results/inference'
)
```

#### `render_novel_view(model, camera_params, output_path=None)`

**功能**: 渲染新视角

**参数**:
- `model`: 4DGS模型实例
- `camera_params` (dict): 相机参数
- `output_path` (str): 输出路径

**返回**: torch.Tensor - 渲染图像

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.apis import render_novel_view

camera_params = {
    'position': [0, 0, 5],
    'rotation': [0, 0, 0],
    'fov': 60.0,
    'width': 800,
    'height': 600
}

rendered_image = render_novel_view(
    model=trained_model,
    camera_params=camera_params,
    output_path='output/novel_view.jpg'
)
```

---

## 📋 错误代码参考

### 常见错误代码

| 错误代码 | 描述 | 解决方案 |
|----------|------|----------|
| `4DGS_001` | 模型初始化失败 | 检查模型参数配置 |
| `4DGS_002` | 数据加载错误 | 验证数据路径和格式 |
| `4DGS_003` | 渲染失败 | 检查GPU内存和相机参数 |
| `4DGS_004` | 检查点加载失败 | 验证检查点文件完整性 |
| `4DGS_005` | 配置文件解析错误 | 检查配置文件语法 |

### 异常类

#### `FourDGSError`

**功能**: 4DGS模块基础异常类

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.exceptions import FourDGSError

try:
    model.train()
except FourDGSError as e:
    print(f"4DGS错误: {e}")
```

#### `FourDGSConfigError`

**功能**: 配置错误异常类

**示例**:
```python
from projects.mmdet3d_plugin.4DGS.exceptions import FourDGSConfigError

try:
    config = load_config('invalid_config.py')
except FourDGSConfigError as e:
    print(f"配置错误: {e}")
```

---

## 📝 使用示例

### 完整训练示例

```python
#!/usr/bin/env python3
"""
完整的4DGS训练示例
"""

import torch
from mmcv import Config
from mmdet3d.apis import train_detector
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# 1. 加载配置
config_path = 'projects/configs/4DGS/4dgs_base.py'
cfg = Config.fromfile(config_path)

# 2. 构建数据集
train_dataset = build_dataset(cfg.data.train)
val_dataset = build_dataset(cfg.data.val)

# 3. 构建模型
model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

# 4. 开始训练
train_detector(
    model=model,
    dataset=train_dataset,
    cfg=cfg,
    distributed=False,
    validate=True,
    timestamp='20240101_120000'
)
```

### 完整推理示例

```python
#!/usr/bin/env python3
"""
完整的4DGS推理示例
"""

import torch
from mmcv import Config
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint

# 1. 加载配置和模型
config_path = 'projects/configs/4DGS/4dgs_base.py'
checkpoint_path = 'work_dirs/4dgs_base/latest.pth'

cfg = Config.fromfile(config_path)
model = build_model(cfg.model, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

# 2. 构建测试数据集
test_dataset = build_dataset(cfg.data.test)
test_loader = build_dataloader(
    test_dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    dist=False,
    shuffle=False
)

# 3. 执行推理
model = model.cuda()
model.eval()
results = single_gpu_test(model, test_loader)

# 4. 评估结果
eval_results = test_dataset.evaluate(results, metric='mAP')
print(f"评估结果: {eval_results}")
```

---

## 📞 技术支持

如需更多技术支持，请参考：

- [集成指南](4dgs_integration_guide.md)
- [实施步骤](4dgs_implementation_steps.md)
- [故障排除](4dgs_troubleshooting.md)
- [技术分析](4dgs_technical_analysis.md)

---

*本API参考手册将随着项目发展持续更新，请关注最新版本。*