# 4DGS使用示例和最佳实践

## 🎯 概述

本文档提供了4DGS模块的详细使用示例和最佳实践指南，帮助开发者快速上手并高效使用4DGS功能。

## 📋 目录

- [快速开始示例](#快速开始示例)
- [训练示例](#训练示例)
- [推理示例](#推理示例)
- [数据处理示例](#数据处理示例)
- [配置最佳实践](#配置最佳实践)
- [性能优化实践](#性能优化实践)
- [调试和监控实践](#调试和监控实践)
- [生产环境部署](#生产环境部署)
- [常见使用场景](#常见使用场景)

---

## 🚀 快速开始示例

### 示例1：最简单的4DGS训练

```python
#!/usr/bin/env python3
"""
最简单的4DGS训练示例
文件位置: examples/quick_start/simple_train.py
"""

import os
import sys
sys.path.append('.')

from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmdet3d.apis import train_model

def simple_4dgs_training():
    """最简单的4DGS训练流程"""
    
    # 1. 加载配置
    config_path = 'projects/configs/4DGS/4dgs_base.py'
    cfg = Config.fromfile(config_path)
    
    # 2. 构建模型
    model = build_model(cfg.model)
    
    # 3. 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    
    # 4. 开始训练
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=None,
        meta=dict()
    )

if __name__ == '__main__':
    simple_4dgs_training()
```

### 示例2：快速推理

```python
#!/usr/bin/env python3
"""
快速推理示例
文件位置: examples/quick_start/simple_inference.py
"""

import torch
import numpy as np
from mmcv import Config
from mmdet3d.models import build_model

def quick_inference():
    """快速推理示例"""
    
    # 1. 加载配置和模型
    cfg = Config.fromfile('projects/configs/4DGS/4dgs_base.py')
    model = build_model(cfg.model)
    
    # 2. 加载预训练权重
    checkpoint_path = 'checkpoints/4dgs_latest.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 3. 准备输入数据
    camera_params = {
        'camera_center': np.array([0.0, 0.0, 0.0]),
        'camera_rotation': np.eye(3),
        'fov': 60.0,
        'width': 800,
        'height': 600
    }
    
    # 4. 执行推理
    with torch.no_grad():
        result = model.render(camera_params)
        rendered_image = result['rendered_image']
    
    # 5. 保存结果
    import cv2
    cv2.imwrite('output/rendered_image.png', rendered_image.cpu().numpy())
    print("推理完成，结果保存到 output/rendered_image.png")

if __name__ == '__main__':
    quick_inference()
```

---

## 🏋️ 训练示例

### 示例3：完整的训练流程

```python
#!/usr/bin/env python3
"""
完整的4DGS训练流程示例
文件位置: examples/training/complete_training.py
"""

import os
import torch
import logging
from datetime import datetime
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.apis import set_random_seed
from mmdet3d.utils import get_root_logger

class FourDGSTrainer:
    """4DGS训练器类"""
    
    def __init__(self, config_path, work_dir=None):
        """
        初始化训练器
        
        Args:
            config_path (str): 配置文件路径
            work_dir (str): 工作目录
        """
        self.cfg = Config.fromfile(config_path)
        self.work_dir = work_dir or f'work_dirs/4dgs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # 创建工作目录
        os.makedirs(self.work_dir, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 设置随机种子
        set_random_seed(42, deterministic=False)
        
    def _setup_logger(self):
        """设置日志记录器"""
        log_file = os.path.join(self.work_dir, 'training.log')
        logger = get_root_logger(log_file=log_file, log_level=logging.INFO)
        return logger
    
    def build_model(self):
        """构建模型"""
        self.logger.info("构建4DGS模型...")
        model = build_model(self.cfg.model)
        
        # 初始化权重
        model.init_weights()
        
        # 移动到GPU
        if torch.cuda.is_available():
            model = model.cuda()
            self.logger.info(f"模型已移动到GPU: {torch.cuda.get_device_name()}")
        
        return model
    
    def build_datasets(self):
        """构建数据集"""
        self.logger.info("构建训练和验证数据集...")
        
        # 训练数据集
        train_dataset = build_dataset(self.cfg.data.train)
        self.logger.info(f"训练数据集大小: {len(train_dataset)}")
        
        # 验证数据集
        val_dataset = build_dataset(self.cfg.data.val)
        self.logger.info(f"验证数据集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def build_dataloaders(self, train_dataset, val_dataset):
        """构建数据加载器"""
        # 训练数据加载器
        train_dataloader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.cfg.data.samples_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=True,
            seed=42
        )
        
        # 验证数据加载器
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False
        )
        
        return train_dataloader, val_dataloader
    
    def setup_optimizer(self, model):
        """设置优化器"""
        from mmdet3d.core.optimizer import build_optimizer
        
        optimizer = build_optimizer(model, self.cfg.optimizer)
        self.logger.info(f"优化器: {type(optimizer).__name__}")
        
        return optimizer
    
    def train(self):
        """执行训练"""
        self.logger.info("开始4DGS训练...")
        
        # 构建组件
        model = self.build_model()
        train_dataset, val_dataset = self.build_datasets()
        train_dataloader, val_dataloader = self.build_dataloaders(train_dataset, val_dataset)
        optimizer = self.setup_optimizer(model)
        
        # 训练循环
        model.train()
        total_iterations = self.cfg.optimization_params.iterations
        
        for iteration in range(total_iterations):
            # 获取数据
            for batch_idx, data_batch in enumerate(train_dataloader):
                # 前向传播
                losses = model.forward_train(**data_batch)
                
                # 计算总损失
                total_loss = sum(losses.values())
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # 日志记录
                if iteration % 100 == 0:
                    self.logger.info(
                        f"Iter {iteration}/{total_iterations}, "
                        f"Loss: {total_loss.item():.4f}"
                    )
                
                # 验证
                if iteration % 1000 == 0 and iteration > 0:
                    self.validate(model, val_dataloader, iteration)
                
                # 保存检查点
                if iteration % 5000 == 0 and iteration > 0:
                    self.save_checkpoint(model, optimizer, iteration)
                
                iteration += 1
                if iteration >= total_iterations:
                    break
            
            if iteration >= total_iterations:
                break
        
        # 保存最终模型
        self.save_checkpoint(model, optimizer, total_iterations, is_final=True)
        self.logger.info("训练完成!")
    
    def validate(self, model, val_dataloader, iteration):
        """验证模型"""
        model.eval()
        total_val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data_batch in val_dataloader:
                losses = model.forward_train(**data_batch)
                total_val_loss += sum(losses.values()).item()
                num_batches += 1
        
        avg_val_loss = total_val_loss / num_batches
        self.logger.info(f"Validation at iter {iteration}, Avg Loss: {avg_val_loss:.4f}")
        
        model.train()
    
    def save_checkpoint(self, model, optimizer, iteration, is_final=False):
        """保存检查点"""
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': iteration,
            'config': self.cfg.pretty_text
        }
        
        if is_final:
            checkpoint_path = os.path.join(self.work_dir, 'final_model.pth')
        else:
            checkpoint_path = os.path.join(self.work_dir, f'iter_{iteration}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存: {checkpoint_path}")

def main():
    """主函数"""
    trainer = FourDGSTrainer(
        config_path='projects/configs/4DGS/4dgs_base.py',
        work_dir='work_dirs/4dgs_training'
    )
    trainer.train()

if __name__ == '__main__':
    main()
```

### 示例4：分布式训练

```python
#!/usr/bin/env python3
"""
分布式训练示例
文件位置: examples/training/distributed_training.py
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader

def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def distributed_train(rank, world_size, config_path):
    """分布式训练函数"""
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 加载配置
    cfg = Config.fromfile(config_path)
    
    # 构建模型
    model = build_model(cfg.model)
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # 构建数据集
    train_dataset = build_dataset(cfg.data.train)
    
    # 构建分布式数据加载器
    train_dataloader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=True,
        seed=42
    )
    
    # 训练循环
    model.train()
    for epoch in range(cfg.total_epochs):
        for batch_idx, data_batch in enumerate(train_dataloader):
            # 训练步骤
            losses = model(**data_batch)
            total_loss = sum(losses.values())
            
            # 反向传播
            total_loss.backward()
            
            if rank == 0:  # 只在主进程打印日志
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")
    
    # 清理
    cleanup_distributed()

def main():
    """主函数"""
    world_size = torch.cuda.device_count()
    config_path = 'projects/configs/4DGS/4dgs_base.py'
    
    mp.spawn(
        distributed_train,
        args=(world_size, config_path),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
```

---

## 🔍 推理示例

### 示例5：批量推理

```python
#!/usr/bin/env python3
"""
批量推理示例
文件位置: examples/inference/batch_inference.py
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from mmcv import Config
from mmdet3d.models import build_model

class FourDGSInference:
    """4DGS推理器"""
    
    def __init__(self, config_path, checkpoint_path):
        """
        初始化推理器
        
        Args:
            config_path (str): 配置文件路径
            checkpoint_path (str): 检查点文件路径
        """
        self.cfg = Config.fromfile(config_path)
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path):
        """加载模型"""
        # 构建模型
        model = build_model(self.cfg.model)
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        
        # 设置为评估模式
        model.eval()
        
        # 移动到GPU
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    def render_single_view(self, camera_params):
        """
        渲染单个视角
        
        Args:
            camera_params (dict): 相机参数
            
        Returns:
            dict: 渲染结果
        """
        with torch.no_grad():
            result = self.model.render(camera_params)
        return result
    
    def render_trajectory(self, camera_trajectory, output_dir):
        """
        渲染相机轨迹
        
        Args:
            camera_trajectory (list): 相机轨迹参数列表
            output_dir (str): 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, camera_params in enumerate(tqdm(camera_trajectory, desc="渲染进度")):
            # 渲染当前视角
            result = self.render_single_view(camera_params)
            
            # 保存图像
            rendered_image = result['rendered_image'].cpu().numpy()
            output_path = os.path.join(output_dir, f'frame_{i:06d}.png')
            
            import cv2
            cv2.imwrite(output_path, rendered_image)
        
        print(f"渲染完成，共{len(camera_trajectory)}帧，保存到{output_dir}")
    
    def render_360_video(self, center, radius, height, num_frames, output_dir):
        """
        渲染360度环绕视频
        
        Args:
            center (np.ndarray): 环绕中心点
            radius (float): 环绕半径
            height (float): 相机高度
            num_frames (int): 帧数
            output_dir (str): 输出目录
        """
        # 生成环绕轨迹
        angles = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        camera_trajectory = []
        
        for angle in angles:
            # 计算相机位置
            camera_pos = center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ])
            
            # 计算相机朝向（朝向中心）
            look_at = center - camera_pos
            look_at = look_at / np.linalg.norm(look_at)
            
            # 构建旋转矩阵
            up = np.array([0, 0, 1])
            right = np.cross(look_at, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, look_at)
            
            rotation_matrix = np.column_stack([right, up, -look_at])
            
            # 相机参数
            camera_params = {
                'camera_center': camera_pos,
                'camera_rotation': rotation_matrix,
                'fov': 60.0,
                'width': 1920,
                'height': 1080
            }
            
            camera_trajectory.append(camera_params)
        
        # 渲染轨迹
        self.render_trajectory(camera_trajectory, output_dir)
        
        # 生成视频
        self._create_video(output_dir, f"{output_dir}/360_video.mp4")
    
    def _create_video(self, image_dir, output_video_path, fps=30):
        """从图像序列创建视频"""
        import cv2
        import glob
        
        # 获取图像文件列表
        image_files = sorted(glob.glob(os.path.join(image_dir, "frame_*.png")))
        
        if not image_files:
            print("未找到图像文件")
            return
        
        # 读取第一张图像获取尺寸
        first_image = cv2.imread(image_files[0])
        height, width, _ = first_image.shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 写入所有帧
        for image_file in tqdm(image_files, desc="创建视频"):
            image = cv2.imread(image_file)
            video_writer.write(image)
        
        video_writer.release()
        print(f"视频已保存: {output_video_path}")

def main():
    """主函数"""
    # 初始化推理器
    inference = FourDGSInference(
        config_path='projects/configs/4DGS/4dgs_base.py',
        checkpoint_path='checkpoints/4dgs_final.pth'
    )
    
    # 渲染360度视频
    center = np.array([0.0, 0.0, 0.0])
    radius = 5.0
    height = 2.0
    num_frames = 120
    
    inference.render_360_video(
        center=center,
        radius=radius,
        height=height,
        num_frames=num_frames,
        output_dir='output/360_video'
    )

if __name__ == '__main__':
    main()
```

### 示例6：交互式推理

```python
#!/usr/bin/env python3
"""
交互式推理示例
文件位置: examples/inference/interactive_inference.py
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from mmcv import Config
from mmdet3d.models import build_model

class InteractiveFourDGSViewer:
    """交互式4DGS查看器"""
    
    def __init__(self, config_path, checkpoint_path):
        """初始化查看器"""
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # 加载模型
        self.model = self._load_model()
        
        # 初始化相机参数
        self.camera_params = {
            'camera_center': np.array([0.0, 0.0, 5.0]),
            'camera_rotation': np.eye(3),
            'fov': 60.0,
            'width': 800,
            'height': 600
        }
        
        # 创建GUI
        self.setup_gui()
        
    def _load_model(self):
        """加载模型"""
        cfg = Config.fromfile(self.config_path)
        model = build_model(cfg.model)
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    def setup_gui(self):
        """设置GUI界面"""
        self.root = tk.Tk()
        self.root.title("4DGS交互式查看器")
        self.root.geometry("1200x800")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 右侧显示区域
        display_frame = ttk.LabelFrame(main_frame, text="渲染结果")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 设置控制面板
        self.setup_control_panel(control_frame)
        
        # 设置显示区域
        self.setup_display_area(display_frame)
        
        # 初始渲染
        self.render_and_display()
    
    def setup_control_panel(self, parent):
        """设置控制面板"""
        # 相机位置控制
        pos_frame = ttk.LabelFrame(parent, text="相机位置")
        pos_frame.pack(fill=tk.X, pady=5)
        
        # X位置
        ttk.Label(pos_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x_var = tk.DoubleVar(value=0.0)
        x_scale = ttk.Scale(pos_frame, from_=-10, to=10, variable=self.x_var, 
                           orient=tk.HORIZONTAL, command=self.on_camera_change)
        x_scale.grid(row=0, column=1, sticky=tk.EW)
        
        # Y位置
        ttk.Label(pos_frame, text="Y:").grid(row=1, column=0, sticky=tk.W)
        self.y_var = tk.DoubleVar(value=0.0)
        y_scale = ttk.Scale(pos_frame, from_=-10, to=10, variable=self.y_var,
                           orient=tk.HORIZONTAL, command=self.on_camera_change)
        y_scale.grid(row=1, column=1, sticky=tk.EW)
        
        # Z位置
        ttk.Label(pos_frame, text="Z:").grid(row=2, column=0, sticky=tk.W)
        self.z_var = tk.DoubleVar(value=5.0)
        z_scale = ttk.Scale(pos_frame, from_=1, to=20, variable=self.z_var,
                           orient=tk.HORIZONTAL, command=self.on_camera_change)
        z_scale.grid(row=2, column=1, sticky=tk.EW)
        
        pos_frame.columnconfigure(1, weight=1)
        
        # 相机旋转控制
        rot_frame = ttk.LabelFrame(parent, text="相机旋转")
        rot_frame.pack(fill=tk.X, pady=5)
        
        # 俯仰角
        ttk.Label(rot_frame, text="俯仰:").grid(row=0, column=0, sticky=tk.W)
        self.pitch_var = tk.DoubleVar(value=0.0)
        pitch_scale = ttk.Scale(rot_frame, from_=-90, to=90, variable=self.pitch_var,
                               orient=tk.HORIZONTAL, command=self.on_camera_change)
        pitch_scale.grid(row=0, column=1, sticky=tk.EW)
        
        # 偏航角
        ttk.Label(rot_frame, text="偏航:").grid(row=1, column=0, sticky=tk.W)
        self.yaw_var = tk.DoubleVar(value=0.0)
        yaw_scale = ttk.Scale(rot_frame, from_=-180, to=180, variable=self.yaw_var,
                             orient=tk.HORIZONTAL, command=self.on_camera_change)
        yaw_scale.grid(row=1, column=1, sticky=tk.EW)
        
        rot_frame.columnconfigure(1, weight=1)
        
        # 视野控制
        fov_frame = ttk.LabelFrame(parent, text="视野角度")
        fov_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(fov_frame, text="FOV:").grid(row=0, column=0, sticky=tk.W)
        self.fov_var = tk.DoubleVar(value=60.0)
        fov_scale = ttk.Scale(fov_frame, from_=10, to=120, variable=self.fov_var,
                             orient=tk.HORIZONTAL, command=self.on_camera_change)
        fov_scale.grid(row=0, column=1, sticky=tk.EW)
        
        fov_frame.columnconfigure(1, weight=1)
        
        # 操作按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="重置相机", command=self.reset_camera).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="保存图像", command=self.save_image).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="录制视频", command=self.record_video).pack(fill=tk.X, pady=2)
    
    def setup_display_area(self, parent):
        """设置显示区域"""
        self.canvas = tk.Canvas(parent, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
    
    def on_camera_change(self, event=None):
        """相机参数改变时的回调"""
        # 更新相机位置
        self.camera_params['camera_center'] = np.array([
            self.x_var.get(),
            self.y_var.get(),
            self.z_var.get()
        ])
        
        # 更新相机旋转
        pitch = np.radians(self.pitch_var.get())
        yaw = np.radians(self.yaw_var.get())
        
        # 构建旋转矩阵
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw * cos_pitch, sin_yaw * sin_pitch],
            [sin_yaw, cos_yaw * cos_pitch, -cos_yaw * sin_pitch],
            [0, sin_pitch, cos_pitch]
        ])
        
        self.camera_params['camera_rotation'] = rotation_matrix
        
        # 更新FOV
        self.camera_params['fov'] = self.fov_var.get()
        
        # 重新渲染
        self.render_and_display()
    
    def render_and_display(self):
        """渲染并显示结果"""
        try:
            # 渲染
            with torch.no_grad():
                result = self.model.render(self.camera_params)
                rendered_image = result['rendered_image'].cpu().numpy()
            
            # 转换为PIL图像
            if rendered_image.dtype != np.uint8:
                rendered_image = (rendered_image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(rendered_image)
            
            # 转换为Tkinter格式
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # 显示在画布上
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("渲染错误", f"渲染失败: {str(e)}")
    
    def reset_camera(self):
        """重置相机参数"""
        self.x_var.set(0.0)
        self.y_var.set(0.0)
        self.z_var.set(5.0)
        self.pitch_var.set(0.0)
        self.yaw_var.set(0.0)
        self.fov_var.set(60.0)
    
    def save_image(self):
        """保存当前图像"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # 重新渲染高分辨率图像
                high_res_params = self.camera_params.copy()
                high_res_params['width'] = 1920
                high_res_params['height'] = 1080
                
                with torch.no_grad():
                    result = self.model.render(high_res_params)
                    rendered_image = result['rendered_image'].cpu().numpy()
                
                if rendered_image.dtype != np.uint8:
                    rendered_image = (rendered_image * 255).astype(np.uint8)
                
                cv2.imwrite(filename, rendered_image)
                messagebox.showinfo("保存成功", f"图像已保存到: {filename}")
                
            except Exception as e:
                messagebox.showerror("保存错误", f"保存失败: {str(e)}")
    
    def record_video(self):
        """录制视频"""
        # 这里可以实现视频录制功能
        messagebox.showinfo("功能开发中", "视频录制功能正在开发中...")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()

def main():
    """主函数"""
    viewer = InteractiveFourDGSViewer(
        config_path='projects/configs/4DGS/4dgs_base.py',
        checkpoint_path='checkpoints/4dgs_final.pth'
    )
    viewer.run()

if __name__ == '__main__':
    main()
```

---

## 📊 数据处理示例

### 示例7：数据预处理

```python
#!/usr/bin/env python3
"""
数据预处理示例
文件位置: examples/data_processing/preprocess_data.py
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

class FourDGSDataPreprocessor:
    """4DGS数据预处理器"""
    
    def __init__(self, input_dir, output_dir):
        """
        初始化预处理器
        
        Args:
            input_dir (str): 输入数据目录
            output_dir (str): 输出数据目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_images(self, target_size=(800, 600)):
        """
        处理图像数据
        
        Args:
            target_size (tuple): 目标图像尺寸
        """
        print("处理图像数据...")
        
        # 创建输出目录
        images_dir = self.output_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(self.input_dir.glob('**/*.jpg')) + \
                     list(self.input_dir.glob('**/*.png')) + \
                     list(self.input_dir.glob('**/*.jpeg'))
        
        processed_images = []
        
        for img_path in tqdm(image_files, desc="处理图像"):
            try:
                # 读取图像
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"无法读取图像: {img_path}")
                    continue
                
                # 调整尺寸
                resized_image = cv2.resize(image, target_size)
                
                # 保存处理后的图像
                output_path = images_dir / f"{img_path.stem}_processed.jpg"
                cv2.imwrite(str(output_path), resized_image)
                
                # 记录图像信息
                processed_images.append({
                    'original_path': str(img_path),
                    'processed_path': str(output_path),
                    'original_size': image.shape[:2],
                    'processed_size': target_size
                })
                
            except Exception as e:
                print(f"处理图像失败 {img_path}: {e}")
        
        # 保存图像信息
        with open(self.output_dir / 'image_info.json', 'w') as f:
            json.dump(processed_images, f, indent=2)
        
        print(f"图像处理完成，共处理{len(processed_images)}张图像")
        return processed_images
    
    def process_camera_poses(self, poses_file):
        """
        处理相机位姿数据
        
        Args:
            poses_file (str): 位姿文件路径
        """
        print("处理相机位姿数据...")
        
        # 读取原始位姿数据
        with open(poses_file, 'r') as f:
            raw_poses = json.load(f)
        
        processed_poses = []
        
        for pose_data in tqdm(raw_poses, desc="处理位姿"):
            try:
                # 提取相机参数
                camera_matrix = np.array(pose_data['camera_matrix'])
                rotation_matrix = np.array(pose_data['rotation'])
                translation_vector = np.array(pose_data['translation'])
                
                # 标准化处理
                # 确保旋转矩阵是正交的
                U, _, Vt = np.linalg.svd(rotation_matrix)
                rotation_matrix = U @ Vt
                
                # 计算相机中心
                camera_center = -rotation_matrix.T @ translation_vector
                
                # 构建标准格式
                processed_pose = {
                    'image_name': pose_data['image_name'],
                    'camera_center': camera_center.tolist(),
                    'camera_rotation': rotation_matrix.tolist(),
                    'camera_matrix': camera_matrix.tolist(),
                    'fov': self._calculate_fov(camera_matrix),
                    'width': pose_data.get('width', 800),
                    'height': pose_data.get('height', 600)
                }
                
                processed_poses.append(processed_pose)
                
            except Exception as e:
                print(f"处理位姿失败: {e}")
        
        # 保存处理后的位姿
        with open(self.output_dir / 'camera_poses.json', 'w') as f:
            json.dump(processed_poses, f, indent=2)
        
        print(f"位姿处理完成，共处理{len(processed_poses)}个位姿")
        return processed_poses
    
    def _calculate_fov(self, camera_matrix):
        """计算视野角度"""
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        width = camera_matrix[0, 2] * 2
        height = camera_matrix[1, 2] * 2
        
        fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
        
        return (fov_x + fov_y) / 2  # 返回平均FOV
    
    def create_train_val_split(self, split_ratio=0.8):
        """
        创建训练/验证数据划分
        
        Args:
            split_ratio (float): 训练数据比例
        """
        print("创建训练/验证数据划分...")
        
        # 读取处理后的数据
        with open(self.output_dir / 'camera_poses.json', 'r') as f:
            poses = json.load(f)
        
        # 随机划分
        np.random.seed(42)
        indices = np.random.permutation(len(poses))
        split_point = int(len(poses) * split_ratio)
        
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        
        # 创建训练集
        train_poses = [poses[i] for i in train_indices]
        with open(self.output_dir / 'train_poses.json', 'w') as f:
            json.dump(train_poses, f, indent=2)
        
        # 创建验证集
        val_poses = [poses[i] for i in val_indices]
        with open(self.output_dir / 'val_poses.json', 'w') as f:
            json.dump(val_poses, f, indent=2)
        
        print(f"数据划分完成: 训练集{len(train_poses)}张，验证集{len(val_poses)}张")
        
        return train_poses, val_poses
    
    def generate_mmdet3d_config(self, class_names=None):
        """
        生成MMDet3D格式的配置文件
        
        Args:
            class_names (list): 类别名称列表
        """
        if class_names is None:
            class_names = ['object']
        
        config_template = f'''
# 4DGS数据集配置文件
_base_ = ['../_base_/default_runtime.py']

# 数据集设置
dataset_type = 'FourDGSDataset'
data_root = '{self.output_dir}'
class_names = {class_names}

# 数据管道
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='Resize', img_scale=(800, 600), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(800, 600),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['img'])
         ])
]

# 数据配置
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/train_poses.json',
        img_prefix=data_root + '/images/',
        classes=class_names,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/val_poses.json',
        img_prefix=data_root + '/images/',
        classes=class_names,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/val_poses.json',
        img_prefix=data_root + '/images/',
        classes=class_names,
        pipeline=test_pipeline
    )
)

# 模型配置
model = dict(
    type='FourDGSModel',
    model_params=dict(
        sh_degree=3,
        source_path=data_root,
        model_path='',
        white_background=False,
        data_device='cuda'
    ),
    optimization_params=dict(
        iterations=30000,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002
    ),
    pipeline_params=dict(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False
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
    warmup_ratio=0.001,
    step=[20000, 25000]
)

# 运行时配置
total_epochs = 100
checkpoint_config = dict(interval=5000)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# 评估配置
evaluation = dict(interval=1000, metric='mse')
'''
        
        # 保存配置文件
        config_path = self.output_dir / 'mmdet3d_config.py'
        with open(config_path, 'w') as f:
            f.write(config_template)
        
        print(f"MMDet3D配置文件已生成: {config_path}")

def main():
    """主函数"""
    preprocessor = FourDGSDataPreprocessor(
        input_dir='data/raw_4dgs_data',
        output_dir='data/processed_4dgs_data'
    )
    
    # 处理图像
    preprocessor.process_images(target_size=(800, 600))
    
    # 处理相机位姿
    preprocessor.process_camera_poses('data/raw_4dgs_data/camera_poses.json')
    
    # 创建训练/验证划分
    preprocessor.create_train_val_split(split_ratio=0.8)
    
    # 生成MMDet3D配置
    preprocessor.generate_mmdet3d_config(class_names=['scene'])
    
    print("数据预处理完成!")

if __name__ == '__main__':
    main()
```

---

## ⚙️ 配置最佳实践

### 最佳实践1：模块化配置

```python
# projects/configs/4DGS/_base_/4dgs_model.py
"""
4DGS模型基础配置
"""

# 模型参数
model_params = dict(
    sh_degree=3,                    # 球谐函数度数
    source_path="",                 # 数据源路径
    model_path="",                  # 模型保存路径
    white_background=False,         # 白色背景
    data_device="cuda",             # 数据设备
    eval=False                      # 评估模式
)

# 优化参数
optimization_params = dict(
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
    densify_from_iter=500,          # 开始密化迭代
    densify_until_iter=15000,       # 结束密化迭代
    densify_grad_threshold=0.0002   # 密化梯度阈值
)

# 管道参数
pipeline_params = dict(
    convert_SHs_python=False,       # 使用CUDA转换球谐函数
    compute_cov3D_python=False,     # 使用CUDA计算3D协方差
    debug=False                     # 调试模式
)

# 模型配置
model = dict(
    type='FourDGSModel',
    model_params=model_params,
    optimization_params=optimization_params,
    pipeline_params=pipeline_params
)
```

```python
# projects/configs/4DGS/_base_/4dgs_dataset.py
"""
4DGS数据集基础配置
"""

# 数据集类型
dataset_type = 'FourDGSDataset'

# 图像标准化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 训练数据管道
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCameraPoses'),
    dict(type='Resize', img_scale=(800, 600), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RandomRotate3D', angle=[-5, 5]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect3D', keys=['img', 'camera_params'])
]

# 测试数据管道
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCameraPoses'),
    dict(type='Resize', img_scale=(800, 600), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect3D', keys=['img', 'camera_params'])
]

# 数据配置
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline
    )
)
```

### 最佳实践2：环境特定配置

```python
# projects/configs/4DGS/4dgs_small_gpu.py
"""
小显存GPU配置（<8GB）
"""
_base_ = [
    './_base_/4dgs_model.py',
    './_base_/4dgs_dataset.py',
    '../_base_/default_runtime.py'
]

# 减少批次大小
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1
)

# 启用混合精度训练
fp16 = dict(loss_scale=512.)

# 启用梯度检查点
model = dict(
    optimization_params=dict(
        gradient_checkpointing=True,
        # 减少高斯点数量
        percent_dense=0.005,
        # 更早停止密化
        densify_until_iter=10000
    )
)

# 更频繁的检查点保存
checkpoint_config = dict(interval=2000)
```

```python
# projects/configs/4DGS/4dgs_high_quality.py
"""
高质量渲染配置
"""
_base_ = [
    './_base_/4dgs_model.py',
    './_base_/4dgs_dataset.py',
    '../_base_/default_runtime.py'
]

# 增加训练迭代次数
model = dict(
    optimization_params=dict(
        iterations=50000,
        # 更高的球谐函数度数
        sh_degree=4,
        # 更密集的高斯点
        percent_dense=0.02,
        # 更长的密化时间
        densify_until_iter=25000,
        # 更低的密化阈值
        densify_grad_threshold=0.0001
    )
)

# 更大的图像尺寸
data = dict(
    train=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadCameraPoses'),
            dict(type='Resize', img_scale=(1600, 1200), keep_ratio=True),
            # 其他管道...
        ]
    )
)
```

---

## 🚀 性能优化实践

### 优化实践1：内存优化

```python
#!/usr/bin/env python3
"""
内存优化示例
文件位置: examples/optimization/memory_optimization.py
"""

import torch
import gc
from contextlib import contextmanager

class MemoryOptimizer:
    """内存优化器"""
    
    @staticmethod
    @contextmanager
    def memory_efficient_training():
        """内存高效训练上下文管理器"""
        try:
            # 启用内存高效设置
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 设置内存分配策略
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(0.9)
            
            yield
            
        finally:
            # 清理内存
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def optimize_model_memory(model):
        """优化模型内存使用"""
        # 启用梯度检查点
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # 使用混合精度
        model = model.half()
        
        return model
    
    @staticmethod
    def optimize_dataloader(dataset, batch_size=1, num_workers=1):
        """优化数据加载器"""
        from torch.utils.data import DataLoader
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2
        )

# 使用示例
def memory_efficient_training():
    """内存高效训练示例"""
    with MemoryOptimizer.memory_efficient_training():
        # 训练代码
        model = build_model(cfg.model)
        model = MemoryOptimizer.optimize_model_memory(model)
        
        # 训练循环
        for batch in dataloader:
            # 训练步骤
            pass
```

### 优化实践2：计算优化

```python
#!/usr/bin/env python3
"""
计算优化示例
文件位置: examples/optimization/compute_optimization.py
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

class ComputeOptimizer:
    """计算优化器"""
    
    def __init__(self):
        self.scaler = GradScaler()
    
    @torch.jit.script
    def optimized_gaussian_render(self, gaussians, camera_params):
        """优化的高斯渲染函数"""
        # 使用JIT编译优化
        # 实际渲染逻辑...
        pass
    
    def mixed_precision_training_step(self, model, data, optimizer):
        """混合精度训练步骤"""
        with autocast():
            # 前向传播
            outputs = model(data)
            loss = self.compute_loss(outputs, data)
        
        # 反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss
    
    def compute_loss(self, outputs, targets):
        """计算损失"""
        # L1损失
        l1_loss = F.l1_loss(outputs['rendered_image'], targets['gt_image'])
        
        # SSIM损失
        ssim_loss = self.compute_ssim_loss(outputs['rendered_image'], targets['gt_image'])
        
        # 总损失
        total_loss = l1_loss + 0.2 * ssim_loss
        
        return total_loss
    
    def compute_ssim_loss(self, pred, target):
        """计算SSIM损失"""
        # 简化的SSIM计算
        mu1 = F.avg_pool2d(pred, 3, 1, 1)
        mu2 = F.avg_pool2d(target, 3, 1, 1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()

# 使用示例
def optimized_training():
    """优化训练示例"""
    optimizer = ComputeOptimizer()
    
    # 编译模型
    model = torch.compile(model, mode='max-autotune')
    
    for batch in dataloader:
        loss = optimizer.mixed_precision_training_step(model, batch, optimizer)
```

---

## 🔍 调试和监控实践

### 监控实践1：训练监控

```python
#!/usr/bin/env python3
"""
训练监控示例
文件位置: examples/monitoring/training_monitor.py
"""

import time
import psutil
import torch
import wandb
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_dir, use_wandb=False):
        """
        初始化监控器
        
        Args:
            log_dir (str): 日志目录
            use_wandb (bool): 是否使用wandb
        """
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project="4dgs-training")
        
        # 监控指标
        self.metrics = {
            'loss': [],
            'lr': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'training_time': []
        }
        
        self.start_time = time.time()
    
    def log_metrics(self, iteration, loss, lr, model=None):
        """记录训练指标"""
        current_time = time.time()
        
        # 基础指标
        self.writer.add_scalar('Loss/Total', loss, iteration)
        self.writer.add_scalar('Learning_Rate', lr, iteration)
        
        # 内存使用
        memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
        self.writer.add_scalar('Memory/GPU_Memory_GB', memory_usage, iteration)
        
        # GPU利用率
        gpu_util = self._get_gpu_utilization()
        self.writer.add_scalar('Hardware/GPU_Utilization', gpu_util, iteration)
        
        # CPU使用率
        cpu_util = psutil.cpu_percent()
        self.writer.add_scalar('Hardware/CPU_Utilization', cpu_util, iteration)
        
        # 训练时间
        elapsed_time = current_time - self.start_time
        self.writer.add_scalar('Time/Elapsed_Hours', elapsed_time / 3600, iteration)
        
        # 如果有模型，记录模型相关指标
        if model is not None:
            self._log_model_metrics(model, iteration)
        
        # wandb记录
        if self.use_wandb:
            wandb.log({
                'loss': loss,
                'learning_rate': lr,
                'memory_usage_gb': memory_usage,
                'gpu_utilization': gpu_util,
                'iteration': iteration
            })
        
        # 保存到内存
        self.metrics['loss'].append(loss)
        self.metrics['lr'].append(lr)
        self.metrics['memory_usage'].append(memory_usage)
        self.metrics['gpu_utilization'].append(gpu_util)
        self.metrics['training_time'].append(elapsed_time)
    
    def _get_gpu_utilization(self):
        """获取GPU利用率"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0
    
    def _log_model_metrics(self, model, iteration):
        """记录模型相关指标"""
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.writer.add_scalar('Model/Total_Parameters', total_params, iteration)
        self.writer.add_scalar('Model/Trainable_Parameters', trainable_params, iteration)
        
        # 梯度统计
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        self.writer.add_scalar('Gradients/Total_Norm', total_norm, iteration)
    
    def log_images(self, iteration, images_dict):
        """记录图像"""
        for name, image in images_dict.items():
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu()
            self.writer.add_image(f'Images/{name}', image, iteration)
    
    def generate_report(self, save_path=None):
        """生成训练报告"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.metrics['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        
        # 学习率曲线
        axes[0, 1].plot(self.metrics['lr'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Learning Rate')
        
        # 内存使用
        axes[1, 0].plot(self.metrics['memory_usage'])
        axes[1, 0].set_title('GPU Memory Usage')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Memory (GB)')
        
        # GPU利用率
        axes[1, 1].plot(self.metrics['gpu_utilization'])
        axes[1, 1].set_title('GPU Utilization')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Utilization (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(f'{self.log_dir}/training_report.png')
        
        plt.close()
    
    def close(self):
        """关闭监控器"""
        self.writer.close()
        if self.use_wandb:
            wandb.finish()

# 使用示例
def monitored_training():
    """带监控的训练示例"""
    monitor = TrainingMonitor('logs/training', use_wandb=True)
    
    try:
        for iteration in range(10000):
            # 训练步骤
            loss = train_step()
            lr = get_current_lr()
            
            # 记录指标
            if iteration % 100 == 0:
                monitor.log_metrics(iteration, loss, lr, model)
            
            # 记录图像
            if iteration % 1000 == 0:
                rendered_image = model.render(test_camera)
                monitor.log_images(iteration, {
                    'rendered': rendered_image,
                    'ground_truth': gt_image
                })
    
    finally:
        monitor.generate_report()
        monitor.close()
```

### 调试实践2：错误诊断

```python
#!/usr/bin/env python3
"""
错误诊断工具
文件位置: examples/debugging/error_diagnosis.py
"""

import torch
import traceback
import logging
from contextlib import contextmanager

class FourDGSDebugger:
    """4DGS调试器"""
    
    def __init__(self, log_level=logging.DEBUG):
        """初始化调试器"""
        self.logger = self._setup_logger(log_level)
        self.error_count = 0
        self.warning_count = 0
    
    def _setup_logger(self, log_level):
        """设置日志记录器"""
        logger = logging.getLogger('4DGS_Debugger')
        logger.setLevel(log_level)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    @contextmanager
    def debug_context(self, operation_name):
        """调试上下文管理器"""
        self.logger.info(f"开始执行: {operation_name}")
        start_time = time.time()
        
        try:
            yield
            elapsed_time = time.time() - start_time
            self.logger.info(f"完成执行: {operation_name}, 耗时: {elapsed_time:.2f}秒")
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"执行失败: {operation_name}")
            self.logger.error(f"错误信息: {str(e)}")
            self.logger.error(f"错误堆栈:\n{traceback.format_exc()}")
            raise
    
    def check_tensor_health(self, tensor, name="tensor"):
        """检查张量健康状态"""
        if not isinstance(tensor, torch.Tensor):
            self.logger.warning(f"{name} 不是张量类型: {type(tensor)}")
            return False
        
        # 检查NaN
        if torch.isnan(tensor).any():
            self.logger.error(f"{name} 包含NaN值")
            return False
        
        # 检查Inf
        if torch.isinf(tensor).any():
            self.logger.error(f"{name} 包含Inf值")
            return False
        
        # 检查形状
        if tensor.numel() == 0:
            self.logger.warning(f"{name} 是空张量")
            return False
        
        # 检查数值范围
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        
        if abs(tensor_min) > 1e6 or abs(tensor_max) > 1e6:
            self.logger.warning(f"{name} 数值范围异常: [{tensor_min:.2e}, {tensor_max:.2e}]")
        
        self.logger.debug(f"{name} 健康检查通过: shape={tensor.shape}, range=[{tensor_min:.4f}, {tensor_max:.4f}]")
        return True
    
    def check_model_health(self, model):
        """检查模型健康状态"""
        self.logger.info("开始模型健康检查...")
        
        # 检查参数
        for name, param in model.named_parameters():
            if not self.check_tensor_health(param, f"参数 {name}"):
                return False
        
        # 检查梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not self.check_tensor_health(param.grad, f"梯度 {name}"):
                    return False
        
        self.logger.info("模型健康检查完成")
        return True
    
    def diagnose_training_issues(self, loss_history, lr_history):
        """诊断训练问题"""
        self.logger.info("开始训练问题诊断...")
        
        # 检查损失趋势
        if len(loss_history) > 10:
            recent_losses = loss_history[-10:]
            if all(l1 <= l2 for l1, l2 in zip(recent_losses[:-1], recent_losses[1:])):
                self.logger.warning("损失持续上升，可能存在梯度爆炸或学习率过高")
            
            if all(abs(l1 - l2) < 1e-6 for l1, l2 in zip(recent_losses[:-1], recent_losses[1:])):
                self.logger.warning("损失停止下降，可能陷入局部最优或学习率过低")
        
        # 检查学习率
        if len(lr_history) > 0:
            current_lr = lr_history[-1]
            if current_lr > 1e-2:
                self.logger.warning(f"学习率可能过高: {current_lr}")
            elif current_lr < 1e-6:
                self.logger.warning(f"学习率可能过低: {current_lr}")
        
        self.logger.info("训练问题诊断完成")
    
    def memory_profiling(self):
        """内存分析"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            self.logger.info(f"GPU内存使用: {allocated:.2f}GB / {reserved:.2f}GB")
            
            if allocated > 0.9 * reserved:
                self.logger.warning("GPU内存使用率过高，可能导致OOM")
    
    def generate_debug_report(self):
        """生成调试报告"""
        report = f"""
=== 4DGS调试报告 ===
错误数量: {self.error_count}
警告数量: {self.warning_count}
GPU内存状态: {torch.cuda.memory_allocated() / 1024**3:.2f}GB
"""
        self.logger.info(report)
        return report

# 使用示例
def debug_training():
    """调试训练示例"""
    debugger = FourDGSDebugger()
    
    with debugger.debug_context("模型初始化"):
        model = build_model(cfg.model)
        debugger.check_model_health(model)
    
    loss_history = []
    lr_history = []
    
    for iteration in range(1000):
        with debugger.debug_context(f"训练迭代 {iteration}"):
            # 训练步骤
            loss = train_step()
            lr = get_current_lr()
            
            # 健康检查
            debugger.check_tensor_health(loss, "损失")
            debugger.memory_profiling()
            
            loss_history.append(loss.item())
            lr_history.append(lr)
            
            # 定期诊断
            if iteration % 100 == 0:
                debugger.diagnose_training_issues(loss_history, lr_history)
    
    # 生成最终报告
    debugger.generate_debug_report()
```

---

## 🚀 生产环境部署

### 部署实践1：模型服务化

```python
#!/usr/bin/env python3
"""
4DGS模型服务
文件位置: examples/deployment/model_service.py
"""

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import cv2
import io
from typing import List, Optional

app = FastAPI(title="4DGS渲染服务", version="1.0.0")

class CameraParams(BaseModel):
    """相机参数模型"""
    camera_center: List[float]
    camera_rotation: List[List[float]]
    fov: float
    width: int
    height: int

class RenderRequest(BaseModel):
    """渲染请求模型"""
    camera_params: CameraParams
    output_format: str = "png"
    quality: int = 95

class FourDGSService:
    """4DGS服务类"""
    
    def __init__(self, model_path: str, config_path: str):
        """初始化服务"""
        self.model = self._load_model(model_path, config_path)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def _load_model(self, model_path: str, config_path: str):
        """加载模型"""
        from mmcv import Config
        from mmdet3d.models import build_model
        
        cfg = Config.fromfile(config_path)
        model = build_model(cfg.model)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
    
    def render(self, camera_params: dict) -> np.ndarray:
        """渲染图像"""
        with torch.no_grad():
            result = self.model.render(camera_params)
            rendered_image = result['rendered_image'].cpu().numpy()
        
        # 确保图像格式正确
        if rendered_image.dtype != np.uint8:
            rendered_image = (rendered_image * 255).astype(np.uint8)
        
        return rendered_image

# 全局服务实例
service = None

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global service
    service = FourDGSService(
        model_path="checkpoints/4dgs_final.pth",
        config_path="projects/configs/4DGS/4dgs_base.py"
    )

@app.post("/render")
async def render_image(request: RenderRequest):
    """渲染图像接口"""
    try:
        # 转换相机参数
        camera_params = {
            'camera_center': np.array(request.camera_params.camera_center),
            'camera_rotation': np.array(request.camera_params.camera_rotation),
            'fov': request.camera_params.fov,
            'width': request.camera_params.width,
            'height': request.camera_params.height
        }
        
        # 渲染
        rendered_image = service.render(camera_params)
        
        # 编码图像
        if request.output_format.lower() == 'jpg':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), request.quality]
            _, buffer = cv2.imencode('.jpg', rendered_image, encode_param)
        else:
            _, buffer = cv2.imencode('.png', rendered_image)
        
        # 返回图像
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type=f"image/{request.output_format.lower()}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model_loaded": service is not None}

@app.get("/model_info")
async def model_info():
    """模型信息接口"""
    if service is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "model_type": "4DGS",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "memory_usage": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 部署实践2：Docker容器化

```dockerfile
# Dockerfile
# 文件位置: examples/deployment/Dockerfile

FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 安装项目
RUN pip3 install -e .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "examples/deployment/model_service.py"]
```

```yaml
# docker-compose.yml
# 文件位置: examples/deployment/docker-compose.yml

version: '3.8'

services:
  4dgs-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - 4dgs-service
    restart: unless-stopped
```

---

## 📚 常见使用场景

### 场景1：虚拟现实应用

```python
#!/usr/bin/env python3
"""
VR应用示例
文件位置: examples/applications/vr_application.py
"""

import numpy as np
import time
from threading import Thread, Queue

class VRRenderer:
    """VR渲染器"""
    
    def __init__(self, model_service):
        """初始化VR渲染器"""
        self.model_service = model_service
        self.render_queue = Queue()
        self.result_queue = Queue()
        self.is_running = False
        
        # VR参数
        self.ipd = 0.064  # 瞳距 (米)
        self.eye_relief = 0.01  # 眼镜距离
        
    def start_rendering(self):
        """开始渲染线程"""
        self.is_running = True
        self.render_thread = Thread(target=self._render_loop)
        self.render_thread.start()
    
    def stop_rendering(self):
        """停止渲染"""
        self.is_running = False
        if hasattr(self, 'render_thread'):
            self.render_thread.join()
    
    def _render_loop(self):
        """渲染循环"""
        while self.is_running:
            try:
                # 获取渲染请求
                if not self.render_queue.empty():
                    head_pose = self.render_queue.get()
                    
                    # 渲染双眼图像
                    left_image = self._render_eye(head_pose, 'left')
                    right_image = self._render_eye(head_pose, 'right')
                    
                    # 返回结果
                    self.result_queue.put({
                        'left': left_image,
                        'right': right_image,
                        'timestamp': time.time()
                    })
                
                time.sleep(0.001)  # 1ms延迟
                
            except Exception as e:
                print(f"渲染错误: {e}")
    
    def _render_eye(self, head_pose, eye):
        """渲染单眼图像"""
        # 计算眼睛位置
        eye_offset = np.array([-self.ipd/2 if eye == 'left' else self.ipd/2, 0, 0])
        eye_position = head_pose['position'] + head_pose['rotation'] @ eye_offset
        
        # 构建相机参数
        camera_params = {
            'camera_center': eye_position,
            'camera_rotation': head_pose['rotation'],
            'fov': 110.0,  # VR典型FOV
            'width': 1920,
            'height': 1080
        }
        
        # 渲染
        return self.model_service.render(camera_params)
    
    def submit_render_request(self, head_pose):
        """提交渲染请求"""
        if not self.render_queue.full():
            self.render_queue.put(head_pose)
    
    def get_rendered_frame(self):
        """获取渲染帧"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

# 使用示例
def vr_demo():
    """VR演示"""
    # 初始化渲染器
    vr_renderer = VRRenderer(service)
    vr_renderer.start_rendering()
    
    try:
        # 模拟VR头显追踪
        for frame in range(1000):
            # 模拟头部姿态
            angle = frame * 0.01
            head_pose = {
                'position': np.array([
                    2 * np.sin(angle),
                    0,
                    2 * np.cos(angle)
                ]),
                'rotation': np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
            }
            
            # 提交渲染请求
            vr_renderer.submit_render_request(head_pose)
            
            # 获取渲染结果
            result = vr_renderer.get_rendered_frame()
            if result:
                print(f"Frame {frame}: 渲染延迟 {time.time() - result['timestamp']:.3f}s")
            
            time.sleep(1/90)  # 90 FPS
    
    finally:
        vr_renderer.stop_rendering()
```

### 场景2：电影制作

```python
#!/usr/bin/env python3
"""
电影制作应用示例
文件位置: examples/applications/film_production.py
"""

import numpy as np
import json
from pathlib import Path

class FilmProductionPipeline:
    """电影制作流水线"""
    
    def __init__(self, model_service, project_dir):
        """初始化制作流水线"""
        self.model_service = model_service
        self.project_dir = Path(project_dir)
        self.shots_dir = self.project_dir / 'shots'
        self.shots_dir.mkdir(parents=True, exist_ok=True)
    
    def create_shot(self, shot_name, camera_path, duration, fps=24):
        """
        创建镜头
        
        Args:
            shot_name (str): 镜头名称
            camera_path (list): 相机路径关键帧
            duration (float): 持续时间（秒）
            fps (int): 帧率
        """
        shot_dir = self.shots_dir / shot_name
        shot_dir.mkdir(exist_ok=True)
        
        # 生成相机轨迹
        total_frames = int(duration * fps)
        camera_trajectory = self._interpolate_camera_path(camera_path, total_frames)
        
        # 渲染所有帧
        for frame_idx, camera_params in enumerate(camera_trajectory):
            print(f"渲染镜头 {shot_name}, 帧 {frame_idx+1}/{total_frames}")
            
            # 高质量渲染参数
            render_params = {
                **camera_params,
                'width': 4096,  # 4K分辨率
                'height': 2160,
                'samples': 256,  # 高采样率
                'quality': 'ultra'
            }
            
            # 渲染
            rendered_image = self.model_service.render(render_params)
            
            # 保存帧
            frame_path = shot_dir / f'frame_{frame_idx:06d}.exr'
            self._save_exr(rendered_image, frame_path)
        
        # 保存镜头信息
        shot_info = {
            'name': shot_name,
            'duration': duration,
            'fps': fps,
            'total_frames': total_frames,
            'resolution': [4096, 2160],
            'camera_path': camera_path
        }
        
        with open(shot_dir / 'shot_info.json', 'w') as f:
            json.dump(shot_info, f, indent=2)
        
        print(f"镜头 {shot_name} 渲染完成")
    
    def _interpolate_camera_path(self, keyframes, total_frames):
        """插值相机路径"""
        trajectory = []
        
        for i in range(total_frames):
            t = i / (total_frames - 1)
            
            # 找到相邻的关键帧
            keyframe_t = t * (len(keyframes) - 1)
            idx = int(keyframe_t)
            local_t = keyframe_t - idx
            
            if idx >= len(keyframes) - 1:
                camera_params = keyframes[-1]
            else:
                # 线性插值
                kf1 = keyframes[idx]
                kf2 = keyframes[idx + 1]
                
                camera_params = {
                    'camera_center': self._lerp(
                        np.array(kf1['camera_center']),
                        np.array(kf2['camera_center']),
                        local_t
                    ),
                    'camera_rotation': self._slerp_rotation(
                        np.array(kf1['camera_rotation']),
                        np.array(kf2['camera_rotation']),
                        local_t
                    ),
                    'fov': self._lerp(kf1['fov'], kf2['fov'], local_t)
                }
            
            trajectory.append(camera_params)
        
        return trajectory
    
    def _lerp(self, a, b, t):
        """线性插值"""
        return a + t * (b - a)
    
    def _slerp_rotation(self, r1, r2, t):
        """球面线性插值旋转矩阵"""
        # 简化实现，实际应用中应使用四元数
        return r1 + t * (r2 - r1)
    
    def _save_exr(self, image, path):
        """保存EXR格式图像"""
        # 这里应该使用OpenEXR库保存高动态范围图像
        # 简化实现
        import cv2
        cv2.imwrite(str(path).replace('.exr', '.png'), image)
    
    def create_sequence(self, sequence_name, shots):
        """创建序列"""
        sequence_dir = self.project_dir / 'sequences' / sequence_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        sequence_info = {
            'name': sequence_name,
            'shots': shots,
            'total_duration': sum(shot['duration'] for shot in shots)
        }
        
        with open(sequence_dir / 'sequence_info.json', 'w') as f:
            json.dump(sequence_info, f, indent=2)

# 使用示例
def film_production_demo():
    """电影制作演示"""
    pipeline = FilmProductionPipeline(service, 'film_project')
    
    # 定义相机路径
    camera_path = [
        {
            'camera_center': [10, 0, 5],
            'camera_rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'fov': 35
        },
        {
            'camera_center': [0, 10, 5],
            'camera_rotation': [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            'fov': 50
        },
        {
            'camera_center': [-10, 0, 5],
            'camera_rotation': [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            'fov': 35
        }
    ]
    
    # 创建镜头
    pipeline.create_shot('shot_001', camera_path, duration=5.0, fps=24)
```

---

## 📝 总结

本文档提供了4DGS模块的全面使用示例和最佳实践，涵盖了从基础使用到高级应用的各个方面。通过这些示例，开发者可以：

1. **快速上手**：通过简单示例了解基本用法
2. **深入学习**：通过完整示例掌握高级功能
3. **优化性能**：通过最佳实践提升效率
4. **解决问题**：通过调试工具快速定位问题
5. **生产部署**：通过部署示例实现产品化

### 🎯 关键要点

- **模块化设计**：保持代码结构清晰，便于维护
- **性能优化**：合理使用内存和计算资源
- **错误处理**：完善的异常处理和日志记录
- **监控调试**：实时监控训练状态，及时发现问题
- **文档完善**：详细的代码注释和使用说明

### 🚀 下一步

建议开发者根据具体需求选择合适的示例进行学习和实践，并结合项目实际情况进行调整和优化。

---

*更多信息请参考项目文档和API参考手册。*