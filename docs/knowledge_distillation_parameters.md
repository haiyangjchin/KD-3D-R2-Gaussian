# 知识蒸馏参数说明

本文档详细说明了 R²-Gaussian 项目中知识蒸馏相关参数的含义和使用方法。

## 1. 蒸馏配置参数 (`distillation_config`)

这些参数用于配置知识蒸馏的整体行为，可通过 JSON 文件或代码字典传入。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `type` | string | `'volume'` | 蒸馏类型。可选值：`'volume'`（体积蒸馏）、`'feature'`（特征蒸馏） |
| `temperature` | float | `2.0` | KL 散度的温度参数。值越高，教师模型输出的概率分布越平滑，便于学生模型学习。通常取值范围：1.0-5.0 |
| `alpha` | float | `0.7` | 平衡因子，用于控制蒸馏损失与原始重建损失的权重比例。取值范围：0.0-1.0 |
| `use_kl` | bool | `true` | 是否使用 KL 散度损失。KL 散度用于衡量教师和学生模型输出分布的差异 |
| `use_l1` | bool | `true` | 是否使用 L1 损失（平均绝对误差）。L1 损失对异常值更鲁棒 |
| `use_ssim` | bool | `false` | 是否使用 SSIM 损失（结构相似性）。注意：3D SSIM 计算量较大，可能显著增加训练时间 |
| `total_iterations` | int | - | 总训练迭代次数，用于渐进式蒸馏调度 |
| `warmup_ratio` | float | `0.0` | 预热比例（0.0-1.0）。预热期间不进行蒸馏，让学生模型先独立学习。例如：0.3 表示前 30% 的迭代不使用蒸馏 |
| `max_distill_weight` | float | `0.5` | 蒸馏损失的最大权重。最终损失 = 标准损失 + weight × 蒸馏损失 |
| `schedule` | string | `'linear'` | 蒸馏权重调度策略。可选值：`'linear'`（线性增长）、`'cosine'`（余弦增长）、`'step'`（阶梯增长） |

## 2. 教师模型参数 (`CNNDistillationTeacher`)

教师模型是预训练的 CNN，用于生成软标签指导学生模型训练。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `cnn_model_path` | string | - | 预训练 CNN 模型检查点的路径（必需） |
| `device` | string | `'cuda'` | 计算设备。可选值：`'cuda'`、`'cpu'` |
| `target_depth` | int | `64` | 输出体积的目标深度维度。应与训练数据的深度维度匹配 |
| `model_size` | string | `'small'` | CNN 模型大小。决定特征通道数：<br>- `'small'`: [16, 32, 64, 128]<br>- `'medium'`: [32, 64, 128, 256]<br>- `'large'`: [64, 128, 256, 512] |

## 3. 体积蒸馏损失参数 (`VolumeDistillationLoss`)

体积级别的蒸馏损失，比较教师和学生模型的三维体积输出。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `temperature` | float | `2.0` | 温度参数，用于软化概率分布 |
| `alpha` | float | `0.7` | 损失平衡因子 |
| `use_kl` | bool | `true` | 是否使用 KL 散度损失 |
| `use_l1` | bool | `true` | 是否使用 L1 损失 |
| `use_ssim` | bool | `false` | 是否使用 SSIM 损失 |

## 4. 特征蒸馏损失参数 (`FeatureDistillationLoss`)

特征级别的蒸馏损失，匹配教师和学生模型的中间层特征统计量。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `feature_matching_layers` | list[int] | `[0, 1, 2, 3]` | 需要进行特征匹配的层索引列表 |
| `weights` | list[float] | `[1.0, 0.8, 0.6, 0.4]` | 每个特征层的权重。通常浅层权重较高，深层权重较低 |

## 5. 渐进式蒸馏参数 (`ProgressiveDistillation`)

控制蒸馏权重随训练进程的变化策略。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `total_iterations` | int | - | 总训练迭代次数（必需） |
| `warmup_ratio` | float | `0.3` | 预热比例。预热期间蒸馏权重为 0 |
| `max_distill_weight` | float | `0.5` | 蒸馏权重的最大值 |
| `schedule` | string | `'linear'` | 权重调度策略：<br>- `'linear'`: 线性增长<br>- `'cosine'`: 余弦曲线增长（初期增长慢，中期快，后期慢）<br>- `'step'`: 阶梯增长（50% 时 0.3，75% 时 0.6，100% 时 1.0） |

## 6. 命令行参数

运行 `train_with_distillation.py` 时可使用的命令行参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `--cnn_model` | string | 预训练 CNN 教师模型的路径 |
| `--no_distill` | flag | 禁用知识蒸馏（回退到标准训练） |
| `--distill_config` | string | 蒸馏配置 JSON 文件的路径 |
| `--no_improvements` | flag | 禁用改进的训练技术 |
| `--resume` | string | 从检查点恢复训练的路径 |

## 7. 配置文件示例

### 7.1 基础蒸馏配置

```yaml
# 蒸馏相关参数
cnn_model: ./cnn_teacher_50epoch_v4/checkpoints/checkpoint_epoch_034.pth
no_distill: false
distill_config: ./distill_config.json
```

### 7.2 蒸馏配置 JSON 文件示例

```json
{
    "type": "volume",
    "temperature": 2.0,
    "alpha": 0.7,
    "use_kl": true,
    "use_l1": true,
    "use_ssim": false,
    "total_iterations": 10000,
    "warmup_ratio": 0.1,
    "max_distill_weight": 0.5,
    "schedule": "linear"
}
```

### 7.3 完整训练配置示例

```yaml
# 数据路径
source_path: ./data/real_dataset/cone_ntrain_25_angle_360/pine
model_path: ./distill_output

# 训练参数
iterations: 10000
position_lr_init: 0.0002
position_lr_final: 2.0e-05

# 损失函数权重
lambda_dssim: 0.25
lambda_tv: 0.05

# 蒸馏参数
cnn_model: ./teacher_model.pth
no_distill: false
```

## 8. 使用示例

### 8.1 使用命令行

```bash
python train_with_distillation.py \
    --config config.yaml \
    --cnn_model ./teacher.pth \
    --distill_config ./distill_config.json \
    --output_dir ./output
```

### 8.2 使用配置文件

```bash
python train_with_distillation.py --config distill_student_config.yaml
```

## 9. 注意事项

1. **教师模型选择**：教师模型应与学生模型的任务匹配，通常使用在相同数据集上预训练的 CNN。

2. **温度参数调优**：较高的温度会使分布更平滑，但过高可能导致信息丢失。建议从 2.0 开始调整。

3. **权重调度**：渐进式蒸馏有助于避免早期训练不稳定。对于小数据集，建议使用较小的 `warmup_ratio`（如 0.1）。

4. **计算开销**：启用 SSIM 损失会显著增加计算时间，建议仅在需要高质量重建时使用。

5. **特征蒸馏**：特征蒸馏需要访问 CNN 的中间层，目前仅支持特定的 CTUNet3D 架构。