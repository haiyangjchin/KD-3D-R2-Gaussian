# ============================================================================
# 在 real_dataset/cone_ntrain_25_angle_360 数据集上测试改进效果
# ============================================================================

## 📁 数据集信息

- **数据集路径**: `data/real_dataset/cone_ntrain_25_angle_360/`
- **包含案例**:
  - pine (松树)
  - seashell (贝壳)
  - walnut (核桃)
- **视图数量**: 25 个训练视图（稀疏视图，改进效果最明显）
- **扫描方式**: Cone beam（锥形束）

---

## 🚀 方法一：自动对比测试（推荐）

自动运行基准版本和改进版本，并对比结果。

### 运行自动测试脚本：

```bash
python test_improvement_cone25.py
```

**这个脚本会自动完成：**
1. ✅ 依次训练 pine, seashell, walnut 三个案例
2. ✅ 每个案例分别用原始方法和改进方法训练
3. ✅ 自动对比 PSNR/SSIM 指标
4. ✅ 生成对比报告文件

**预计总时间**: 约 30-90 分钟（3个案例 × 2个版本 × 5-15分钟）

**输出目录结构**:
```
output/real_dataset/cone_ntrain_25_angle_360/
├── baseline/
│   ├── pine/
│   ├── seashell/
│   └── walnut/
├── improved/
│   ├── pine/
│   ├── seashell/
│   └── walnut/
├── comparison_pine.txt
├── comparison_seashell.txt
└── comparison_walnut.txt
```

---

## 🎯 方法二：手动单案例测试

如果您只想测试一个案例（如 pine），可以手动运行：

### 步骤 1: 训练基准版本

```bash
python train.py \
    -s data/real_dataset/cone_ntrain_25_angle_360/pine \
    -m output/real_dataset/cone_ntrain_25_angle_360/baseline/pine
```

### 步骤 2: 训练改进版本

```bash
python train_improved.py \
    -s data/real_dataset/cone_ntrain_25_angle_360/pine \
    -m output/real_dataset/cone_ntrain_25_angle_360/improved/pine
```

### 步骤 3: 查看结果

训练完成后，在以下位置查看评估结果：

**基准版本**:
```
output/real_dataset/cone_ntrain_25_angle_360/baseline/pine/eval/iter_030000/
├── eval3d.yml          # 3D重建指标 (PSNR, SSIM)
├── eval2d_render_train.yml  # 训练集2D投影指标
└── eval2d_render_test.yml   # 测试集2D投影指标
```

**改进版本**:
```
output/real_dataset/cone_ntrain_25_angle_360/improved/pine/eval/iter_030000/
├── eval3d.yml
├── eval2d_render_train.yml
└── eval2d_render_test.yml
```

---

## 📊 如何查看和对比结果

### 方式1: 查看 YAML 文件（手动）

**打开 eval3d.yml**:
```yaml
psnr_3d: 35.234
ssim_3d: 0.9123
ssim_3d_x: 0.9145
ssim_3d_y: 0.9156
ssim_3d_z: 0.9067
```

**打开 eval2d_render_test.yml**:
```yaml
psnr_2d: 33.567
ssim_2d: 0.8934
psnr_2d_projs: [33.2, 33.8, ...]  # 每个投影的PSNR
ssim_2d_projs: [0.89, 0.91, ...]  # 每个投影的SSIM
```

手动对比两个版本的数值差异。

### 方式2: 使用 TensorBoard 可视化

```bash
# 查看基准版本
tensorboard --logdir output/real_dataset/cone_ntrain_25_angle_360/baseline/pine

# 同时对比两个版本
tensorboard --logdir output/real_dataset/cone_ntrain_25_angle_360
```

在浏览器打开 http://localhost:6006，可以看到：
- 训练曲线（loss_total, loss_render, loss_dssim, loss_tv）
- PSNR/SSIM 随迭代变化
- 重建的体积切片可视化
- **改进版本特有**：重要性采样统计（importance_max, importance_min）

### 方式3: 自动对比脚本生成的报告

如果使用了自动测试脚本，查看生成的对比报告：

```bash
cat output/real_dataset/cone_ntrain_25_angle_360/comparison_pine.txt
```

---

## 🔧 高级选项

### 只启用重要性采样（不启用增强正则化）

```bash
python train_improved.py \
    -s data/real_dataset/cone_ntrain_25_angle_360/pine \
    -m output/improved_sampling_only/pine \
    --no_enhanced_regularization
```

### 只启用增强正则化（不启用重要性采样）

```bash
python train_improved.py \
    -s data/real_dataset/cone_ntrain_25_angle_360/pine \
    -m output/improved_reg_only/pine \
    --no_importance_sampling
```

### 快速测试（减少迭代次数）

如果只想快速验证改进方向是否正确：

```bash
python train_improved.py \
    -s data/real_dataset/cone_ntrain_25_angle_360/pine \
    -m output/quick_test/pine \
    --iterations 10000 \
    --test_iterations 5000 10000
```

这样只需要约 2-5 分钟就能看到初步效果。

---

## 📈 预期改进效果

基于理论分析，在 **25 视图（稀疏视图）** 场景下：

| 指标 | 基准版本 | 改进版本 | 预期提升 |
|------|---------|---------|---------|
| **PSNR 3D** | ~33-36 dB | ~35-38 dB | **+1.5~2.5 dB** |
| **SSIM 3D** | ~0.88-0.92 | ~0.90-0.94 | **+0.02~0.03** |
| **PSNR 2D** | ~32-35 dB | ~33-36 dB | **+0.8~1.5 dB** |
| **SSIM 2D** | ~0.86-0.90 | ~0.88-0.92 | **+0.01~0.02** |

**注**: 实际效果取决于具体数据集。稀疏视图下改进更明显！

---

## ⚠️ 注意事项

1. **GPU 内存**: 确保有足够的 GPU 内存（建议 8GB+）
2. **训练时间**: 每个案例约 10-17 分钟（RTX 3090）
3. **输出目录**: 脚本会自动创建，无需手动创建
4. **随机性**: 由于训练有随机性，建议多次运行取平均值

---

## 🐛 故障排除

### 问题1: ModuleNotFoundError

```bash
# 确保在正确目录
cd c:\Users\24764\Desktop\r2_gaussian

# 确认环境
python -c "import torch; print(torch.__version__)"
python -c "import r2_gaussian; print('r2_gaussian imported successfully')"
```

### 问题2: CUDA out of memory

减小 batch 或体积大小：
```bash
python train_improved.py -s ... -m ... --tv_vol_size 16
```

### 问题3: 训练很慢

检查是否使用了 GPU：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📞 需要帮助？

如果遇到任何问题，请提供：
- 错误信息截图
- 运行的完整命令
- GPU 型号和显存大小
