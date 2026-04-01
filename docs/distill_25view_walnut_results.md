# Distillation 25视图 Walnut 实验结果

| iteration | PSNR_3D | SSIM_3D | PSNR_2D_train | SSIM_2D_train | PSNR_2D_test | SSIM_2D_test |
|:----------|--------:|--------:|--------------:|--------------:|-------------:|-------------:|
| 10000 | 29.37 | 0.726 | 35.31 | 0.937 | 32.11 | 0.903 |

**数据集**: Walnut (25视图)  
**训练迭代**: 10000  
**教师模型**: `./cnn_teacher_walnut_75view/best_model.pth` (75视图Walnut教师)  
**lambda_tv**: 0.1

## 评估指标说明
- **PSNR_3D / SSIM_3D**: 体积重建质量
- **PSNR_2D_train / SSIM_2D_train**: 训练集2D渲染质量
- **PSNR_2D_test / SSIM_2D_test**: 测试集2D渲染质量

## 与Baseline对比

| 方法 | PSNR_3D | SSIM_3D | 提升 |
|-----|--------:|--------:|------|
| Baseline (25view) | 29.42 | 0.676 | - |
| Distillation (25view) | 29.37 | 0.726 | PSNR -0.05, SSIM +0.050 |

蒸馏在SSIM上有明显提升 (+0.05)，PSNR基本持平。