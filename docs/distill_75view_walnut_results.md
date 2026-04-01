# Distillation 75视图 Walnut 实验结果

| iteration | PSNR_3D | SSIM_3D | PSNR_2D_train | SSIM_2D_train | PSNR_2D_test | SSIM_2D_test |
|:----------|--------:|--------:|--------------:|--------------:|-------------:|-------------:|
| 10000 | 32.58 | 0.776 | 34.58 | 0.927 | 34.34 | 0.923 |

**数据集**: Walnut (75视图)  
**训练迭代**: 10000  
**教师模型**: `./cnn_teacher_walnut_75view/best_model.pth` (75视图Walnut教师)  
**lambda_tv**: 0.1

## 与Baseline对比

| 方法 | PSNR_3D | SSIM_3D | 提升 |
|-----|--------:|--------:|------|
| Baseline (75view) | 33.49 | 0.728 | - |
| Distillation (75view) | 32.58 | 0.776 | PSNR -0.91, **SSIM +0.048** |

蒸馏在SSIM上有明显提升，PSNR略低于Baseline。