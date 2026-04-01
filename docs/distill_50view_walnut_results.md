# Distillation 50视图 Walnut 实验结果

| iteration | PSNR_3D | SSIM_3D | PSNR_2D_train | SSIM_2D_train | PSNR_2D_test | SSIM_2D_test |
|:----------|--------:|--------:|--------------:|--------------:|-------------:|-------------:|
| 10000 | 31.77 | 0.763 | 34.45 | 0.929 | 33.70 | 0.919 |

**数据集**: Walnut (50视图)  
**训练迭代**: 10000  
**教师模型**: `./cnn_teacher_walnut_75view/best_model.pth` (75视图Walnut教师)  
**lambda_tv**: 0.1

## 与Baseline对比

| 方法 | PSNR_3D | SSIM_3D | 提升 |
|-----|--------:|--------:|------|
| Baseline (50view) | 32.48 | 0.717 | - |
| Distillation (50view) | 31.77 | 0.763 | PSNR -0.71, **SSIM +0.046** |

蒸馏在SSIM上有明显提升，PSNR略低于Baseline。