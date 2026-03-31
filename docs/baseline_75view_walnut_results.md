# Baseline 75视图 Walnut 实验结果

| iteration | PSNR_3D | SSIM_3D | PSNR_2D_train | SSIM_2D_train | PSNR_2D_test | SSIM_2D_test |
|:----------|--------:|--------:|--------------:|--------------:|-------------:|-------------:|
| 10000 | 33.49 | 0.728 | 33.55 | 0.926 | 33.28 | 0.920 |

**数据集**: Walnut (75视图)  
**训练迭代**: 10000  
**lambda_tv**: 0.1

## 评估指标说明
- **PSNR_3D / SSIM_3D**: 体积重建质量
- **PSNR_2D_train / SSIM_2D_train**: 训练集2D渲染质量
- **PSNR_2D_test / SSIM_2D_test**: 测试集2D渲染质量