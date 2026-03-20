import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_model():
    # 读取预测体积和真实体积
    vol_pred = np.load('distill_student_epoch34_v1/point_cloud/iteration_5000/vol_pred.npy')
    vol_gt = np.load('distill_student_epoch34_v1/point_cloud/iteration_5000/vol_gt.npy')
    
    print(f"预测体积形状: {vol_pred.shape}")
    print(f"真实体积形状: {vol_gt.shape}")
    print(f"预测体积范围: [{vol_pred.min():.6f}, {vol_pred.max():.6f}]")
    print(f"真实体积范围: [{vol_gt.min():.6f}, {vol_gt.max():.6f}]")
    
    # 确保数据范围相同（归一化到0-1）
    if vol_gt.max() > 1.0:
        vol_gt_normalized = vol_gt / vol_gt.max()
        vol_pred_normalized = vol_pred / vol_gt.max()  # 使用相同的归一化因子
    else:
        vol_gt_normalized = vol_gt
        vol_pred_normalized = vol_pred
    
    # 计算MAE (Mean Absolute Error)
    mae = np.mean(np.abs(vol_pred_normalized - vol_gt_normalized))
    
    # 计算MSE (Mean Squared Error)
    mse = np.mean((vol_pred_normalized - vol_gt_normalized) ** 2)
    
    # 计算PSNR (Peak Signal-to-Noise Ratio)
    # 数据范围是0-1
    psnr = peak_signal_noise_ratio(vol_gt_normalized, vol_pred_normalized, data_range=1.0)
    
    # 计算SSIM (Structural Similarity Index)
    # 对于3D体积数据，计算SSIM
    if len(vol_pred_normalized.shape) == 3:
        # 对于3D体积，我们可以使用多通道SSIM或者分别计算每个切片的SSIM
        # 这里我们尝试计算整个3D体积的SSIM
        try:
            ssim_val = structural_similarity(
                vol_gt_normalized, 
                vol_pred_normalized,
                data_range=1.0,
                channel_axis=None
            )
        except:
            # 如果失败，计算每个切片的平均SSIM
            ssim_vals = []
            for i in range(vol_gt_normalized.shape[0]):
                ssim_vals.append(
                    structural_similarity(
                        vol_gt_normalized[i], 
                        vol_pred_normalized[i],
                        data_range=1.0
                    )
                )
            ssim_val = np.mean(ssim_vals)
    else:
        ssim_val = structural_similarity(
            vol_gt_normalized, 
            vol_pred_normalized,
            data_range=1.0
        )
    
    print("\n" + "="*60)
    print("高斯模型性能评估结果")
    print("="*60)
    print(f"MAE  (Mean Absolute Error):     {mae:.6f}")
    print(f"MSE  (Mean Squared Error):      {mse:.6f}")
    print(f"RMSE (Root Mean Squared Error): {np.sqrt(mse):.6f}")
    print(f"PSNR (Peak Signal-to-Noise):    {psnr:.2f} dB")
    print(f"SSIM (Structural Similarity):   {ssim_val:.6f}")
    print("="*60)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'psnr': psnr,
        'ssim': ssim_val
    }

if __name__ == '__main__':
    results = evaluate_model()
