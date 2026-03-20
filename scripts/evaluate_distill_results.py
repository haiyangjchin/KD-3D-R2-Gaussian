import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import glob


def evaluate_volume_pair(vol_pred_path, vol_gt_path):
    """
    评估一对预测体积和真实体积
    """
    print(f"加载预测体积: {vol_pred_path}")
    print(f"加载真实体积: {vol_gt_path}")

    vol_pred = np.load(vol_pred_path)
    vol_gt = np.load(vol_gt_path)

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
    psnr = peak_signal_noise_ratio(
        vol_gt_normalized, vol_pred_normalized, data_range=1.0
    )

    # 计算SSIM (Structural Similarity Index)
    if len(vol_pred_normalized.shape) == 3:
        # 对于3D体积，计算每个切片的SSIM然后平均
        ssim_vals = []
        for i in range(vol_gt_normalized.shape[0]):
            ssim_vals.append(
                structural_similarity(
                    vol_gt_normalized[i], vol_pred_normalized[i], data_range=1.0
                )
            )
        ssim_val = np.mean(ssim_vals)
    else:
        ssim_val = structural_similarity(
            vol_gt_normalized, vol_pred_normalized, data_range=1.0
        )

    print("\n" + "=" * 60)
    print("体积重建性能评估结果")
    print("=" * 60)
    print(f"MAE  (Mean Absolute Error):     {mae:.6f}")
    print(f"MSE  (Mean Squared Error):      {mse:.6f}")
    print(f"RMSE (Root Mean Squared Error): {np.sqrt(mse):.6f}")
    print(f"PSNR (Peak Signal-to-Noise):    {psnr:.2f} dB")
    print(f"SSIM (Structural Similarity):   {ssim_val:.6f}")
    print("=" * 60)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": np.sqrt(mse),
        "psnr": psnr,
        "ssim": ssim_val,
    }


def evaluate_all_iterations(base_path="./distill_student_10k_pine"):
    """
    评估所有迭代的结果
    """
    results = {}

    # 查找所有迭代目录
    iter_dirs = glob.glob(os.path.join(base_path, "point_cloud", "iteration_*"))
    iter_dirs.sort(key=lambda x: int(os.path.basename(x).replace("iteration_", "")))

    print(f"找到 {len(iter_dirs)} 个迭代目录")

    for iter_dir in iter_dirs:
        iter_name = os.path.basename(iter_dir)
        vol_pred_path = os.path.join(iter_dir, "vol_pred.npy")
        vol_gt_path = os.path.join(iter_dir, "vol_gt.npy")

        if os.path.exists(vol_pred_path) and os.path.exists(vol_gt_path):
            print(f"\n{'=' * 80}")
            print(f"评估迭代: {iter_name}")
            print(f"{'=' * 80}")

            try:
                metrics = evaluate_volume_pair(vol_pred_path, vol_gt_path)
                results[iter_name] = metrics
            except Exception as e:
                print(f"评估 {iter_name} 时出错: {e}")
        else:
            print(f"跳过 {iter_name}: 缺少体积文件")

    # 打印汇总结果
    if results:
        print(f"\n{'=' * 80}")
        print("所有迭代评估结果汇总")
        print(f"{'=' * 80}")
        print(f"{'迭代':<15} {'MAE':<12} {'RMSE':<12} {'PSNR (dB)':<12} {'SSIM':<12}")
        print(f"{'-' * 80}")

        for iter_name, metrics in results.items():
            print(
                f"{iter_name:<15} {metrics['mae']:<12.6f} {metrics['rmse']:<12.6f} "
                f"{metrics['psnr']:<12.2f} {metrics['ssim']:<12.6f}"
            )

    return results


def save_results_to_csv(results, output_path="distill_evaluation_results.csv"):
    """
    将结果保存到CSV文件
    """
    import pandas as pd

    if results:
        # 转换为DataFrame
        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "iteration"
        df.reset_index(inplace=True)

        # 保存CSV
        df.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")

        # 也保存为Markdown表格
        md_path = output_path.replace(".csv", ".md")
        with open(md_path, "w") as f:
            f.write("# 蒸馏训练评估结果\n\n")
            f.write(df.to_markdown(index=False))
        print(f"Markdown表格已保存到: {md_path}")


if __name__ == "__main__":
    # 评估所有迭代
    results = evaluate_all_iterations()

    # 保存结果
    if results:
        save_results_to_csv(results)
