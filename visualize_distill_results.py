import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
from mpl_toolkits.mplot3d import Axes3D

# Set matplotlib parameters for better visualization
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": ["DejaVu Sans", "Arial", "sans-serif"],  # Use available fonts
    }
)


def plot_volume_slices(vol_pred_path, vol_gt_path, output_dir="./visualizations"):
    """
    绘制体积的2D切片对比图
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载体积数据
    vol_pred = np.load(vol_pred_path)
    vol_gt = np.load(vol_gt_path)

    print(f"预测体积形状: {vol_pred.shape}")
    print(f"真实体积形状: {vol_gt.shape}")

    # 归一化到0-1范围
    vol_pred_norm = vol_pred / vol_pred.max() if vol_pred.max() > 0 else vol_pred
    vol_gt_norm = vol_gt / vol_gt.max() if vol_gt.max() > 0 else vol_gt

    # 选择中间切片
    slices = [
        vol_pred.shape[0] // 2,  # 轴向中间
        vol_pred.shape[1] // 2,  # 冠状面中间
        vol_pred.shape[2] // 2,  # 矢状面中间
    ]

    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    slice_names = ["Axial Slice", "Coronal Slice", "Sagittal Slice"]

    for i, (ax_idx, slice_name) in enumerate(zip([0, 1, 2], slice_names)):
        # 预测体积切片
        if i == 0:  # 轴向
            slice_pred = vol_pred_norm[slices[i], :, :]
            slice_gt = vol_gt_norm[slices[i], :, :]
        elif i == 1:  # 冠状面
            slice_pred = vol_pred_norm[:, slices[i], :]
            slice_gt = vol_gt_norm[:, slices[i], :]
        else:  # 矢状面
            slice_pred = vol_pred_norm[:, :, slices[i]]
            slice_gt = vol_gt_norm[:, :, slices[i]]

        # 预测切片
        im1 = axes[0, i].imshow(slice_pred, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Predicted - {slice_name}", fontsize=13, pad=10)
        axes[0, i].axis("off")
        cbar1 = plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)

        # 真实切片
        im2 = axes[1, i].imshow(slice_gt, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Ground Truth - {slice_name}", fontsize=13, pad=10)
        axes[1, i].axis("off")
        cbar2 = plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=9)

    plt.suptitle("Volume Reconstruction Slice Comparison", fontsize=16, y=0.98)
    plt.tight_layout()

    # 从路径提取迭代信息
    iter_name = os.path.basename(os.path.dirname(vol_pred_path))
    output_path = os.path.join(output_dir, f"{iter_name}_slices_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"切片对比图已保存: {output_path}")

    # 创建误差图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for i, slice_name in enumerate(slice_names):
        if i == 0:  # 轴向
            slice_pred = vol_pred_norm[slices[i], :, :]
            slice_gt = vol_gt_norm[slices[i], :, :]
        elif i == 1:  # 冠状面
            slice_pred = vol_pred_norm[:, slices[i], :]
            slice_gt = vol_gt_norm[:, slices[i], :]
        else:  # 矢状面
            slice_pred = vol_pred_norm[:, :, slices[i]]
            slice_gt = vol_gt_norm[:, :, slices[i]]

        # 绝对误差
        error = np.abs(slice_pred - slice_gt)
        im = axes[i].imshow(error, cmap="hot", vmin=0, vmax=0.1)
        axes[i].set_title(f"{slice_name} - Absolute Error", fontsize=13, pad=10)
        axes[i].axis("off")
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

    plt.suptitle("Volume Reconstruction Error Distribution", fontsize=16, y=0.98)
    plt.tight_layout()

    error_path = os.path.join(output_dir, f"{iter_name}_error_slices.png")
    plt.savefig(error_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"误差分布图已保存: {error_path}")


def plot_point_cloud(pickle_path, output_dir="./visualizations", max_points=10000):
    """
    可视化点云数据
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载点云数据
    with open(pickle_path, "rb") as f:
        point_cloud_data = pickle.load(f)

    print(f"点云数据键: {list(point_cloud_data.keys())}")

    # 提取位置和密度
    xyz = point_cloud_data["xyz"]
    density = point_cloud_data.get("density", np.ones(len(xyz)))

    print(f"点云数量: {len(xyz)}")
    print(f"位置形状: {xyz.shape}")
    print(f"密度形状: {density.shape}")

    # 随机采样以减少点数
    if len(xyz) > max_points:
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        density = density[indices]

    # 创建3D点云图
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 根据密度着色
    scatter = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2], c=density, cmap="viridis", s=2, alpha=0.7
    )

    ax.set_xlabel("X Axis", fontsize=12, labelpad=10)
    ax.set_ylabel("Y Axis", fontsize=12, labelpad=10)
    ax.set_zlabel("Z Axis", fontsize=12, labelpad=10)
    ax.set_title("3D Gaussian Point Cloud Distribution", fontsize=14, pad=20)
    ax.tick_params(axis="both", which="major", labelsize=10)

    cbar = plt.colorbar(scatter, ax=ax, label="Density", shrink=0.6, pad=0.1)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Density", fontsize=12)

    # 从路径提取迭代信息
    iter_name = os.path.basename(os.path.dirname(pickle_path))
    output_path = os.path.join(output_dir, f"{iter_name}_point_cloud.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"点云图已保存: {output_path}")

    # 创建2D投影图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # XY平面投影
    axes[0].scatter(xyz[:, 0], xyz[:, 1], c=density, cmap="viridis", s=2, alpha=0.7)
    axes[0].set_xlabel("X Axis", fontsize=12)
    axes[0].set_ylabel("Y Axis", fontsize=12)
    axes[0].set_title("XY Plane Projection", fontsize=13, pad=10)
    axes[0].tick_params(axis="both", which="major", labelsize=10)
    axes[0].axis("equal")

    # XZ平面投影
    axes[1].scatter(xyz[:, 0], xyz[:, 2], c=density, cmap="viridis", s=2, alpha=0.7)
    axes[1].set_xlabel("X Axis", fontsize=12)
    axes[1].set_ylabel("Z Axis", fontsize=12)
    axes[1].set_title("XZ Plane Projection", fontsize=13, pad=10)
    axes[1].tick_params(axis="both", which="major", labelsize=10)
    axes[1].axis("equal")

    # YZ平面投影
    axes[2].scatter(xyz[:, 1], xyz[:, 2], c=density, cmap="viridis", s=2, alpha=0.7)
    axes[2].set_xlabel("Y Axis", fontsize=12)
    axes[2].set_ylabel("Z Axis", fontsize=12)
    axes[2].set_title("YZ Plane Projection", fontsize=13, pad=10)
    axes[2].tick_params(axis="both", which="major", labelsize=10)
    axes[2].axis("equal")

    plt.suptitle("Point Cloud 2D Projections", fontsize=16, y=0.98)
    plt.tight_layout()

    proj_path = os.path.join(output_dir, f"{iter_name}_point_cloud_2d.png")
    plt.savefig(proj_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"点云投影图已保存: {proj_path}")


def visualize_all_iterations(base_path="./distill_student_10k_pine"):
    """
    可视化所有迭代的结果
    """
    output_dir = os.path.join(base_path, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有迭代目录
    iter_dirs = glob.glob(os.path.join(base_path, "point_cloud", "iteration_*"))
    iter_dirs.sort(key=lambda x: int(os.path.basename(x).replace("iteration_", "")))

    print(f"找到 {len(iter_dirs)} 个迭代目录")

    for iter_dir in iter_dirs:
        iter_name = os.path.basename(iter_dir)
        print(f"\n{'=' * 80}")
        print(f"处理迭代: {iter_name}")
        print(f"{'=' * 80}")

        # 体积切片可视化
        vol_pred_path = os.path.join(iter_dir, "vol_pred.npy")
        vol_gt_path = os.path.join(iter_dir, "vol_gt.npy")

        if os.path.exists(vol_pred_path) and os.path.exists(vol_gt_path):
            try:
                plot_volume_slices(vol_pred_path, vol_gt_path, output_dir)
            except Exception as e:
                print(f"体积切片可视化失败: {e}")
        else:
            print(f"跳过体积可视化: 缺少体积文件")

        # 点云可视化
        pickle_path = os.path.join(iter_dir, "point_cloud.pickle")

        if os.path.exists(pickle_path):
            try:
                plot_point_cloud(pickle_path, output_dir)
            except Exception as e:
                print(f"点云可视化失败: {e}")
        else:
            print(f"跳过错可视化: 缺少点云文件")

    print(f"\n{'=' * 80}")
    print(f"所有可视化结果已保存到: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    visualize_all_iterations()
