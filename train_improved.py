#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# ===================================================================
# IMPROVED VERSION:
# 1. 重要性加权视图采样 (Importance-weighted view sampling)
# 2. 更强的正则化 (Enhanced regularization for sparse views)
# ===================================================================

import os
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice


class ViewImportanceSampler:
    """
    视图重要性采样器
    基于每个视图的重建损失来计算重要性权重，并进行加权采样
    """
    def __init__(self, num_views, warmup_iters=500, update_interval=100, alpha=0.9):
        """
        Args:
            num_views: 训练视图总数
            warmup_iters: 预热迭代次数，在此之前使用均匀采样
            update_interval: 更新重要性权重的间隔
            alpha: 指数移动平均的权重因子（用于平滑损失历史）
        """
        self.num_views = num_views
        self.warmup_iters = warmup_iters
        self.update_interval = update_interval
        self.alpha = alpha

        # 初始化每个视图的损失历史（使用指数移动平均）
        self.view_loss_ema = torch.ones(num_views)
        self.view_sample_count = torch.zeros(num_views)

        # 初始化重要性权重（初始均匀分布）
        self.importance_weights = torch.ones(num_views) / num_views

    def update_view_loss(self, view_idx, loss_value):
        """更新指定视图的损失值"""
        if self.view_sample_count[view_idx] == 0:
            # 第一次采样，直接设置
            self.view_loss_ema[view_idx] = loss_value
        else:
            # 指数移动平均
            self.view_loss_ema[view_idx] = (
                self.alpha * self.view_loss_ema[view_idx] +
                (1 - self.alpha) * loss_value
            )
        self.view_sample_count[view_idx] += 1

    def update_importance_weights(self, iteration):
        """基于损失历史更新重要性权重"""
        if iteration < self.warmup_iters:
            # 预热期间使用均匀分布
            return

        # 基于损失大小计算重要性
        # 损失越大的视图，重要性越高（需要更多训练）
        loss_normalized = self.view_loss_ema / (self.view_loss_ema.sum() + 1e-8)

        # 应用温度参数平滑分布（避免过度集中在少数视图）
        temperature = 0.5
        weights = torch.pow(loss_normalized, 1.0 / temperature)
        self.importance_weights = weights / (weights.sum() + 1e-8)

    def sample_view(self, iteration):
        """基于重要性权重采样一个视图索引"""
        if iteration < self.warmup_iters:
            # 预热期间使用均匀采样
            return randint(0, self.num_views - 1)
        else:
            # 基于重要性权重采样
            view_idx = torch.multinomial(self.importance_weights, 1).item()
            return view_idx

    def get_importance_stats(self):
        """获取重要性统计信息"""
        return {
            'max_importance': self.importance_weights.max().item(),
            'min_importance': self.importance_weights.min().item(),
            'top5_views': torch.topk(self.importance_weights, k=min(5, self.num_views)).indices.tolist(),
            'avg_loss': self.view_loss_ema.mean().item(),
        }


def depth_consistency_loss(volume, reduction='mean'):
    """
    深度一致性损失：鼓励相邻体素之间的平滑过渡
    计算体积在三个方向上的梯度，并惩罚大的梯度变化

    Args:
        volume: 3D体积张量 [D, H, W]
        reduction: 'mean' 或 'sum'
    """
    # 计算三个方向的一阶梯度
    grad_x = torch.abs(volume[1:, :, :] - volume[:-1, :, :])
    grad_y = torch.abs(volume[:, 1:, :] - volume[:, :-1, :])
    grad_z = torch.abs(volume[:, :, 1:] - volume[:, :, :-1])

    # 计算梯度的平均值
    loss = grad_x.mean() + grad_y.mean() + grad_z.mean()

    if reduction == 'mean':
        return loss / 3.0
    else:
        return loss


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    use_importance_sampling=True,
    enhanced_regularization=True,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # ============================================================
    # 改进1: 初始化视图重要性采样器
    # ============================================================
    num_train_views = len(scene.getTrainCameras())
    if use_importance_sampling:
        view_sampler = ViewImportanceSampler(
            num_views=num_train_views,
            warmup_iters=500,  # 前500次迭代使用均匀采样
            update_interval=100,  # 每100次迭代更新一次权重
            alpha=0.9  # 指数移动平均的平滑因子
        )
        print(f"使用重要性加权视图采样（训练视图数: {num_train_views}）")
    else:
        view_sampler = None
        print("使用原始均匀随机采样")

    # ============================================================
    # 改进2: 增强正则化参数
    # ============================================================
    if enhanced_regularization:
        # 增加TV损失权重（从0.05增加到0.1）
        original_lambda_tv = opt.lambda_tv
        opt.lambda_tv = max(0.1, opt.lambda_tv)

        # 启用深度一致性损失
        use_depth_consistency = True
        lambda_depth = 0.02  # 深度一致性损失权重

        print(f"增强正则化已启用:")
        print(f"  - TV损失权重: {original_lambda_tv} -> {opt.lambda_tv}")
        print(f"  - 深度一致性损失权重: {lambda_depth}")
    else:
        use_depth_consistency = False
        lambda_depth = 0.0

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1

    # 获取所有训练相机（用于重要性采样）
    all_train_cameras = scene.getTrainCameras()

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # ============================================================
        # 改进1: 使用重要性加权采样选择视图
        # ============================================================
        if use_importance_sampling and view_sampler is not None:
            # 基于重要性权重采样视图
            view_idx = view_sampler.sample_view(iteration)
            viewpoint_cam = all_train_cameras[view_idx]

            # 定期更新重要性权重
            if iteration % view_sampler.update_interval == 0:
                view_sampler.update_importance_weights(iteration)
        else:
            # 原始方法：均匀随机采样
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            view_idx = randint(0, len(viewpoint_stack) - 1)
            viewpoint_cam = viewpoint_stack.pop(view_idx)

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]

        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim

        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

            # ============================================================
            # 改进2: 添加深度一致性损失
            # ============================================================
            if use_depth_consistency:
                loss_depth = depth_consistency_loss(vol_pred, reduction='mean')
                loss["depth"] = loss_depth
                loss["total"] = loss["total"] + lambda_depth * loss_depth

        # ============================================================
        # 改进1: 更新视图重要性权重
        # ============================================================
        if use_importance_sampling and view_sampler is not None:
            # 使用总损失作为该视图的重要性指标
            view_sampler.update_view_loss(view_idx, loss["total"].item())

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                postfix_dict = {
                    "loss": f"{loss['total'].item():.1e}",
                    "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                }
                # 显示重要性采样统计
                if use_importance_sampling and view_sampler is not None and iteration >= view_sampler.warmup_iters:
                    stats = view_sampler.get_importance_stats()
                    postfix_dict["imp_max"] = f"{stats['max_importance']:.3f}"

                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]

            # 添加重要性采样统计到日志
            if use_importance_sampling and view_sampler is not None and iteration >= view_sampler.warmup_iters:
                stats = view_sampler.get_importance_stats()
                metrics["importance_max"] = stats['max_importance']
                metrics["importance_min"] = stats['min_importance']
                metrics["avg_view_loss"] = stats['avg_loss']

            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
            )


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    # ============================================================
    # 新增参数：控制改进功能的开关
    # ============================================================
    parser.add_argument("--use_importance_sampling", action="store_true", default=True,
                       help="使用重要性加权视图采样（默认启用）")
    parser.add_argument("--no_importance_sampling", action="store_false", dest="use_importance_sampling",
                       help="禁用重要性加权视图采样")
    parser.add_argument("--enhanced_regularization", action="store_true", default=True,
                       help="使用增强正则化（默认启用）")
    parser.add_argument("--no_enhanced_regularization", action="store_false", dest="enhanced_regularization",
                       help="禁用增强正则化")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)
    print("=" * 60)
    print("改进功能状态:")
    print(f"  - 重要性加权视图采样: {'启用' if args.use_importance_sampling else '禁用'}")
    print(f"  - 增强正则化: {'启用' if args.enhanced_regularization else '禁用'}")
    print("=" * 60)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        use_importance_sampling=args.use_importance_sampling,
        enhanced_regularization=args.enhanced_regularization,
    )

    # All done
    print("Training complete.")
