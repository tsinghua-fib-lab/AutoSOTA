import torch
import utils.general_utils as utils


def gsplat_densification(iteration, scene, gaussians, batched_screenspace_pkg):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()
    if getattr(args, "pure_ssd_offload", False):
        return

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        timers.start("densification")

        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            assert (
                args.stop_update_param == False
            ), "stop_update_param must be false for densification; because it is a flag for debugging."

            gaussians.optimizer.zero_grad(
                set_to_none=True
            )  # free old tensors' grads before densification

            timers.start("densify_and_prune")
            size_threshold = 20 if iteration > args.opacity_reset_interval else None
            gaussians.densify_and_prune(
                args.densify_grad_threshold,
                args.min_opacity,
                scene.cameras_extent,
                size_threshold,
            )
            timers.stop("densify_and_prune")

            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=True
            )

            utils.inc_densify_iter()

        if utils.check_update_at_this_iter(
            iteration, args.bsz, args.opacity_reset_interval, 0
        ):
            timers.start("reset_opacity")
            gaussians.reset_opacity()
            timers.stop("reset_opacity")

        timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=False
            )


def update_densification_stats_offload_accum_grads(
    scene,
    gaussians,
    image_height,
    image_width,
    send2gpu_final_filter_indices,
    means2d_grad,
    radii,
):
    iteration = utils.get_cur_iter()
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()
    if getattr(args, "pure_ssd_offload", False):
        return

    assert (
        radii.shape[0] == send2gpu_final_filter_indices.shape[0]
    ), f"radii.shape[0]={radii.shape[0]}, send2gpu_final_filter_indices.shape[0]={send2gpu_final_filter_indices.shape[0]}"
    assert (
        send2gpu_final_filter_indices.shape[0] == means2d_grad.shape[0]
    ), f"send2gpu_final_filter_indices.shape[0]={send2gpu_final_filter_indices.shape[0]}, means2d_grad.shape[0]={means2d_grad.shape[0]}"

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        # timers.start("densification")

        # timers.start("densification_update_stats")
        gaussians.gsplat_add_densification_stats_exact_filter(
            means2d_grad,
            radii,
            send2gpu_final_filter_indices,
            image_width,
            image_height,
        )
        # timers.stop("densification_update_stats")

        # timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=False
            )


def update_densification_stats_baseline_accum_grads(
    scene,
    gaussians,
    image_height,
    image_width,
    means2d_grad,
    radii,
    visibility,
):
    iteration = utils.get_cur_iter()
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()
    if getattr(args, "pure_ssd_offload", False):
        return

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        # timers.start("densification")

        # timers.start("densification_update_stats")
        radii = radii.squeeze(0)
        visibility_filter = radii > 0

        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        gaussians.gsplat_add_densification_stats(
            means2d_grad.squeeze(0),
            visibility_filter,
            visibility_filter,
            image_width,
            image_height,
        )
        # timers.stop("densification_update_stats")

        # timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        ):
            utils.check_memory_usage(
                log_file, args, iteration, gaussians, before_densification_stop=False
            )
