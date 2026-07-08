"""
Module: loss landscape.py

This module provides routines for visualizing and animating the 3D loss surface and
decision boundaries of adversarially perturbed images, primarily in the FGSM–PGD
perturbation space.

Original code adapted and extended from:
    - GitHub Repository:
      https://github.com/Harry24k/catastrophic-overfitting
    - Associated Paper:
      Hoki Kim, Woojin Lee, and Jaewook Lee.
      "Understanding Catastrophic Overfitting in Single-step Adversarial Training."
      Advances in Neural Information Processing Systems (NeurIPS), 2020.
      arXiv: https://arxiv.org/abs/2010.01799

In this adaptation, docstrings, plotting conventions, and color schemes are refined
for publication consistency, with added parameter handling for integration into
adversarial training pipelines.

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchhk
from torchhk.vis import cal_perturb_x ,plot_perturb_plt
from IPython.display import HTML
from base64 import b64encode
from PIL import Image
from attacks.fgsm import fgsm
from attacks.pgd import pgd

#3D Plot Loss Surface
def loss_plot(images, labels, model, image_index, classes, attack_params, use_directions=False, directions=None,
              return_directions=False, bigger_plot_range=False):
    """
    Generate a 3D loss surface for a single datapoint using FGSM and PGD perturbation directions.

    Computes FGSM and PGD adversarial examples from the given image/label, uses them to form
    the x- and y-axis perturbation directions, and samples the model's cross-entropy loss
    over a grid in this 2D perturbation space.

    Args:
        images (torch.Tensor): Batch of input images (B×C×H×W).
        labels (torch.Tensor): Ground truth labels (B,).
        model (torch.nn.Module): Model to evaluate.
        image_index (int): Index of the image within `images` to visualize.
        classes (list[str]): Class names for interpretation of color regions.
        attack_params (dict): Parameters for FGSM/PGD attacks (eps, alpha, steps, etc.).
        use_directions (bool): If True, use provided `directions` instead of computing them.
        directions (tuple[torch.Tensor, torch.Tensor] or None): Precomputed (fgsm_dir, pgd_dir).
        return_directions (bool): If True, return computed perturbation directions.
        bigger_plot_range (bool): If True, set perturbation ranges to (-2,2)² instead of (0,1)².

    Returns:
        tuple:
            (fgsm_dir, pgd_dir) (torch.Tensor): Perturbation directions used.
            zs (np.ndarray): Loss surface values over the 2D grid.
            fgsm_result: Loss at extreme FGSM perturbation.
            pgd_result: Loss at extreme PGD perturbation.
    """
    j = image_index
    fgsm_images = fgsm(model, images, labels, **attack_params)
    pgd_images = pgd(model, images, labels, **attack_params)
    if bigger_plot_range:
        range_x = (-2, 2)
        range_y = (-2, 2)
    else:
        range_x = (0, 1)
        range_y = (0, 1)        
    #FGSM Direction
    if use_directions:
        fgsm_direction = directions[0]
    else:
        fgsm_direction = fgsm_images[j] - images[j]
    #PGD Direction
    # rand_direction = EPSILON * torch.rand_like(images[j])
    if use_directions:
        pgd_direction = directions[1]
    else:
        pgd_direction = pgd_images[j] - images[j]
    rx, ry, zs, colors = cal_perturb_x(model=model,
                                        image=images[j], label=labels[j],
                                        vec_x=fgsm_direction, vec_y=pgd_direction,
                                        range_x=range_x, range_y=range_y, 
                                        grid_size=30,
                                        loss=nn.CrossEntropyLoss(reduction='none'),
                                        )
    # Extract FGSM And PGD Results
    fgsm_result = colors[0][-1]
    pgd_result = colors[-1][0]
    # Check classes in the adversarial direction.
    set_colors = set(colors.reshape(-1).tolist())
    set_colors = [classes[i] for i in set_colors]

    # Draw the loss surface
    plot_perturb_plt(rx, ry, zs, colors==colors[0][0],
                     z_by_loss=True, color_by_loss=False,
                     color=["#6858ab", "#53cddb"],
                     min_value=None, max_value=None,
                     title=None, width=8, height=7, linewidth = 0.1,
                     x_ratio=1, y_ratio=1, z_ratio=1,
                     edge_color='#f2fafb',
                     pane_color=(1.0, 1.0, 1.0, 0.0),
                     tick_pad_x=0, tick_pad_y=0, tick_pad_z=1.5,
                     xticks=None, yticks=None, zticks=None,
                     xlabel='FGSM', ylabel='PGD', zlabel=r'$\ell$',
                     view_azimuth=235, view_altitude=20,
                     light_azimuth=0, light_altitude=20, light_exag=0)
    return (fgsm_direction, pgd_direction), zs, fgsm_result, pgd_result

#Visualize Loss Surface
def plot_loss_surface(images, labels, model, image_index, high_resolution, index, plot_list,
                      use_directions=False, directions=None, return_directions=False, bigger_plot_range=False):
    """
    Wrapper to render and save a 3D loss surface plot for a given training/evaluation step.

    Args:
        images (torch.Tensor): Batch of input images.
        labels (torch.Tensor): Ground truth labels.
        model (torch.nn.Module): Model to visualize.
        image_index (int): Index within the batch to visualize.
        classes (list[str]): Class names.
        attack_params (dict): Parameters for FGSM/PGD attacks.
        high_resolution (bool): If True, tag the plot as 'Batch', else 'Epoch'.
        index (int): Step/epoch index for naming.
        plot_list (list[PIL.Image.Image]): Accumulator for animation frames.
        use_directions, directions, return_directions, bigger_plot_range: Passed to `loss_plot`.
        root_path (str): Root directory to save plots.

    Returns:
        tuple:
            directions, loss_vals, fgsm_result, pgd_result.
    """
    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    if high_resolution:
        tag = "Batch"
    else:
        tag = "Epoch"
    directions, loss_vals, fgsm_result, pgd_result = loss_plot(
        images, labels, model, image_index, use_directions, directions, return_directions, bigger_plot_range)
    plt.title(f"{tag}: {index}")
    plt.savefig(f"{root_path}/loss_plots/{tag}{index}.png")
    img = Image.open(f"{root_path}/loss_plots/{tag}{index}.png")
    plot_list.append(img)
    plt.close()
    return directions, loss_vals, fgsm_result, pgd_result

#Visualize Decision Boundry
def plot_decision_boundry(loss_vals, index, plot_list):
    """
    Render and save a decision boundary map from loss values.

    Args:
        loss_vals (np.ndarray): 2D array of loss values.
        index (int): Epoch index.
        plot_list (list): Accumulator for animation frames.
        root_path (str): Save directory.
    """
    plt.imshow(loss_vals)
    plt.title(f"Epoch: {index}")
    plt.axis("off")
    plt.savefig(f"{root_path}/decision_boundry_plots/{index}.png")
    img = Image.open(f"{root_path}/decision_boundry_plots/{index}.png")
    plot_list.append(img)
    plt.close()
    
#Animate a Sequence of Plots
def animate_plots(plot_list, name, size=(5,5)):
    """
    Animate a sequence of Matplotlib figure frames into GIF and MP4.

    Args:
        plot_list (list[PIL.Image.Image]): Sequence of images to animate.
        name (str): Base filename (without extension).
        size (tuple[float, float]): Figure size in inches.
        root_path (str): Output directory.
        server (bool): If True, skip one of the MP4 saves for server context.
    """
    frames = []
    fig = plt.figure(figsize=size)
    plt.axis('off')
    num_plots = len(plot_list)
    for i in range(num_plots):
        frames.append([plt.imshow(plot_list[i],animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=800, blit=True,
                                    repeat_delay=1000, repeat=True)

    # Save as GIF
    ani.save(f'{root_path}/{name}.gif', writer='pillow')
    if not SERVER:
        ani.save(f'{root_path}/{name}.mp4')

    # Save as MP4 using ffmpeg writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1.25, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(f'{root_path}/{name}.mp4', writer=writer)

#Show Animation In Terminal
def show_animation(path, width=400):
    """
    Generate HTML to display an MP4 animation inline (Jupyter/IPython).

    Args:
        path (str): Path to the MP4 file.
        width (int): Display width in pixels.

    Returns:
        IPython.display.HTML: HTML wrapper for the given video.
    """
    mp4 = open(path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f"""
    <video width={width} controls>
          <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
