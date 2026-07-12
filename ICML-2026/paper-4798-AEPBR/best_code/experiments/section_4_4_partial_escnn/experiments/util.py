from escnn import group, gspaces, nn
import torch
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as TF
from escnn2.kernels.irreps_mapping import IrrepsMapFourierBLact
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from math import pi
import matplotlib.pyplot as plt
import os
import random
import sys

sys.path.append("..")
from networks import RPPBlock, EqResBlock, ResBlock

from PIL import Image

plt.rcParams["agg.path.chunksize"] = 200

plt.switch_backend("agg")


class _NoOpTable:
    def __init__(self, *args, **kwargs):
        pass

    def add_data(self, *args, **kwargs):
        pass


class _NoOpPlot:
    @staticmethod
    def line(*args, **kwargs):
        return None


class _NoOpWandb:
    Table = _NoOpTable
    plot = _NoOpPlot()

    @staticmethod
    def log(*args, **kwargs):
        pass

    @staticmethod
    def Image(value):
        return value

    @staticmethod
    def Html(value):
        return value


if wandb is None:
    wandb = _NoOpWandb()


def RPPApproxConv_L2(mdl, conv_wd, basic_wd):
    conv_l2 = 0.0
    basic_l2 = 0.0
    for block0 in mdl.layers_eq:
        for block1 in block0:
            if isinstance(block1, nn.SequentialModule):
                for operator in block1:
                    if operator.__class__.__name__ in ["RPPBlock"]:
                        conv_l2 += sum(
                            [
                                p.pow(2).sum()
                                for p in operator.conveq.parameters()
                                if p.requires_grad
                            ]
                        )
                        basic_l2 += sum(
                            [
                                p.pow(2).sum()
                                for p in operator.conv.parameters()
                                if p.requires_grad
                            ]
                        )
            elif isinstance(block1, RPPBlock):
                conv_l2 += sum(
                    [
                        p.pow(2).sum()
                        for p in block1.conveq.parameters()
                        if p.requires_grad
                    ]
                )
                basic_l2 += sum(
                    [
                        p.pow(2).sum()
                        for p in block1.conv.parameters()
                        if p.requires_grad
                    ]
                )
    return conv_wd * conv_l2 + basic_wd * basic_l2


def set_seed(iteration):
    random.seed(iteration)
    torch.manual_seed(iteration)
    np.random.seed(iteration)


def test_eq(
    network,
    data_loader,
    config,
    device,
    group=group.o2_group(),
    n=100,
):
    act = gspaces.flipRot2dOnR2()
    trans_type = nn.FieldType(act, [act.trivial_repr])
    img_b, _ = next(iter(data_loader))
    img_b = img_b[:20].to(device)
    transformed_images_list = [
        trans_type(img_b).transform(trans).tensor
        for trans in group.testing_elements(n=n)
    ]
    network.eval()

    errors = []
    with torch.no_grad():
        preds = network(img_b)
        preds_norm = torch.linalg.norm(preds, axis=1)

        for transformed_images in transformed_images_list:
            trans_preds = network(transformed_images)
            error = (torch.linalg.norm(trans_preds - preds, axis=1) / preds_norm).mean()
            errors.append(error.item())

    g_elements = np.arange(4 * np.pi, n)

    data = [
        [g_element, error, network.network_name, config.dataset_symmetries]
        for (g_element, error) in zip(g_elements, errors)
    ]

    table = wandb.Table(
        data=data,
        columns=["transformation element", "error", "network_name", "symmetries"],
    )

    wandb.log(
        {
            "Equivariance error": wandb.plot.line(
                table, "transformation element", "error", title="Equivariance error"
            )
        }
    )
    network.train()
    return errors


class Rotate90Transform:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle_small = np.random.uniform(-180, 180)
        # im = Image.fromarray((x.numpy().squeeze() * 255).astype(np.uint8))
        # im.save(f"{angle_small}-before.png")
        x = TF.rotate(x, int(angle_small), InterpolationMode.BILINEAR)
        angle = np.random.choice(self.angles) - angle_small
        img = TF.rotate(x, int(angle), InterpolationMode.BILINEAR)
        # im = Image.fromarray((img.numpy().squeeze() * 255).astype(np.uint8))
        # im.save(f"{angle_small}-after.png")
        # print("hokee")
        return img


def get_kl_loss(irrepmaps):
    irrepmaps = {irrepmap.layer_id: irrepmap for irrepmap in irrepmaps}
    kl_loss_sum = 0
    kl_uniform = 0
    for layer_id, irrepmap in irrepmaps.items():
        if layer_id == 0 or (layer_id - 1) not in irrepmaps:
            kl_uniform += irrepmap.kl_divergence(None).squeeze() / len(irrepmaps)
        else:
            kl_loss_sum += irrepmap.kl_divergence(
                irrepmaps[layer_id - 1]
            ).squeeze() / len(irrepmaps)
        # print(kl_loss)
    return kl_loss_sum, kl_uniform


def get_shift_loss(irrepsmaps):
    loss = 0
    for irrepmap in irrepsmaps:
        if hasattr(irrepmap, "shift_loss"):
            loss += irrepmap.shift_loss.flatten()[0]

    return loss


def get_irrepsmaps(model):
    irrepmaps = list(
        filter(lambda x: isinstance(x, IrrepsMapFourierBLact), model.modules())
    )
    return irrepmaps


def number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_accuracy(preds, targets):
    pred_class = torch.argmax(preds, dim=1)
    return ((pred_class == targets).sum() / pred_class.shape[0]).item()


def plot_signal(sphere_data: dict, label=None, f_max=None, f_min=None) -> None:
    # set up figure layouts
    _axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=True,
        title="",
        nticks=3,
    )

    _layout = dict(
        scene=dict(
            xaxis=dict(
                **_axis,
            ),
            yaxis=dict(
                **_axis,
            ),
            zaxis=dict(
                **_axis,
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=1, r=1, t=1, b=1),
    )

    _cmap_bwr = [
        [0, "rgb(0,50,255)"],
        [0.5, "rgb(200,200,200)"],
        [1, "rgb(255,50,0)"],
    ]

    f, f_identity, n, grid, n_theta = (
        sphere_data["f"],
        sphere_data["f_id"],
        sphere_data["n"],
        sphere_data["grid"],
        sphere_data["n_theta"],
    )
    if f_max is None:
        f_max = max(np.max(f), np.max(f_identity))
    if f_min is None:
        f_min = min(np.min(f), np.min(f_identity))

    n_points = (n**2) * n_theta
    # n_points = (n**2) * len(alphas)

    fs = np.split(f, len(f) // n_points)
    grid = grid
    if isinstance(grid[0].to("EV"), tuple):
        grid_array = np.array([elem.to("EV")[1] for elem in grid])
    else:
        grid_array = np.array([elem.to("EV") for elem in grid])
    thetas = np.linalg.norm(grid_array, axis=1)
    unique_theta = np.unique(np.round(thetas, 3))
    # grid_array /= thetas[:, None]
    # x = torch.outer(torch.cos(gamma), torch.sin(beta))
    # y = torch.outer(torch.sin(gamma), torch.sin(beta))
    # z = torch.outer(torch.ones(n), torch.cos(beta))

    x, y, z = (
        grid_array[: n**2, 0] / unique_theta[0],
        grid_array[: n**2, 1] / unique_theta[0],
        grid_array[: n**2, 2] / unique_theta[0],
    )

    thetas = [0] + list(unique_theta)
    figs = []
    for j, f in enumerate(fs):
        frames = []
        for i, theta in enumerate(thetas):
            data = []
            if i == 0:
                f_theta = np.ones_like(f[: n**2].reshape(n, n)) * f_identity[j]
                x_max, y_max, z_max = 0, 0, 0
                x_min, y_min, z_min = 0, 0, 0
            else:
                f_theta = f[(i - 1) * (n**2) : i * (n**2)].reshape(n, n)
                max_ind = np.argmax(f_theta)
                min_ind = np.argmin(f_theta)
                x_max, y_max, z_max = x[max_ind], y[max_ind], z[max_ind]
                x_min, y_min, z_min = x[min_ind], y[min_ind], z[min_ind]

            data += [
                go.Surface(
                    x=x.reshape(n, n),
                    y=y.reshape(n, n),
                    z=z.reshape(n, n),
                    surfacecolor=f_theta,
                    colorscale=_cmap_bwr,
                    cmin=f_min - 1e-3,
                    cmax=f_max,
                    opacity=1,
                    name=f"Equivariance",
                    showlegend=True,
                ),
                go.Scatter3d(
                    x=[x_max * 0.9, x_max * 1.5],
                    y=[y_max * 0.9, y_max * 1.5],
                    z=[z_max * 0.9, z_max * 1.5],
                    mode="lines+text",
                    line=dict(color="purple", width=30),
                    name="Maximum equivariance",
                    showlegend=True,
                    # text=["", f"({x_max:.2f}, {y_max:.2f}, {z_max:.2f})"],
                    # textfont=dict(size=60, color="purple"),
                ),
                go.Scatter3d(
                    x=[x_min * 0.9, x_min * 1.5],
                    y=[y_min * 0.9, y_min * 1.5],
                    z=[z_min * 0.9, z_min * 1.5],
                    mode="lines+text",
                    line=dict(color="darkgreen", width=30),
                    name="Minimum equivariance",
                    showlegend=True,
                    # text=["", f"({x_min:.2f}, {y_min:.2f}, {z_min:.2f})"],
                    # textfont=dict(size=60, color="darkgreen"),
                ),
                go.Scatter3d(
                    x=[-x_max * 0.9, -x_max * 1.5],
                    y=[-y_max * 0.9, -y_max * 1.5],
                    z=[-z_max * 0.9, -z_max * 1.5],
                    mode="lines+text",
                    line=dict(color="purple", width=30, dash="longdash"),
                    name="Maximum equivariance complement",
                    showlegend=True,
                    # text=["", f"({x_max:.2f}, {y_max:.2f}, {z_max:.2f})"],
                    # textfont=dict(size=60, color="purple"),
                ),
                go.Scatter3d(
                    x=[-x_min * 0.9, -x_min * 1.5],
                    y=[-y_min * 0.9, -y_min * 1.5],
                    z=[-z_min * 0.9, -z_min * 1.5],
                    mode="lines+text",
                    line=dict(color="darkgreen", width=30, dash="longdash"),
                    name="Minimum equivariance complement",
                    showlegend=True,
                    # text=["", f"({x_min:.2f}, {y_min:.2f}, {z_min:.2f})"],
                    # textfont=dict(size=60, color="darkgreen"),
                ),
                go.Scatter3d(
                    x=[-1.5, 1.5],
                    y=[0, 0],
                    z=[0, 0],
                    mode="lines+text",
                    line=dict(color="black", width=10),
                    name="x-axis",
                    # showlegend=True,
                    text=["", "x"],
                    textfont=dict(size=60),
                ),
                go.Scatter3d(
                    x=[0, 0],
                    y=[-1.5, 1.5],
                    z=[0, 0],
                    mode="lines+text",
                    line=dict(color="black", width=10),
                    name="y-axis",
                    # showlegend=True,
                    text=["", "y"],
                    textfont=dict(size=60),
                ),
                go.Scatter3d(
                    x=[0, 0],
                    y=[0, 0],
                    z=[-1.5, 1.5],
                    mode="lines+text",
                    line=dict(color="black", width=10),
                    name="z-axis",
                    # showlegend=True,
                    text=["", "z"],
                    textfont=dict(size=60),
                ),
            ]
            frames.append(
                go.Frame(
                    data=data,
                    name=f"{theta/pi:.02f} \u03C0",  # pi
                )
            )
        # To show surface at figure initialization
        fig = go.Figure(layout=_layout, frames=frames)
        f_theta = np.ones_like(f[: n**2].reshape(n, n)) * f_identity[j]
        fig.add_trace(
            go.Surface(
                x=x.reshape(n, n),
                y=y.reshape(n, n),
                z=z.reshape(n, n),
                surfacecolor=f_theta,
                colorscale=_cmap_bwr,
                cmin=f_min - 1e-3,
                cmax=f_max,
                opacity=1,
                name=f"Equivariance",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                y=[-0.1, 0.1],
                x=[-0.1, 0.1],
                z=[-0.1, 0.1],
                mode="lines+text",
                line=dict(color="purple", width=30),
                name="Maximum equivariance",
                showlegend=True,
                textfont=dict(size=60, color="purple"),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[-0.1, 0.1],
                y=[-0.1, 0.1],
                z=[-0.1, 0.1],
                mode="lines+text",
                line=dict(color="darkgreen", width=30),
                name="Minimum equivariance",
                showlegend=True,
                textfont=dict(size=60, color="darkgreen"),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                y=[-0.1, 0.1],
                x=[-0.1, 0.1],
                z=[-0.1, 0.1],
                mode="lines",
                line=dict(color="purple", width=30, dash="longdash"),
                name="Maximum equivariance complement",
                showlegend=True,
                textfont=dict(size=60),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                y=[-0.1, 0.1],
                x=[-0.1, 0.1],
                z=[-0.1, 0.1],
                mode="lines",
                line=dict(color="darkgreen", width=30, dash="longdash"),
                name="Minimum equivariance complement",
                showlegend=True,
                textfont=dict(size=60),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[-1.5, 1.5],
                y=[0, 0],
                z=[0, 0],
                mode="lines+text",
                line=dict(color="black", width=10),
                name="x-axis",
                # showlegend=True,
                text=["", "x"],
                textfont=dict(size=60),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[-1.5, 1.5],
                z=[0, 0],
                mode="lines+text",
                line=dict(color="black", width=10),
                name="y-axis",
                # showlegend=True,
                text=["", "y"],
                textfont=dict(size=60),
            ),
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[-1.5, 1.5],
                mode="lines+text",
                line=dict(color="black", width=10),
                name="z-axis",
                # showlegend=True,
                text=["", "z"],
                textfont=dict(size=60),
            ),
        )
        sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 1.5,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 0},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition": {"duration": 0, "easing": "linear"},
                            },
                        ],
                        "label": str(f.name),
                        "method": "animate",
                    }
                    for f in fig.frames
                ],
            }
        ]

        fig.update_layout(
            sliders=sliders,
            legend=dict(
                yanchor="top", y=1.59, xanchor="left", x=0.01, title_text=label
            ),
        )
        figs.append(fig)
    return figs


def log_learned_equivariance(irrepmaps, epoch, config, sphere=False):
    layer_ids = set()
    layer_ids = set([irrepmap.layer_id for irrepmap in irrepmaps])
    if list(layer_ids) == []:
        return

    layer_offset = min(layer_ids)
    n = 100
    for irrepmap in irrepmaps:
        layer_id = irrepmap.layer_id
        (prob_fn, g_elements, data) = irrepmap.get_distribution(n=n, sphere=sphere)
        if data is not None and sphere:
            figs = plot_signal(data)
            for i, fig in enumerate(figs):
                try:
                    os.makedirs("temp/", exist_ok=True)
                except FileNotFoundError:
                    pass
                except FileExistsError:
                    pass
                reflection = " reflection" if i else ""
                table = wandb.Table(columns=["test plots"])
                path_to_plotly = f"temp/{layer_id}{i}.html"
                fig.write_html(path_to_plotly, auto_play=False)
                table.add_data(wandb.Html(path_to_plotly))
                wandb.log(
                    {
                        f"Learned degree of equivariance layer {layer_id - layer_offset} 3-Sphere{reflection} plot": table
                    },
                    step=epoch,
                )

                # reflection = " reflection" if i else ""
                # wandb.log(
                #     {
                #         f"Learned degree of equivariance layer {layer_id - layer_offset} 3-Sphere{reflection}": fig
                #     },
                #     step=epoch,
                # )
        plt.title(f"Degree of equivariance layer {layer_id - layer_offset}")
        plt.ylabel("likelihood of h")
        plt.xlabel("groupelement g")
        for angle_id in range(prob_fn.shape[0]):
            angle_name = f" angle: {angle_id}" if prob_fn.shape[0] > 1 else ""
            data = [
                [g_element, prob, layer_id, config.run_name, angle_id]
                for (g_element, prob) in zip(g_elements[angle_id], prob_fn[angle_id])
            ]

            table = wandb.Table(
                data=data,
                columns=[
                    "transformation element",
                    "equivariance degree",
                    "layer_id",
                    "run_name",
                    "angle_id",
                ],
            )
            wandb.log(
                {
                    f"Learned Degree of equivariance layer {layer_id - layer_offset}{angle_name}": wandb.plot.line(
                        table,
                        "transformation element",
                        "equivariance degree",
                        title=f"Degree of equivariance layer {layer_id - layer_offset}{angle_name}",
                    )
                },
                step=epoch,
            )
            plt.plot(g_elements[angle_id], prob_fn[angle_id], label=angle_name)

        if prob_fn.shape[0] > 1:
            plt.legend()

        plt.ylim([0, 1.5])

        wandb.log(
            {
                f"gradual Degree of equivariance layer {layer_id - layer_offset}": wandb.Image(
                    plt
                )
            },
            step=epoch,
        )
        plt.close()
