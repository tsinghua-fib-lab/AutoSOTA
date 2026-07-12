from typing import Tuple, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from .transform import AffineTransformation, orbit_sampling, identity


def entropy(p: Tensor) -> Tensor:
    """
    Computes the entropy of the given probability.

    Parameters:
    p (Tensor): The input tensor.

    Returns:
    Tensor: The entropy.
    """
    return - torch.sum(p * torch.log2(p) + 1e-10, dim=-1)


def curvature(s: Tensor) -> Tensor:
    """
    Computes the curvature of a given tensor by calculating the second-order derivative.

    Parameters:
    s (Tensor): The input tensor for which the curvature is to be computed.

    Returns:
    Tensor: The curvature of the input tensor, which is the second-order derivative of the input tensor.
    """
    # Calculate the first-order derivative (gradient)
    # edge_order=2 to handle endpoint gradients
    g = torch.gradient(s, edge_order=1, dim=-1)[0]
    # Compute the curvature (second-orde derivative) of the gradient
    return torch.gradient(g, edge_order=1, dim=-1)[0]


def highlight_subplot(ax, color='red', line_width=8):
    """
    Highlights a subplot by adding a colored rectangle at the top of the subplot.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object of the subplot to be highlighted.
    color (str, optional): The color of the rectangle. Default is 'red'.
    line_width (int, optional): The width of the rectangle's line. Default is 8.

    Returns:
    None
    """
    ax_coordinates = ax.axis()
    rec = Rectangle(
        (ax_coordinates[0], ax_coordinates[2]),  # xy
        ax_coordinates[1],  # width
        1,  # height
        fill=True, lw=line_width, color=color)
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)


def gaussian_filter1d(inputs: Tensor, sigma: float, radius: int, mode: str = 'replicate') -> Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    gaussian = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = gaussian / gaussian.sum()
    # out_channels x in_channels x kernel_size
    kernel = kernel.view(1, 1, -1).tile((inputs.shape[1], inputs.shape[1], 1))
    padded = F.pad(inputs, (radius, radius), mode=mode)
    filtered = F.conv1d(padded, kernel.to(inputs.device))
    return filtered


class InverseTransformationSearch:
    """
    This is the ITS inference module, which you simply insert
    as a pre-processing step in front of classifier during inference.
    """

    batch_size: int
    T: Tensor
    history: dict
    transformation_schedule: list

    def __init__(
        self,
        model,
        its_mode,
        n_samples=17,
        n_hypotheses=1,
        mc_steps=50,
        change_of_mind='score',
        en_unique_class_condition=False,
        labels=None,
        line_thickness=1,
        fontsize=16,
        alternative_energy=None
     ):
        """
        Initialises the ITS inference module.
        Parameters:
        n_hypotheses (int>0): The number of hypotheses to be considered in the search.
                              If `en_unique_class_condition=True` this is bounded to max `n_hypotheses=n_classes`.
        change_of_mind (str): The method for changing of mind during the search.
                              This can be either:
                                - `score`: This uses the evaluation scores.
                                           This is a greedy pick of the highest scored result over the entire search.
                                - `class`: This uses the class counts per hypothesis branch. The highest support wins.
                                - `off`: Disable change of mind and use the initial best hypothesis.
        """
        assert change_of_mind in ['score', 'class', 'off'], "Choose either of 'score', 'class', 'off'."

        # You can adjust these settings without re-instantiating the class
        # e.g. by changing `its.n_hypotheses=3` and then run `its.infer(x)`
        self.change_of_mind = change_of_mind
        self.model = model
        self.its_mode = its_mode
        self.n_samples = n_samples
        self.en_unique_class_condition = en_unique_class_condition
        self.n_hypotheses = n_hypotheses
        self.mc_steps = mc_steps

        if self.its_mode == 2:
            transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value
            ]
            domains = [(torch.pi,), (0.5,)]
        elif self.its_mode == 3:
            # values from the paper (ITS^3)
            transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value
            ]
            domains = [(torch.pi,), (0.25,), (0.25,)]
        elif self.its_mode == 4:
            transformations = [
                AffineTransformation.TRANSLATION.value,
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value
            ]
            domains = [(0.5, 0.5), (torch.pi,), (0.25,), (0.25,)]
        elif self.its_mode == 5:
            # values from the paper (ITS^5)
            transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value,
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value
            ]
            domains = [(torch.pi,), (0.25,), (0.25,), (torch.pi,), (0.25,)]
        elif self.its_mode == 8:
            transformations = [
                AffineTransformation.TRANSLATION.value,
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value,
                AffineTransformation.TRANSLATION.value,
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value
            ]
            domains = [
                (0.5, 0.5),
                (torch.pi,), (0.25,), (0.25,),
                (0.5, 0.5),
                (torch.pi,), (0.25,), (0.25,)
            ]
        else:
            raise NotImplementedError(f"Unknown ITS mode: {self.its_mode}. Choose either 2, 3, 4, 5, or 8.")

        self.transformation = transformations
        self.transformation_schedule = transformations
        self.domain = domains

        # parameters initialised in `infer`
        self.batch_size = None
        self.T = None
        self.history = None

        # parameters for plotting
        self.en_plot = False
        self.line_thickness = line_thickness
        self.fontsize = fontsize
        self.labels = np.arange(100) if labels is None else labels

        # alternative energy function
        self.alternative_energy = alternative_energy

    def infer(self, x: Tensor, plot_idx: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        This is the inference function which initiates the search process.

        Parameters:
        x (Tensor): The input image batch. (batch x channel x width x height)
        plot_idx (bool, optional): The index of the batch for which to plot the search results
                                   Default is None, which means no plotting is performed.

        Returns:
        x_t (Tensor): The predicted canonical form of the input.
                            (batch x n_hypotheses x channel x width x height)
        z (Tensor): The output logit scores of the model for the predicted canonical form.
                          (batch x n_hypotheses x embedding_dim)
        """
        self.batch_size = int(x.shape[0])
        self.T = identity((self.batch_size, self.n_hypotheses)).to(x.device)
        self.history = {"orbit": [], "embedding": [], "score": [], "n_max": [], "T": []}
        self.en_plot = plot_idx is not None

        # ensure consistent shapes
        x = torch.tile(x[:, None, ...], [1, self.n_hypotheses, 1, 1, 1])
        # main search loop: iterate over all possible transformations
        x_t, z, scores = None, None, None
        for i in range(len(self.transformation_schedule)):
            # keep the same x as input but the transformation matrix is composed during search
            x_t, z, scores = self._breadth_first_search_step(x, level=i)

        assert isinstance(x_t, Tensor) and isinstance(z, Tensor) and isinstance(scores, Tensor)

        # plot the gathered search results
        if self.en_plot:
            self._plot_tree(batch_index=plot_idx)

        # hypothesis testing: change of mind
        if self.change_of_mind != 'off':
            x_t, z = self._change_of_mind(x_t, z, scores=scores)

        return x_t, z

    def _change_of_mind(self, x_t: Tensor, z: Tensor, scores: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        In each level of the search, the identity transform can be selected as candidate.
        Therefore, only the final candidate list must be reordered by the change of mind logic.

        Args:
            x_t: The hypotheses list canonical form of the input.
            z: The hypotheses list of output logit scores of the model.
            scores (Tensor, optional): Evaluation scores of each transformation. Required for `score` mode.

        Returns:
            x_t: The rearranged canonical form of the input.
            z: The rearranged output logit scores of the model.
        """
        if self.change_of_mind == 'score':
            # compute accumulated score value over all levels
            # (n_levels x batch x n_hypotheses x n_samples) -> (batch x n_hypotheses)
            scores = torch.from_numpy(np.stack(self.history["score"])).max(-1).values.sum(0)
            indices = torch.sort(scores, dim=-1, descending=True).indices.to(x_t.device)
        else:
            raise NotImplementedError(f"The change of mind mode {self.change_of_mind} is currently not implemented. "
                                      f"We refractored our prototypical code for the sake of simplicity "
                                      f"and a cleaner code base. The `class`-based change of mind mechanism requires "
                                      f"a stored class list over the search. Due to the minor differences "
                                      f"(see our paper), we see no need for it in this implementation.")

        x_t = torch.gather(x_t, dim=1, index=torch.tile(indices[..., None, None, None], [1, 1] + list(x_t.shape[-3:])))
        z = torch.gather(z, dim=1, index=torch.tile(indices[..., None], [1, 1, z.shape[-1]]))
        return x_t, z

    @staticmethod
    def _estimate_confidence(z: Tensor) -> Tensor:
        """
        Estimates the confidence score of the input `z`.

        Args:
            z: The output logit scores of the model (batch x n_hypotheses x n_classes)

        Returns:
            confidence: The estimated confidence of the input `z` (batch x n_hypotheses)
        """
        energies = torch.logsumexp(z, dim=-1)
        energies = gaussian_filter1d(energies, sigma=2, radius=3, mode='replicate')
        # todo add energy option -torch.tensor(energies[..., 1:self.n_samples+1], device=z.device)
        return -curvature(torch.tensor(energies, device=z.device))

    def _predict(self, x: Tensor) -> Tensor:
        """
        Calls `self.model` with the input `x`.
        If `mc_samples` == 1: Then the predictions are deterministic.
        Else : The predictions are stochastic.

        Help: For Huggingface or Torch models, you can access the logits by
        resnet logits model.fn; vgg logits model.classifier[6] or similar.

        Args:
            x (Tensor): The input image batch (batch * n_hypotheses * samples x n_channels x height x width)

        Returns:
            logits (Tensor): The output logit scores. (batch * n_hypotheses * samples x n_classes)
        """
        in_channels = getattr(self.model, "conv1", None)
        if in_channels is not None and in_channels.in_channels == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.mc_steps == 1:
            self.model.eval()
            return self.model(x)
        self.model.train()
        predictions = []
        for _ in range(self.mc_steps):
            with torch.no_grad():
                predictions.append(self.model(x))
        return torch.stack(predictions).mean(dim=0)

    def _evaluate_orbit(self, x: Tensor, level: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Evaluates the orbit at the current level by
            (i) Obtaining regular samples along orbit and
            (ii) Evaluating the confidence of the samples.

        Args:
            x (Tensor): The input image batch (batch x n_hypotheses x samples x n_channels x height x width)
            level (int): The current level of the search

        Returns:
            orbit (Tensor): The regular samples. (batch x n_hypotheses x n_samples x n_channels x height x width)
            scores (Tensor): Evaluation scores of the current orbit. (batch x n_hypotheses x samples)
            z (Tensor): Evaluation scores of the current orbit. (batch x n_hypotheses x n_classes)
            T (Tensor): The transformation matrix. (batch x n_hypotheses x n_samples x 3 x 3)
        """
        # (batch x n_hypotheses x n_samples x 3 x 3) -> (batch * n_hypotheses x n_samples+2 x 3 x 3)
        T_prior = self.T.flatten(end_dim=1)[:, None].expand((-1, self.n_samples + 2, -1, -1))
        # Obtain regular samples along orbit (batch x n_hypotheses x n_samples+2 x n_channels x height x width)
        # as well as the corresponding transformation matrices (batch x n_hypotheses x n_samples x 3 x 3)
        orbit, T = orbit_sampling(
            x.flatten(end_dim=1),                  # collapse batch and n_hypotheses
            self.transformation_schedule[level],  # apply the transformation of that orbit
            n_samples=self.n_samples,             # number of regular samples
            domain=[self.domain[level],],         # the domain of the current level
            T=T_prior)                            # the prior transformation matrix
        # obtain the representation of the input (batch * n_hypotheses * n_samples+2 x embedding_dim)
        z = self._predict(orbit.flatten(end_dim=2))
        z = z.unflatten(0, (self.batch_size, self.n_hypotheses, self.n_samples + 2))
        # squeeze away single orbit dim as len(T) = 1
        orbit = orbit.unflatten(0, (self.batch_size, self.n_hypotheses)).squeeze(dim=2)
        # T is a list of length 1
        T = T[0].unflatten(0, (self.batch_size, self.n_hypotheses))

        # estimate the confidence scores (batch x n_hypotheses x n_samples)
        if self.alternative_energy is None:
            score = self._estimate_confidence(z)
        else:
            bsize, n_hypotheses, n_samples, c, h, w = orbit.shape
            assert c > 0 and h > 0 and w > 0
            # (..., c, h, w) -> (new_bsize, 1, h, w)
            orbit_ = orbit.flatten(start_dim=0, end_dim=2).mean(1, keepdim=True)
            # _estimate_confidence uses logsumexp internally, when energy is -logsumexp.
            # we need to invert the sign here
            energies = -1 * self.alternative_energy.inner_forward(orbit_, None)
            energies = energies.view(bsize, n_hypotheses, n_samples)
            energies = gaussian_filter1d(energies, sigma=2, radius=3, mode='replicate')
            score = -curvature(torch.tensor(energies, device=orbit.device))

        # remove the padding again
        return (
            orbit[:, :, 1:self.n_samples + 1],
            score[:, :, 1:self.n_samples + 1],
            z[:, :, 1:self.n_samples + 1],
            T[:, :, 1:self.n_samples + 1]
        )

    def select_candidates(self, score, pred_class, level):
        """
        Selects the top-k candidates for each hypothesis based on their scores.
        If `en_unique_class_condition` is True, the selection is done based on the
        maximum score for each hypothesis across all classes. Otherwise, the selection
        is done based on the maximum score for each hypothesis on the current level.

        Args:
            score: The evaluation scores of the current orbit. (batch x n_hypotheses x n_samples)
            pred_class: The predicted classes of the input samples. (batch x n_hypotheses x n_samples)
            level: The current level of the search

        Returns:
            selected_candidates: The indices of the selected candidates. (batch x n_hypotheses)
        """
        if not self.en_unique_class_condition:

            if level == 0:
                return torch.topk(score[:, 0], k=self.n_hypotheses, dim=-1).indices

            return torch.argmax(score, dim=-1)

        if level == 0:
            score, pred_class = score[:, 0], pred_class[:, 0]

        n_max = torch.zeros((self.batch_size, self.n_hypotheses), device=score.device, dtype=torch.int64)

        for i in range(self.n_hypotheses):
            # find the first/next max score index
            if level == 0:
                # index: (B x 1 x 1)
                index = torch.argmax(score, dim=-1)
                # get the class associated with this index (on the orbit)
                c = pred_class[torch.arange(score.shape[0]), index]
                # replace every occurrence of the class with a zero score
                score = torch.where(pred_class == c[:, None], torch.full_like(score, -torch.inf), score)
                # update the top-k list
                n_max[:, i] = index
            else:
                # index: (B, )
                index = torch.argmax(score[:, i], dim=-1)
                # get the class associated with this index (on the orbit)
                c = torch.gather(pred_class[:, i], -1, index[:, None])[:, None]
                # replace every occurrence of the class with a zero score
                score = torch.where(pred_class == c, torch.full_like(score, -torch.inf), score)
                # update the top-k list
                n_max[:, i] = index
        return n_max

    def _breadth_first_search_step(self, x: Tensor, level: int) -> Tuple[Tensor, Tensor, Tensor]:
        """ Performs one breadth first search step.

        Args:
            x (Tensor): The input to this search step (batch x channel x height x width).
            level (int): The current level of the search.

        Returns:
            x_incumbent (Tensor): The best performing input transformation.
            z_incumbent (Tensor): The corresponding embedding.
            s_primal (Tensor): The corresponding score.
        """
        # compute the highest class scores and orbits
        orbit, s, z, T = self._evaluate_orbit(x, level)
        # select the k best samples from the orbit -> (batch x n_hypotheses)
        n_max = self.select_candidates(s, torch.argmax(z, dim=-1), level)
        # update incumbent (input): (batch x n_hypotheses x n_samples x channel x height x width)
        x_incumbent = torch.gather(orbit, dim=2, index=torch.tile(
            n_max[..., None, None, None, None], [1, 1, 1] + list(orbit.shape[-3:]))).squeeze(2)
        # update the accumulative transformation matrix: (batch x n_hypotheses x n_samples x 3 x 3)
        self.T = torch.gather(T, dim=2, index=torch.tile(
            n_max[..., None, None, None], [1, 1, 1, 3, 3])).squeeze(2)
        # update incumbent (embedding): (batch x n_hypotheses x n_samples x n_classes)
        z_incumbent = torch.gather(z, dim=2, index=torch.tile(
            n_max[..., None, None], [1, 1, 1, z.shape[-1]])).squeeze(2)
        # update primal (confidence score): (batch x n_hypotheses x n_samples)
        s_primal = torch.gather(s, dim=2, index=n_max[..., None]).squeeze(2)
        # store the results
        self.history["orbit"].append(orbit.detach().cpu().numpy())
        self.history["embedding"].append(z.detach().cpu().numpy())
        self.history["score"].append(s.detach().cpu().numpy())
        self.history["n_max"].append(n_max.detach().cpu().numpy())
        self.history["T"].append(T.detach().cpu().numpy())

        return x_incumbent, z_incumbent, s_primal

    def _plot_tree(self, batch_index=0):
        """
        Plots the tree for a given batch index.

        Args:
            batch_index: The index of the batch to plot.

        Returns:
            Returns the plot as a matplotlib figure.

        """
        fig = plt.figure(layout='constrained', figsize=(self.n_samples * 1.5, self.n_hypotheses * 2 * len(self.transformation)))
        subfigs = fig.subfigures(len(self.transformation), 1, wspace=0.07)
        subfigs = subfigs if len(self.transformation) > 1 else [subfigs, ]

        # normalize scores globally over the entire tree
        # scores: level x (B x K x O)
        stacked_tensor = np.stack(self.history["score"]).transpose((1, 0, 2, 3)) # bring batch up front
        min_val = stacked_tensor.reshape(self.batch_size, -1).min(axis=-1)[:, None, None]
        max_val = stacked_tensor.reshape(self.batch_size, -1).max(axis=-1)[:, None, None]

        for i in range(len(self.transformation)):
            scores = (self.history["score"][i] - min_val) / (max_val - min_val)
            self._plot_level(level=i, batch_index=batch_index, fig=subfigs[i],
                             orbit_samples=self.history["orbit"][i],
                             embedding=self.history["embedding"][i],
                             candidates=self.history["n_max"][i],
                             scores=scores)
        return fig

    def _plot_level(self, fig, orbit_samples: Tensor, embedding: Tensor, scores: Tensor,
                     candidates: Tensor, level: int, batch_index=0):
        """
        Plots on level of the search tree.

        Args:
            fig: The figure to plot on.
            orbit_samples: The samples of the orbit.
            embedding: The embedding.
            scores: The scores.
            candidates: The candidates.
            level: The level of the tree.
            batch_index: The batch index.

        Returns:
            None.
        """
        ax = fig.subplots(self.n_hypotheses if level != 0 else 1, self.n_samples)
        ax = ax if level != 0 else ax[None]

        # map scores to colors
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', '#176fc1'], N=256)
        colors = cmap(scores)

        # loop over branch stacks
        for k in range(self.n_hypotheses if level != 0 else 1):
            label_idx = np.argmax(embedding[batch_index, k], axis=-1)

            # loop over orbit samples (elements in a branch stack)
            for s in range(self.n_samples):

                # extend the image by a box above
                im = orbit_samples[batch_index, k, s].transpose((1, 2, 0))
                im = np.tile(im, (1, 1, 3)) if im.shape[-1] == 1 else im

                # some pre-processors do not map to [0,1]
                if im.min() != 0.:
                    im = (im - im.min()) / (im.max() - im.min())

                # add score-colored rectangle
                color_rec = np.tile(np.array(colors[batch_index, k, s, :3])[None, None],(5*self.line_thickness, im.shape[1], 1))
                im = np.concatenate([color_rec, im], axis=0)

                # plot the image
                ax[k, s].imshow(im, cmap='gray')
                ax[k, s].set_xlim(0, im.shape[1])
                ax[k, s].set_ylim(im.shape[0], 0)
                ax[k, s].axis('off')

                # add predicted label
                pred_label = str(self.labels[label_idx[s]]).split(',')[0]
                ax[k, s].text(im.shape[1] / 2, 2.5 if self.line_thickness == 1 else 7.,
                              pred_label, color='black', ha='center', va='center',
                              weight='bold', fontsize=self.fontsize)

                # highlight candidates
                if ((level == 0 and np.any(s == candidates[batch_index]))
                        or (level > 0 and s == candidates[batch_index, k])):
                    highlight_subplot(ax[k, s], plt.get_cmap('coolwarm')(1.), 1*self.line_thickness)
