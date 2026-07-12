# Taken from https://github.com/acerbilab/amortized-conditioning-engine/blob/main/src/model/embedder.py
import torch
import torch.nn as nn
from .utils import positional_encoding_init, build_mlp, build_mlp_with_linear_skipcon
from typing import Optional, List

class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{name!r} not found")
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"{name!r} not found")



class NonLinearEmbedder(nn.Module):
    """
    Non-Linear TNPD embedder used in TNP paper.

    This class defines a non-linear embedder that constructs an input tensor
    from a batch of context and target data, and then passes it through a
    multi-layer perceptron (MLP) to produce embeddings.

    Attributes:
        embedder (nn.Module): A multi-layer perceptron (MLP) model for embedding.
        name (str): The name of the embedder.
    """

    def __init__(
        self,
        dim_xc: int,
        dim_yc: int,
        num_latent: int,
        dim_hid: int,
        dim_out: int,
        emb_depth: int,
        name: str,
    ) -> None:
        """
        Initializes the NonLinearEmbedder class with the specified dimensions
        and model parameters.

        Args:
            dim_xc (int): Dimension of context inputs including markers.
            dim_yc (int): Dimension of context outputs.
            num_latent (int): Number of latent dimensions (not currently used).
            dim_hid (int): Dimension of the hidden layers in the MLP.
            dim_out (int): Dimension of the output of the MLP.
            emb_depth (int): Depth of the MLP (number of layers).
            name (str): Name of the embedder.
        """

        super().__init__()

        # -1 for removing markers from context inputs
        self.embedder = build_mlp(dim_xc - 1 + dim_yc, dim_hid, dim_out, emb_depth)
        self.name = name

    def construct_input(self, batch: AttrDict) -> torch.Tensor:
        """
        Constructs the input tensor for the embedder from the batch data. For this
        embedder, it is expected that no latents are present in the batch.

        Args:
            batch (AttrDict): A batch of data containing context and target information.
                The batch is expected to have attributes `xc`, `yc`, `xt`, `yt`.

        Returns:
            torch.Tensor: A tensor that concatenates context and target data,
                formatted to be fed into the MLP embedder.
        """
        # remove markers for fair comparison with ours
        x_y_ctx = torch.cat((batch.xc[:, :, 1:], batch.yc), dim=-1)
        x_0_tar = torch.cat((batch.xt[:, :, 1:], torch.zeros_like(batch.yt)), dim=-1)
        inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        return inp

    def forward(self, batch: AttrDict) -> torch.Tensor:
        """
        Performs a forward pass of the embedder with the given batch of data.

        Args:
            batch (AttrDict): A batch of data containing context and target information.
                The batch is expected to have attributes `xc`, `yc`, `xt`, `yt`.

        Returns:
            torch.Tensor: The embedded output after passing the constructed input
                through the MLP embedder.
        """

        inp = self.construct_input(batch)
        return self.embedder(inp)


class EmbedderMarker(nn.Module):
    """
    An embedder class that handles embedding for context and target data with both
    continuous and discrete values, using markers to distinguish between them.

    Attributes:
        embedder_marker (nn.Embedding): Embedding layer for markers.
        embedder_discrete (Optional[nn.Embedding]): Embedding layer for discrete
            values, if specified.
        embedderx (nn.Module): Embedding network for continuous context inputs.
        embedderyc (nn.Module): Embedding network for continuous context outputs.
        name (Optional[str]): Name of the embedder.
        discrete_index (Optional[List[int]]): List of discrete indices, if applicable.
    """

    def __init__(
        self,
        dim_xc: int,
        dim_yc: int,
        num_latent: int,
        dim_hid: int,
        dim_out: int,
        emb_depth: int,
        name: Optional[str] = None,
        pos_emb_init: bool = False,
        use_skipcon_mlp: bool = False,
        discrete_index: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the EmbedderMarker class with specified parameters for embedding
        layers.

        Args:
            dim_xc (int): Dimension of context inputs including markers.
            dim_yc (int): Dimension of context outputs.
            num_latent (int): Number of latent dimensions.
            dim_hid (int): Dimension of the hidden layers in the MLP.
            dim_out (int): Dimension of the output embeddings.
            emb_depth (int): Depth of the MLP (number of layers).
            name (Optional[str]): Name of the embedder.
            pos_emb_init (bool): Whether to initialize positional embeddings.
            use_skipcon_mlp (bool): Whether to use MLP with linear skip connections.
            discrete_index (Optional[List[int]]): List of indices for discrete embeddings.
        """
        super().__init__()
        self.embedder_marker = nn.Embedding(2 + num_latent, dim_out)  # E_c
        if discrete_index:
            self.embedder_discrete = nn.Embedding(len(discrete_index), dim_out)

        if use_skipcon_mlp:
            self.embedderx = build_mlp_with_linear_skipcon(
                dim_xc - 1, dim_hid, dim_out, emb_depth
            )  # f_cov
            self.embedderyc = build_mlp_with_linear_skipcon(
                dim_yc, dim_hid, dim_out, emb_depth
            )  # f_val
        else:
            self.embedderx = build_mlp(dim_xc - 1, dim_hid, dim_out, emb_depth)  # f_cov
            self.embedderyc = build_mlp(dim_yc, dim_hid, dim_out, emb_depth)  # f_val

        self.name = name
        self.discrete_index = discrete_index

        # positional embedding initialization
        if pos_emb_init:
            self.embedder_marker.weight = positional_encoding_init(
                2 + num_latent, dim_out, 2 + num_latent
            )

    def forward(self, batch):
        """
        Performs a forward pass of the embedder with the given batch of data.

        Args:
            batch (AttrDict): A batch of data containing context and target information.
                The batch is expected to have attributes `xc`, `yc`, `xt`, `yt`.

        Returns:
            torch.Tensor: The embedded output after passing the constructed input
                through the MLP embedder.
        """
        # xc and yc is context:
        # xc is either a continuous variable or 0 for latent
        # xce is a marker 1 for data >1 for latent

        if torch.any(torch.isnan(batch.yc)):
            print("yc contains NaN values:", batch.yc)

        # batch.xc [B, T, [marker, x1, x2, x3, ...]] where marker is 1 for data.
        # batch.xc [B, T, [marker, 0, 0, ...]] for latent marker  > 1 for latent.
        xc = batch.xc[:, :, 1:]
        xce = batch.xc[:, :, :1]
        yc = batch.yc

        xt = batch.xt[:, :, 1:]
        xte = batch.xt[:, :, :1]
        yt = batch.yt[:, :, -1:]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create the mask
        if self.discrete_index:
            discrete_values = torch.tensor(self.discrete_index, device=device)
            mask_discrete = torch.isin(batch.yc, torch.tensor(discrete_values)).float()
            inverse_mask_discrete = 1 - mask_discrete
        else:
            mask_discrete = torch.zeros_like(batch.yc, device=device).float()

        mask_context_x = (xce.int() == 1).float()

        if self.discrete_index:
            context_embedding = (
                torch.add(
                    self.embedderx(xc) * mask_context_x,  # Embedd for continuous x data
                    self.embedderyc(yc)
                    * inverse_mask_discrete,  # Embedd continuous y data
                )
                + self.embedder_discrete((yc * mask_discrete)[:, :, 0].int())
                * mask_discrete
            )  # Embedd discrete y data
        else:
            if xc.shape[-1] == 0:
                context_embedding = self.embedderyc(yc)
            else:
                context_embedding = torch.add(
                    self.embedderx(xc) * mask_context_x,
                    self.embedderyc(yc),
                )

        context_embedding = torch.add(
            context_embedding, self.embedder_marker(xce[:, :, 0].int())
        )  # Embedd for markers [1,2,3,...]

        # add xt and marker_? embeddings
        # set embedderx output to zero when point in target set is latent
        mask_target_x = (xte.int() == 1).float()

        if xt.shape[-1] == 0:
            target_embedding = self.embedder_marker(
                torch.zeros_like(yt).int()[
                    :, :, 0
                ]  # we use the 0th index of the embedder for "?" marker.
            )
        else:
            target_embedding = torch.add(
                self.embedderx(xt) * mask_target_x,
                self.embedder_marker(
                    torch.zeros_like(yt).int()[
                        :, :, 0
                    ]  # we use the 0th index of the embedder for "?" marker.
                ),
            )

        target_embedding = torch.add(
            target_embedding,
            self.embedder_marker(xte[:, :, 0].int()),  # Embedd for markers [1,2,3,...]
        )  # Embedd for markers [1,2,3,...]

        res = torch.cat((context_embedding, target_embedding), dim=1)
        return res


class EmbedderMarkerPrior(nn.Module):
    def __init__(
        self,
        dim_xc: int,
        dim_yc: int,
        num_latent: int,
        dim_hid: int,
        dim_out: int,
        emb_depth: int,
        num_bins: int,
        name: Optional[str] = None,
        pos_emb_init: bool = False,
        use_skipcon_mlp: bool = False,
        discrete_index: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the EmbedderMarkerPrior module, which embeds context and
        target points using various embeddings based on the type of data
        (continuous, discrete, latent).

        Args:
            dim_xc (int): Dimension of the context features excluding the marker.
            dim_yc (int): Dimension of the context labels.
            num_latent (int): Number of latent points.
            dim_hid (int): Dimension of the hidden layers in the MLPs.
            dim_out (int): Output dimension of the embeddings.
            emb_depth (int): Depth of the MLP used for embedding.
            num_bins (int): Number of bins used in the latent distribution.
            name (Optional[str]): Optional name for the embedder.
            pos_emb_init (bool): If True, initialize positional embeddings.
            use_skipcon_mlp (bool): If True, use MLP with linear skip connections.
            discrete_index (Optional[List[int]]): Indices for discrete embeddings,
                if any.

        Attributes:
            embedder_marker (nn.Embedding): Embedding for markers with dimension
                [2 + num_latent, dim_out].
            embedder_discrete (nn.Embedding): Embedding for discrete indices if
                provided.
            embedderx (nn.Module): MLP for embedding x context features.
            embedderyc (nn.Module): MLP for embedding y context labels.
            embedderbin (nn.Module): MLP for embedding bin weights.
            name (Optional[str]): Optional name for the embedder.
            discrete_index (Optional[List[int]]): Indices for discrete embeddings,
                if any.
            num_latent (int): Number of latent points.
            mlp_dim_out (int): Output dimension of the MLP used in embeddings.

        """
        super().__init__()
        self.embedder_marker = nn.Embedding(2 + num_latent, dim_out)  # E_c
        if discrete_index:
            self.embedder_discrete = nn.Embedding(len(discrete_index), dim_out)

        if use_skipcon_mlp:
            self.embedderx = build_mlp_with_linear_skipcon(
                dim_xc - 1, dim_hid, dim_out, emb_depth
            )  # f_cov
            self.embedderyc = build_mlp_with_linear_skipcon(
                dim_yc, dim_hid, dim_out, emb_depth
            )  # f_val
            self.embedderbin = build_mlp_with_linear_skipcon(
                num_bins, dim_hid, dim_out, emb_depth
            )
        else:
            self.embedderx = build_mlp(dim_xc - 1, dim_hid, dim_out, emb_depth)  # f_cov
            self.embedderyc = build_mlp(dim_yc, dim_hid, dim_out, emb_depth)  # f_val
            self.embedderbin = build_mlp(
                num_bins, dim_hid, dim_out, emb_depth
            )  # for bin weights

        self.name = name
        self.discrete_index = discrete_index
        self.num_latent = num_latent
        self.mlp_dim_out = dim_out

        # positional embedding initialization
        if pos_emb_init:
            self.embedder_marker.weight = positional_encoding_init(
                2 + num_latent, dim_out, 2 + num_latent
            )

    def forward(self, batch):
        """
        Forward pass to embed context and target points.

        xc and yc is context:
        xc is either a continuous variable or 0 for latent
        xce is a marker 1 for data >1 for latent

        Note for this prior case:

        We assume that we INCLUDE all latents inside batch.xc on the left most
        elements. HOWEVER, we treat the encoding of latent's y differently. For
        latent "observed" in context, we use yc encoding same as y of the data.
        For latent not observed in context set (live in target set), we use prior
        bins in the context. To do this different encoding, we use following
        variables:

        1. batch.latent_bin_weights [batch_size, num_ctx, num_bins] is the prior
        bins where the [:, :num_latents, :] elements are the prior bins for each
        latents and the [:, num_latents:, :] elements are replaced by zeros and
        wont be used

        2. batch.bin_weights_mask [batch_size, num_ctx, 1] is the prior mask, when
        it's True, the elements of the context are used as prior bins instead of
        observed y

        Args:
            batch (AttrDict): A batch of data containing context and target features
                and labels, as well as latent bin weights and masks.

        Returns:
            Tensor: Concatenated embeddings of context and target points with shape
                [batch_size, num_ctx + num_targets, dim_out].
        """

        if torch.any(torch.isnan(batch.yc)):
            print("yc contains NaN values:", batch.yc)

        # batch.xc [B, T, [marker, x1, x2, x3, ...]] where marker is 1 for data.
        # batch.xc [B, T, [marker, 0, 0, ...]] for latent marker  > 1 for latent.

        xc = batch.xc[:, :, 1:]
        xce = batch.xc[:, :, :1]
        yc = batch.yc

        xt = batch.xt[:, :, 1:]
        xte = batch.xt[:, :, :1]
        yt = batch.yt[:, :, -1:]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.embedderbin(batch.latent_bin_weights)
        # Create the mask
        if self.discrete_index:
            discrete_values = torch.tensor(self.discrete_index, device=device)
            mask_discrete = torch.isin(batch.yc, torch.tensor(discrete_values)).float()
            inverse_mask_discrete = 1 - mask_discrete
        else:
            mask_discrete = torch.zeros_like(batch.yc, device=device).float()

        mask_context_x = (xce.int() == 1).float()  # masking for x data
        bin_weights_mask = (
            batch.bin_weights_mask
        )  # masking for latent as prior (non observed y in context)
        inverse_bin_weight_mask = (
            ~bin_weights_mask
        )  # masking for real observed y values in context

        # self.embedderbin(batch.latent_bin_weights)[bin_weights_mask]

        if self.discrete_index:
            context_embedding = (
                torch.add(
                    self.embedderx(xc) * mask_context_x,  # Embedd for continuous x data
                    self.embedderyc(yc)
                    * inverse_bin_weight_mask
                    * inverse_mask_discrete,  # Embedd continuous y data
                )
                + self.embedder_discrete(
                    (yc * inverse_bin_weight_mask * mask_discrete)[:, :, 0].int()
                )
                * mask_discrete
                + self.embedderbin(batch.latent_bin_weights) * bin_weights_mask
            )  # Embedd discrete y data
        else:
            context_embedding = (
                torch.add(
                    self.embedderx(xc) * mask_context_x,
                    self.embedderyc(yc) * inverse_bin_weight_mask,
                )
                + self.embedderbin(batch.latent_bin_weights) * bin_weights_mask
            )

        context_embedding = torch.add(
            context_embedding, self.embedder_marker(xce[:, :, 0].int())
        )  # Embedd for markers [1,2,3,...]

        # add xt and marker_? embeddings
        # set embedderx output to zero when point in target set is latent
        mask_target_x = (xte.int() == 1).float()

        target_embedding = torch.add(
            self.embedderx(xt) * mask_target_x,
            self.embedder_marker(
                torch.zeros_like(yt).int()[
                    :, :, 0
                ]  # we use the 0th index of the embedder for "?" marker.
            ),
        )

        target_embedding = torch.add(
            target_embedding,
            self.embedder_marker(xte[:, :, 0].int()),  # Embedd for markers [1,2,3,...]
        )  # Embedd for markers [1,2,3,...]

        res = torch.cat((context_embedding, target_embedding), dim=1)
        return res


class EmbedderMarkerPriorInjectionBin(nn.Module):
    def __init__(
        self,
        dim_xc,
        dim_yc,
        num_latent,
        num_bins,
        dim_hid,
        dim_out,
        emb_depth,
        name=None,
        pos_emb_init=False,
        use_skipcon_mlp=False,
    ):
        super().__init__()

        self.embedder_marker = nn.Embedding(2 + num_latent, dim_out)  # E_c

        if use_skipcon_mlp:
            self.embedderx = build_mlp_with_linear_skipcon(
                dim_xc - 1, dim_hid, dim_out, emb_depth
            )  # f_cov
            self.embedderyc = build_mlp_with_linear_skipcon(
                dim_yc, dim_hid, dim_out, emb_depth
            )  # f_val
            self.embedderbin = build_mlp_with_linear_skipcon(
                num_bins, dim_hid, dim_out, emb_depth
            )
        else:
            self.embedderx = build_mlp(dim_xc - 1, dim_hid, dim_out, emb_depth)  # f_cov
            self.embedderyc = build_mlp(dim_yc, dim_hid, dim_out, emb_depth)  # f_val
            self.embedderbin = build_mlp(
                num_bins, dim_hid, dim_out, emb_depth
            )  # for bin weights

        self.name = name
        # positional embedding initialization
        if pos_emb_init:
            self.embedder_marker.weight = positional_encoding_init(
                2 + num_latent, dim_out, 2 + num_latent
            )

    def forward(self, batch):
        # data embedding
        embedding_xc_data = self.embedderx(batch.xc_data[:, :, 1:])  # [B, Nc, D]
        embedding_yc_data = self.embedderyc(batch.yc_data)  # [B, Nc, D]

        # known latent embedding
        embedding_yc_known = self.embedderyc(batch.yc_latent_known)  # [B, known, D]

        # unknown latent embedding
        embedding_yc_unknown = self.embedderbin(
            batch.bins_latent_unknown
        )  # [B, unknown, D]

        # concat latent embedding
        embedding_yc_latent = torch.cat(
            (embedding_yc_known, embedding_yc_unknown), dim=1
        )  # [B, Nl, D]

        embedding_data = torch.add(embedding_xc_data, embedding_yc_data)
        # embedding_latent = torch.add(embedding_yc_latent_mean, embedding_yc_latent_std)
        context_embedding = torch.cat((embedding_data, embedding_yc_latent), dim=1)

        # add marker_c embedding
        context_embedding = torch.add(
            context_embedding, self.embedder_marker(batch.xc[:, :, 0].int())
        )

        # add xt and marker_? embeddings
        # set embedderx output to zero when point in target set is latent
        mask_t_latent = (batch.xt[:, :, :1].int() == 1).float()
        target_embedding = torch.add(
            self.embedderx(batch.xt[:, :, 1:]) * mask_t_latent,
            self.embedder_marker(
                torch.zeros_like(batch.yt).int()[:, :, 0]
            ),  # the "?" marker
        )
        # add marker_t embedding
        target_embedding = torch.add(
            target_embedding, self.embedder_marker(batch.xt[:, :, 0].int())
        )
        res = torch.cat((context_embedding, target_embedding), dim=1)
        return res