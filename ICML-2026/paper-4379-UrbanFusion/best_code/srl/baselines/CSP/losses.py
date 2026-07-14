"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
"""

import math

import torch
import utils as ut


def bce_loss(pred):
    return -torch.log(pred + 1e-5)


def embed_l2_normalize(embed, dim=-1):
    """
    embedding L2 normalize
    """
    norm = torch.norm(embed, dim=dim, keepdim=True)
    return embed / norm


def rand_samples(batch_size, params, rand_type="uniform"):
    """
    randomly sample background locations, generate (lon, lat, date) and put into pre loc encoder
    Note that the generated (lon, lat) are between [-1, 1] for geo_net
    But for our spa_enc, they generate real (lat, lon)
    Return:
        rand_feats: shape (batch_size, input_feat_dim)
    """
    spa_enc_type = params["spa_enc_type"]

    # randomly sample background locations and date
    # the generated location and date from [-1, 1]
    rand_feats_orig = torch.rand(batch_size, 3).to(params["device"]) * 2 - 1

    # this is the version used in the ICCV paper - it introduces some biases at poles
    # randomly sample background locations
    if rand_type == "sphericalold":
        # theta is between (0, 2*pi), computed based on latitude
        theta = ((rand_feats_orig[:, 1].unsqueeze(1) + 1) / 2.0) * (
            2 * math.pi
        )
        r_lon = torch.sqrt(
            1.0 - rand_feats_orig[:, 0].unsqueeze(1) ** 2
        ) * torch.cos(theta)
        r_lat = torch.sqrt(
            1.0 - rand_feats_orig[:, 0].unsqueeze(1) ** 2
        ) * torch.sin(theta)
        # rand_feats_orig: (batch_size, 3)
        rand_feats_orig = torch.cat(
            (r_lon, r_lat, rand_feats_orig[:, 2].unsqueeze(1)), 1
        )

    # randomly sample background locations
    if rand_type == "spherical":
        """
        See https://math.stackexchange.com/a/1586185 to randomly generate points on a unit sphere
        """
        rand_feats_orig = torch.rand(batch_size, 3).to(params["device"])
        rand_feats_orig[:, 2] = (
            rand_feats_orig[:, 2] * 2.0 - 1.0
        )  # make dates between -1 and 1
        # theta1: (0, 2*pi), this is the correct lon on a unit sphere
        theta1 = 2.0 * math.pi * rand_feats_orig[:, 0]
        # theta2: in (0, pi), theta2- pi/2 is the corrent lat on a unit sphere
        theta2 = torch.acos(2.0 * rand_feats_orig[:, 1] - 1.0)
        # lat: in (-1, 1), normalized lat in (-1, 1) for ut.encode_loc_time()
        lat = 1.0 - 2.0 * theta2 / math.pi
        # lon: in (-1, 1),normalized lon in (-1, 1) for ut.encode_loc_time()
        lon = (theta1 / math.pi) - 1.0
        # rand_feats: shape (batch_size, 3)
        #      three dim: [lon, lat, time], all noramlized to [-1, 1]
        rand_feats = torch.cat(
            (
                lon.unsqueeze(1),
                lat.unsqueeze(1),
                rand_feats_orig[:, 2].unsqueeze(1),
            ),
            1,
        )

    if spa_enc_type == "geo_net":
        rand_feats = ut.encode_loc_time(
            rand_feats_orig[:, :2],
            rand_feats_orig[:, 2],
            concat_dim=1,
            params=params,
        )

    elif spa_enc_type in ut.get_spa_enc_list():
        lon = torch.unsqueeze(rand_feats_orig[:, 0] * 180, dim=1)
        lat = torch.unsqueeze(rand_feats_orig[:, 1] * 90, dim=1)
        # rand_feats: shape (batch_size, input_feat_dim = 2)
        rand_feats = torch.cat((lon, lat), 1).to(params["device"])
    else:
        raise Exception("spa_enc not defined!!!")

    return rand_feats


def l2regress_loss(model, params, loc_feat, cnn_features, inds):
    """
    We are doing l2regress loss, given loc_feat, encode it into location embedding,
    Then dec it to 2048 and match with cnn_features
    Args:
        model:
        param:
        loc_feat: shape (batch_size, input_feat_dim)
        cnn_features: shape (batch_size, cnn_feat_dim = 2048)
        inds: tensor, [0,1,2,...,batch_size-1]
    """
    # we are doing l2regress
    assert params["unsuper_loss"] == "l2regress"
    # the model has loc_dec
    assert "loc_dec" in vars(model)["_modules"]

    batch_size = loc_feat.shape[0]

    # loc_emb: shape (batch_size, num_filts)
    loc_emb = model(loc_feat, return_feats=True)
    # loc_cnn_preds: shape (batch_size, cnn_feat_dim = 2048)
    loc_cnn_preds = model.loc_dec(loc_emb)

    mseloss = torch.nn.MSELoss(reduction="mean")

    loss = mseloss(loc_cnn_preds, cnn_features)
    return loss


def contsoftmax_loss(model, params, loc_feat, cnn_features, inds):
    """
    We are doing contrastive loss, given loc_feat, encode it into location embedding,
    Then the cnn_features are projected to num_files dimention and compare with location embeddings

    All loss are following the contrastive loss (softmax) objective
    1. Location Image Loss: (X, I) and (X, I'),  in batch loss, I' is negative image from the same batch
    2. Location Negative Sampling Loss: (X, I) and (X^{-}, I) loss, X^{-} are randomly sampled negative location
    3. SimCSE loss: (X, X^{+}) and (X, X^{+}') loss, X^{+} are another forward pass of the same X, X^{+}' is another X from the same batch

    Args:
        model:
        param:
        loc_feat: shape (batch_size, input_feat_dim)
        cnn_features: shape (batch_size, cnn_feat_dim = 2048)
        inds: tensor, [0,1,2,...,batch_size-1]
    """

    assert params["unsuper_temp_inbatch"] > 0
    assert params["unsuper_temp_negloc"] > 0
    assert params["unsuper_temp_simcse"] > 0

    # we are doing imgcontloss
    assert "contsoftmax" in params["unsuper_loss"]
    # the model has loc_dec
    assert "img_dec" in vars(model)["_modules"]

    loss_crsent = torch.nn.CrossEntropyLoss()

    batch_size = loc_feat.shape[0]

    # loc_emb: shape (batch_size, num_filts)
    loc_emb = model(loc_feat, return_feats=True)
    loc_emb_norm = embed_l2_normalize(embed=loc_emb, dim=-1)

    # cnn_loc_emb: shape (batch_size, num_filts), the predicted location embedding from image CNN features
    cnn_loc_emb = model.img_dec(cnn_features)
    cnn_loc_emb_norm = embed_l2_normalize(embed=cnn_loc_emb, dim=-1)

    ######################## 1. in batch loss ##################################
    # 1. compute cosine similarity (X, I) and (X, I')
    # compute the in batch similarity beween each image embedding and each location embedding
    # loc_img_sims: shape (batch_size, batch_size)
    loc_img_sims = torch.matmul(
        loc_emb_norm, torch.transpose(cnn_loc_emb_norm, 0, 1)
    )

    # add temperature value
    loc_img_sims_ = loc_img_sims / params["unsuper_temp_inbatch"]

    loc_img_labels = inds[:batch_size]
    # 1. in batch loss, contrastive learning
    # Contrast loc for all image in batch
    loss_inbatch = loss_crsent(loc_img_sims_, loc_img_labels)
    if params["unsuper_loss"] == "contsoftmaxsym":
        # contrast image with all location in batch
        loss_inbatch += loss_crsent(
            torch.transpose(loc_img_sims_, 0, 1), loc_img_labels
        )

    ######################## 2. negative location loss ##################################
    # 2. compute similarity (X', I)
    # create random background samples
    # loc_feat_rand: (batch_size * num_neg_rand_loc, input_feat_dim)
    loc_feat_rand = rand_samples(
        batch_size * params["num_neg_rand_loc"],
        params,
        rand_type=params["neg_rand_type"],
    )

    # loc_emb_rand: shape (batch_size * num_neg_rand_loc, num_filts),
    #               the location embedding for random selected samples
    loc_emb_rand = model(loc_feat_rand, return_feats=True)
    loc_emb_rand_norm = embed_l2_normalize(embed=loc_emb_rand, dim=-1)

    # loc_emb_rand_norm_: shape (num_neg_rand_loc, batch_size, num_filts),
    loc_emb_rand_norm_ = torch.reshape(
        loc_emb_rand_norm, (params["num_neg_rand_loc"], batch_size, -1)
    )

    # loc_emb_norm_sq: shape (1, batch_size, num_filts)
    loc_emb_norm_sq = loc_emb_norm.unsqueeze(0)

    # loc_emb_norm_cat: shape (1 + num_neg_rand_loc, batch_size, num_filts)
    loc_emb_norm_cat = torch.cat([loc_emb_norm_sq, loc_emb_rand_norm_], dim=0)

    # loc_rand_img_sims: shape (1 + num_neg_rand_loc, batch_size), sim(X', I)
    loc_rand_img_sims = torch.sum(
        torch.einsum("nbd,bd->nbd", loc_emb_norm_cat, cnn_loc_emb_norm),
        dim=-1,
        keepdim=False,
    )

    # loc_rand_img_sims_: shape (batch_size, 1 + num_neg_rand_loc), sim(X', I),
    loc_rand_img_sims_ = (
        torch.transpose(loc_rand_img_sims, 0, 1)
        / params["unsuper_temp_negloc"]
    )

    # loc_rand_labels: shape (batch_size), the 1st one is positive
    loc_rand_labels = torch.zeros(batch_size).long().to(params["device"])

    # 2. negative location loss, contrastive loss
    loss_negloc = loss_crsent(loc_rand_img_sims_, loc_rand_labels)

    ######################## 3. SimCSE loss ##################################
    # loc_emb: shape (batch_size, num_filts), same location embedding with different dropout masks
    loc_emb_ = model(loc_feat, return_feats=True)
    loc_emb_norm_ = embed_l2_normalize(embed=loc_emb_, dim=-1)

    # compute the in batch similarity beween each location embedding and their anpther dropout version -> X * X^{+}
    # loc_loc_sims: shape (batch_size, batch_size),
    loc_loc_sims = torch.matmul(
        loc_emb_norm, torch.transpose(loc_emb_norm_, 0, 1)
    )

    # add temperature value
    loc_loc_sims_ = loc_loc_sims / params["unsuper_temp_simcse"]

    loc_loc_labels = inds[:batch_size]

    # 1. in batch loss, contrastive learning
    loss_simcse = loss_crsent(loc_loc_sims_, loc_loc_labels)

    loss = (
        loss_inbatch
        + loss_negloc * params["rand_sample_weight"]
        + loss_simcse * params["simcse_weight"]
    )

    return loss


def imgcontloss_loss(model, params, loc_feat, cnn_features, inds):
    """
    We are doing imgcontloss loss, given loc_feat, encode it into location embedding,
    Then the cnn_features are projected to num_files dimention and compare with location embeddings
    Args:
        model:
        param:
        loc_feat: shape (batch_size, input_feat_dim)
        cnn_features: shape (batch_size, cnn_feat_dim = 2048)
        inds: tensor, [0,1,2,...,batch_size-1]
    """
    # we are doing imgcontloss
    assert "imgcontloss" in params["unsuper_loss"]
    # the model has loc_dec
    assert "img_dec" in vars(model)["_modules"]

    batch_size = loc_feat.shape[0]

    # loc_emb: shape (batch_size, num_filts)
    loc_emb = model(loc_feat, return_feats=True)
    loc_emb_norm = embed_l2_normalize(embed=loc_emb, dim=-1)

    # cnn_loc_emb: shape (batch_size, num_filts), the predicted location embedding from image CNN features
    cnn_loc_emb = model.img_dec(cnn_features)
    cnn_loc_emb_norm = embed_l2_normalize(embed=cnn_loc_emb, dim=-1)

    # 1. compute similarity (X, I) and (X, I')
    # compute the in batch similarity beween each image embedding and each location embedding
    # loc_img_sims: shape (batch_size, batch_size)
    loc_img_sims = torch.matmul(
        loc_emb_norm, torch.transpose(cnn_loc_emb_norm, 0, 1)
    )

    # sig_loc_img: shape (batch_size, batch_size)
    sig_loc_img = torch.sigmoid(loc_img_sims)

    # 1.1 positive loss, -log(sigmoid( sim(X, I) ))
    # loss_pos: shape (batch_size), -log(sigmoid( sim(X, I) ))
    loss_pos = bce_loss(sig_loc_img[inds[:batch_size], inds[:batch_size]])
    pos_weight = batch_size - 1

    # the others are similarity (X, I')
    # make the diagnal items 0, so -log(1 - x) = 0, not affect loss
    # sig_loc_img[inds[:batch_size], inds[:batch_size]] = 0

    # loss_neg_img: shape (batch_size, batch_size), -log(1 - sigmoid( sim(X, I') ))
    loss_neg_img = bce_loss(1.0 - sig_loc_img)

    loss_neg_img[inds[:batch_size], inds[:batch_size]] = pos_weight * loss_pos

    """
    1.1 positive loss, -log(sigmoid( sim(X, I) ))
    1.2 negative image loss, mean( -log(1 - sigmoid( sim(X, I') )) ), the average loss of (batch_size-1) egative samples
    loss_pos_neg_img: shape (batch_size, ),  -log(sigmoid( sim(X, I) ))  -log(1 - sigmoid( sim(X, I') ))
    """
    loss_pos_neg_img = torch.sum(loss_neg_img, dim=-1, keepdim=False) * (
        1.0 / (batch_size - 1)
    )

    if params["unsuper_loss"] in ["imgcontloss", "imgcontlosssimcse"]:
        # 2. compute similarity (X', I)
        # create random background samples
        # loc_feat_rand: (batch_size * num_neg_rand_loc, input_feat_dim)
        loc_feat_rand = rand_samples(
            batch_size * params["num_neg_rand_loc"],
            params,
            rand_type=params["neg_rand_type"],
        )

        # loc_emb_rand: shape (batch_size * num_neg_rand_loc, num_filts),
        #               the location embedding for random selected samples
        loc_emb_rand = model(loc_feat_rand, return_feats=True)
        loc_emb_rand_norm = embed_l2_normalize(embed=loc_emb_rand, dim=-1)

        # loc_emb_rand_norm_: shape (num_neg_rand_loc, batch_size, num_filts),
        loc_emb_rand_norm_ = torch.reshape(
            loc_emb_rand_norm, (params["num_neg_rand_loc"], batch_size, -1)
        )

        # loc_rand_img_sims: shape (num_neg_rand_loc, batch_size), sim(X', I)
        loc_rand_img_sims = torch.sum(
            torch.einsum("nbd,bd->nbd", loc_emb_rand_norm_, cnn_loc_emb_norm),
            dim=-1,
            keepdim=False,
        )

        # 2. negative location loss, -log(1 - sigmoid( sim(X', I) ))
        # loss_loc_rand: shape (num_neg_rand_loc, batch_size), -log(1 - sigmoid( sim(X', I) ))
        loss_loc_rand = bce_loss(1.0 - torch.sigmoid(loc_rand_img_sims))

        # loss_loc_rand_mean: shape (batch_size), -log(1 - sigmoid( sim(X', I) ))
        loss_loc_rand_mean = torch.mean(loss_loc_rand, dim=0, keepdim=False)

        if params["unsuper_loss"] == "imgcontloss":
            loss = (
                loss_pos_neg_img.mean()
                + loss_loc_rand_mean.mean() * params["rand_sample_weight"]
            )
        elif params["unsuper_loss"] == "imgcontlosssimcse":
            # loc_emb: shape (batch_size, num_filts)
            loc_emb_ = model(loc_feat, return_feats=True)
            loc_emb_norm_ = embed_l2_normalize(embed=loc_emb_, dim=-1)

            # compute the in batch similarity beween each location embedding and their anpther dropout version -> X * X^{+}
            # loc_loc_sims: shape (batch_size, batch_size),
            loc_loc_sims = torch.matmul(
                loc_emb_norm, torch.transpose(loc_emb_norm_, 0, 1)
            )

            # sig_loc_loc: shape (batch_size, batch_size)
            sig_loc_loc = torch.sigmoid(loc_loc_sims)

            # 3.1 positive loss, -log(sigmoid( sim(X, X^{+}) ))
            # loss_loc_loc_pos: shape (batch_size), -log(sigmoid( sim(X, X^{+}) ))
            loss_loc_loc_pos = bce_loss(
                sig_loc_loc[inds[:batch_size], inds[:batch_size]]
            )
            # pos_weight = batch_size - 1

            # loss_loc_loc_neg: shape (batch_size, batch_size), -log(1 - sigmoid( sim(X, X^{+}') ))
            loss_loc_loc_neg = bce_loss(1.0 - sig_loc_loc)

            loss_loc_loc_neg[inds[:batch_size], inds[:batch_size]] = (
                batch_size - 1
            ) * loss_loc_loc_pos

            """
            3.1 positive loss, -log(sigmoid( sim(X, X^{+}) ))
            3.2 negative image loss, mean( -log(1 - sigmoid( sim(X, X^{+}') )) ), the average loss of (batch_size-1) egative samples
            loss_pos_neg_img: shape (batch_size, ),  -log(sigmoid( sim(X, I) ))  -log(1 - sigmoid( sim(X, I') ))
            """
            loss_loc_loc = torch.sum(
                loss_loc_loc_neg, dim=-1, keepdim=False
            ) * (1.0 / (batch_size - 1))

            loss = (
                loss_pos_neg_img.mean()
                + loss_loc_rand_mean.mean() * params["rand_sample_weight"]
                + loss_loc_loc.mean() * params["simcse_weight"]
            )

    elif params["unsuper_loss"] == "imgcontlossnolocneg":
        loss = loss_pos_neg_img.mean()
    return loss


def imgcontloss_eval(model, params, loc_feat, cnn_features, inds):
    """
    For imgcontloss loss, given loc_feat, encode it into location embedding,
    Then the cnn_features are projected to num_files dimention and compare with location embeddings

    Here, we compute the cosine similarity between location embedding and image embedding
    Args:
        model:
        param:
        loc_feat: shape (batch_size, input_feat_dim)
        cnn_features: shape (batch_size, cnn_feat_dim = 2048)
        inds: tensor, [0,1,2,...,batch_size-1]
    """

    # we are doing imgcontloss
    assert "imgcontloss" in params["unsuper_loss"]
    # the model has loc_dec
    assert "img_dec" in vars(model)["_modules"]

    batch_size = loc_feat.shape[0]

    # loc_emb: shape (batch_size, num_filts)
    loc_emb = model(loc_feat, return_feats=True)
    loc_emb_norm = embed_l2_normalize(embed=loc_emb, dim=-1)

    # cnn_loc_emb: shape (batch_size, num_filts), the predicted location embedding from image CNN features
    cnn_loc_emb = model.img_dec(cnn_features)
    cnn_loc_emb_norm = embed_l2_normalize(embed=cnn_loc_emb, dim=-1)

    # loc_img_sims: shape (batch_size, )
    loc_img_sims = torch.sum(
        loc_emb_norm * cnn_loc_emb_norm, dim=-1, keepdim=False
    )

    # loss_pos: shape (batch_size), -log(sigmoid( sim(X, I) ))
    loss_pos = bce_loss(torch.sigmoid(loc_img_sims))

    return loss_pos.mean()


def embedding_loss(
    model,
    params,
    loc_feat,
    loc_class,
    user_ids,
    inds,
    neg_rand_type="spherical",
):
    """
    Args:
        model:
        param:
        loc_feat: shape (batch_size, input_feat_dim)
        loc_class: shape (batch_size)
        user_ids: shape (batch_size)
        inds: tensor, [0,1,2,...,batch_size-1]
    """

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # create random background samples
    # loc_feat_rand: (batch_size, input_feat_dim)
    loc_feat_rand = rand_samples(batch_size, params, rand_type=neg_rand_type)

    # get location embeddings
    # loc_cat: (2*batch_size, input_feat_dim)
    loc_cat = torch.cat((loc_feat, loc_feat_rand), 0)
    loc_emb_cat = model(loc_cat, return_feats=True)
    # the location embedding for training samples, (batch_size, num_filts)
    loc_emb = loc_emb_cat[:batch_size, :]
    # the location embedding for random selected samples, (batch_size, num_filts)
    loc_emb_rand = loc_emb_cat[batch_size:, :]

    # the prediction distribution for training samples, (batch_size, num_classes)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    # the prediction distribution for random selected samples, (batch_size, num_classes)
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # data loss
    # see equation 7 in paper https://arxiv.org/abs/1906.05272
    pos_weight = params["num_classes"]
    # loss_pos: (batch_size, num_classes)
    loss_pos = bce_loss(1.0 - loc_pred)  # neg
    # update probability at the training sample's correct class
    loss_pos[inds[:batch_size], loc_class] = pos_weight * bce_loss(
        loc_pred[inds[:batch_size], loc_class]
    )  # pos
    loss_bg = bce_loss(1.0 - loc_pred_rand)

    if "user" in params["train_loss"]:

        # user location loss
        # see equation 8 in paper https://arxiv.org/abs/1906.05272

        # note: self.user_emb.weight shape (num_users, num_filts)
        # get the user embedding for each data sample
        # user: (batch_size, num_filts)
        user = model.user_emb.weight[user_ids, :]
        # p_u_given_l/p_u_given_randl:  (batch_size)
        p_u_given_l = torch.sigmoid((user * loc_emb).sum(1))
        p_u_given_randl = torch.sigmoid((user * loc_emb_rand).sum(1))

        # user_loc_pos_loss/user_loc_neg_loss: (batch_size)
        user_loc_pos_loss = bce_loss(p_u_given_l)
        user_loc_neg_loss = bce_loss(1.0 - p_u_given_randl)

        # user class loss
        # see equation 9 in paper https://arxiv.org/abs/1906.05272
        # p_c_given_u: (batch_size, num_classes)
        p_c_given_u = torch.sigmoid(
            torch.matmul(user, model.class_emb.weight.transpose(0, 1))
        )
        # user_class_loss: (batch_size, num_classes)
        user_class_loss = bce_loss(1.0 - p_c_given_u)
        user_class_loss[inds[:batch_size], loc_class] = pos_weight * bce_loss(
            p_c_given_u[inds[:batch_size], loc_class]
        )

        # total loss
        loss = (
            loss_pos.mean()
            + loss_bg.mean()
            + user_loc_pos_loss.mean()
            + user_loc_neg_loss.mean()
            + user_class_loss.mean()
        )

    else:

        # total loss
        loss = loss_pos.mean() + loss_bg.mean() * params["rand_sample_weight"]

    return loss
