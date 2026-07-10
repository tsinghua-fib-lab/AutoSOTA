import torch
import copy
import torch.nn.functional as F


class ReverseKL:
    def __init__(self, config, fabric, model, loss_fun, opt):
        super(ReverseKL, self).__init__()
        self.loss_fun = copy.deepcopy(loss_fun)
        model, self.opt = fabric.setup(model, opt)
        model.mark_forward_method("forward_distr")
        self.name = "Reverse-KL"
        self.alpha_reverse = config[
            "alpha_reverse"
        ]  # as the reverse-KL-regularizer scale
        self.alpha_forward = config["alpha_forward"]
        self.augment_softmax = config["augment_softmax"]

    def compute_kl(
        self, mean_s, std_s, sample_s, distr_s, mean_t, std_t, sample_t, distr_t, device
    ):
        mix_coeff_source = torch.distributions.categorical.Categorical(
            torch.ones(mean_s.shape[0], device=device)
        )
        mixture_source = torch.distributions.mixture_same_family.MixtureSameFamily(
            mix_coeff_source, distr_s
        )
        mix_coeff_target = torch.distributions.categorical.Categorical(
            torch.ones(mean_t.shape[0], device=device)
        )
        mixture_target = torch.distributions.mixture_same_family.MixtureSameFamily(
            mix_coeff_target, distr_t
        )
        kl_reg = (
            self.alpha_reverse
            * (
                mixture_target.log_prob(sample_t) - mixture_source.log_prob(sample_t)
            ).mean()
        )
        if self.alpha_forward != 0.0:
            kl_reg += (
                self.alpha_forward
                * (
                    mixture_source.log_prob(sample_s)
                    - mixture_target.log_prob(sample_s)
                ).mean()
            )
        return kl_reg

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        mean_s, std_s, sample_s, out_s, distr_s = model.forward_distr(X_source)
        mean_t, std_t, sample_t, out_t, distr_t = model.forward_distr(X_target)

        out_s = torch.softmax(out_s, 1)
        if self.augment_softmax != 0.0:
            scale_down = 1 - self.augment_softmax * out_s.shape[1]
            out_s = out_s * scale_down + self.augment_softmax
        err = F.nll_loss(torch.log(out_s), torch.argmax(y_source, dim=1))
        # err = self.loss_fun(out_s, y_source)
        err += self.compute_kl(
            mean_s,
            std_s,
            sample_s,
            distr_s,
            mean_t,
            std_t,
            sample_t,
            distr_t,
            fabric.device,
        )

        err.backward()
        self.opt.step()
        self.opt.zero_grad()

    @torch.no_grad()
    def validate(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        mean_s, std_s, sample_s, out_s, distr_s = model.forward_distr(X_source)
        mean_t, std_t, sample_t, out_t, distr_t = model.forward_distr(X_target)
        source_loss = self.loss_fun(out_s, y_source)
        kl_reg = self.compute_kl(
            mean_s,
            std_s,
            sample_s,
            distr_s,
            mean_t,
            std_t,
            sample_t,
            distr_t,
            fabric.device,
        )
        print(f"Source_loss: {source_loss}, kl_reg: {kl_reg}")
