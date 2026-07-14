import inspect
import re
import torch.nn as nn
from .kl import KL
from .layer import LinearBayesian, _BaseBayesianNet


class BayesianModelWrapper(nn.Module):
    def __init__(self, model, posterior=None, kl=None):
        super().__init__()
        if not isinstance(model, nn.Module):
            raise ValueError("`model` must be an instance of nn.Module")

        self._nn = nn.ModuleDict({"model": model, "posterior": posterior})
        self.kl_fn = kl or KL()
        self._alphas = None
        self._warmup_done = False

    def pre_forward(self, *, return_alphas=False, **kwargs):
        if self.posterior is None:
            raise RuntimeError("Posterior not set. Use `add_posterior_net()`.")

        if not self._warmup_done:
            raise RuntimeError("Call warm_up before making the model Bayesian.")

        # posterior forward (might take a subset of kwargs)
        sig = inspect.signature(self.posterior.forward)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        alphas_node, alphas_edge = self.posterior(**filtered_kwargs)

        if alphas_node.shape[-1] != self.num_alphas_node:
            raise RuntimeError(
                "Wrong number of alphas for node update layers."
                f"Expected {self.num_alphas_node} got {alphas_node.shape[-1]}."
            )
        if alphas_edge.shape[-1] != self.num_alphas_edge:
            raise RuntimeError(
                "Wrong number of alphas for edge update layers."
                f"Expected {self.num_alphas_edge} got {alphas_edge.shape[-1]}."
            )

        # broadcast alphas to the model
        self._alphas = {"node": alphas_node, "edge": alphas_edge}

    def forward(self, *, return_alphas=False, **kwargs):
        # perform pre-forward
        self.pre_forward(return_alphas=return_alphas, **kwargs)

        # model forward (might take a subset of kwargs)
        sig = inspect.signature(self.model.forward)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        output = self.model(**filtered_kwargs)

        if return_alphas:
            return {"output": output, "alphas": self._alphas}
        return output

    def predict(self, *, return_alphas=False, **kwargs):  # used in ORB
        # perform pre-forward
        self.pre_forward(return_alphas=return_alphas, **kwargs)

        # model forward (might take a subset of kwargs)
        sig = inspect.signature(self.model.forward)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        output = self.model.predict(**filtered_kwargs)

        if return_alphas:
            return {"output": output, "alphas": self._alphas}
        return output

    def kl_loss(self, edge_index):
        kl_node = self.kl_fn(self._alphas["node"])
        kl_edge = self.kl_fn(self._alphas["edge"], edge_index)
        return kl_node + kl_edge

    def warm_up(
        self,
        num_nodes,
        num_edges,
        regex_pattern=[r".*"],
        **forward_kwargs,
    ):
        # check forward_kwargs signature
        sig = inspect.signature(self.model.forward)
        sig.bind(**forward_kwargs)
        # temporarily attach domain labeling hooks
        handles = [
            m.register_forward_pre_hook(self._domain_hook(num_nodes, num_edges))
            for m in self.model.modules()
            if isinstance(m, nn.Linear)
        ]
        # trigger the hook
        self.model(**forward_kwargs)
        # free hooks
        for handle in handles:
            handle.remove()
        # activate bayesian layers + attach hooks
        self._setup_bayesian_net(regex_pattern)
        self._attach_alpha_hooks()
        # set warmup as finished
        self._warmup_done = True

    def add_posterior_net(self, posterior):
        if not isinstance(posterior, nn.Module):
            raise ValueError("`posterior` must be an instance of nn.Module.")
        self._nn["posterior"] = posterior

    def find_linear_paths(self):
        return [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]

    def _setup_bayesian_net(self, regex_pattern):
        pattern = re.compile("|".join(regex_pattern))

        def get_full_path(module, parent_path=""):
            for name, child in module.named_children():
                full_path = f"{parent_path}.{name}" if parent_path else name
                yield full_path, module, name, child
                yield from get_full_path(child, full_path)

        for full_path, parent_module, name, child in get_full_path(self.model):
            if isinstance(child, nn.Linear) and pattern.search(full_path):
                bayesian_layer = LinearBayesian(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                )
                bayesian_layer._domain = getattr(child, "_domain", "unknown")
                bayesian_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    bayesian_layer.bias.data.copy_(child.bias.data)
                setattr(parent_module, name, bayesian_layer)
            if isinstance(child, nn.Dropout) and pattern.search(full_path):
                setattr(parent_module, name, nn.Dropout(p=0.0))

    def _attach_alpha_hooks(self):
        alpha_counters = {"node": 0, "edge": 0}
        for module in self.model.modules():
            for child in module.children():
                if isinstance(child, _BaseBayesianNet):
                    domain = getattr(child, "_domain", "unknown")
                    if domain in alpha_counters:
                        child.alpha_index = alpha_counters[domain]
                        hook_fn = self._alpha_hook(domain)
                        child.register_forward_pre_hook(hook_fn)
                        alpha_counters[domain] += 1
        self._alpha_counters = alpha_counters

    def _alpha_hook(self, domain):
        def hook(module, input):
            alpha = self._alphas[domain][..., module.alpha_index]
            return input[0], alpha.unsqueeze(-1)

        return hook

    def _domain_hook(self, num_nodes, num_edges):
        def hook(module, input):
            size = input[0].size(0)
            module._domain = (
                "node"
                if size == num_nodes
                else "edge" if size == num_edges else "unknown"
            )

        return hook

    @property
    def model(self):
        return self._nn["model"]

    @property
    def posterior(self):
        return self._nn["posterior"]

    @property
    def num_bayesian_layers(self):
        return sum(isinstance(m, _BaseBayesianNet) for m in self.model.modules())

    @property
    def num_alphas_edge(self):
        return self._alpha_counters["edge"]

    @property
    def num_alphas_node(self):
        return self._alpha_counters["node"]
