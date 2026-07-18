import torch
import torch.nn as nn


def project_l1_ball(x, eps):
    original_shape = x.shape
    x = x.view(x.shape[0], -1)

    projected = []
    for i in range(x.shape[0]):
        v = x[i]
        u = torch.abs(v)
        if u.sum() <= eps:
            projected.append(v)
            continue

        u_sorted, _ = torch.sort(u, descending=True)
        cumsum = torch.cumsum(u_sorted, dim=0)
        k = torch.arange(1, len(u_sorted) + 1, device=x.device, dtype=torch.float32)

        condition = u_sorted > (cumsum - eps) / k
        rho = len(condition) - torch.flip(condition, [0]).long().argmax().item()
        theta = (cumsum[rho - 1] - eps) / rho if rho > 0 else 0
        projected.append(torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0))

    return torch.stack(projected).view(original_shape)


class PGDAttack:

    def __init__(self, model, eps, alpha=None, num_iter=None, norm="linf",
                 randomize=True, normalize_fn=None):
        self.model = model
        self.eps = eps
        self.norm = norm
        self.randomize = randomize
        self.normalize = normalize_fn or (lambda x: x)
        self.alpha = alpha if alpha is not None else (eps / 4 if norm == "linf" else eps / 10)
        self.num_iter = num_iter if num_iter is not None else 40

    def project_perturbation(self, delta, norm):
        if norm == "linf":
            return torch.clamp(delta, -self.eps, self.eps)
        if norm == "l2":
            delta_flat = delta.view(delta.size(0), -1)
            norm_delta = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            scale = torch.clamp(norm_delta / self.eps, min=1.0)
            return (delta_flat / scale).view_as(delta)
        if norm == "l1":
            return project_l1_ball(delta, self.eps)
        raise ValueError(f"Unknown norm: {norm}")

    def perturb(self, x, y, x_min=0.0, x_max=1.0):
        if self.randomize:
            if self.norm == "linf":
                delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
            elif self.norm == "l2":
                delta = torch.randn_like(x)
                delta_flat = delta.view(delta.size(0), -1)
                norm_delta = torch.norm(delta_flat, p=2, dim=1, keepdim=True) + 1e-10
                delta = delta / norm_delta.view(delta.size(0), 1, 1, 1)
                delta = delta * (torch.rand(x.size(0), 1, 1, 1).to(x.device) * self.eps)
            elif self.norm == "l1":
                delta = torch.zeros_like(x)
                mask = torch.rand_like(x) < 0.1
                delta[mask] = torch.empty(mask.sum()).uniform_(-self.eps / 10, self.eps / 10).to(x.device)
                delta = self.project_perturbation(delta, self.norm)
            else:
                raise ValueError(f"Unknown norm: {self.norm}")
        else:
            delta = torch.zeros_like(x)

        x_adv = torch.clamp(x + delta, x_min, x_max)

        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            outputs = self.model(self.normalize(x_adv))
            loss = nn.CrossEntropyLoss()(outputs, y)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                if self.norm == "linf":
                    x_adv = x_adv + self.alpha * grad.sign()
                elif self.norm == "l2":
                    grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
                    x_adv = x_adv + self.alpha * grad / (grad_norm + 1e-10)
                elif self.norm == "l1":
                    x_adv = x_adv + self.alpha * grad.sign()

                delta = self.project_perturbation(x_adv - x, self.norm)
                x_adv = torch.clamp(x + delta, x_min, x_max)

            x_adv = x_adv.detach()

        return (x_adv - x).detach()


def evaluate_accuracy(model, dataloader, device, normalize_fn=None):
    model.eval()
    normalize = normalize_fn or (lambda x: x)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(normalize(inputs))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total
