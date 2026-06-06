# train.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse, os
import torch
import numpy as np
from tqdm import tqdm

from opacus.accountants import GaussianAccountant

from utils.data_utils import load_dataset
from utils.dp_utils import calibrateAnalyticGaussianMechanism
from models.model import M3SVM, regularized_loss

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        dp_method=None,
        dp_eps=None,
        dp_delta=None,
        dp_clip=None
    ):
        self.model     = model.to(device)
        self.opt       = optimizer
        self.loss_fn   = loss_fn
        self.device    = device
        self.dp_method = dp_method
        self.dp_eps    = dp_eps
        self.dp_delta  = dp_delta
        self.dp_clip   = dp_clip

        if self.dp_method == 'G':
            self.accountant = GaussianAccountant()

    def fit(
        self,
        train_loader,
        test_loader,
        epochs,
        p,
        lam,
        log_interval=5
    ):
        n_samples = len(train_loader.dataset)
        step_count = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for x, y in train_loader:
                step_count += 1
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss   = self.loss_fn(self.model, n_samples, logits, y, p, lam)
                self.opt.zero_grad()
                loss.backward()

                if self.dp_method == 'G':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.dp_clip)
                    sigma = calibrateAnalyticGaussianMechanism(
                        self.dp_eps, self.dp_delta, self.dp_clip
                    )
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad += torch.randn_like(param.grad) * sigma

                    noise_multiplier = 1.0 / sigma
                    sample_rate      = x.size(0) / n_samples
                    self.accountant.step(
                        noise_multiplier=noise_multiplier,
                        sample_rate=sample_rate
                    )

                    eps_spent = self.accountant.get_epsilon(
                        delta=self.dp_delta,
                        poisson=False
                    )
                    print(f" [DP step {step_count}] ε_spent={eps_spent:.4f}, δ={self.dp_delta}")

                self.opt.step()
                total_loss += loss.item() * x.size(0)

            if epoch % log_interval == 0 or epoch == epochs:
                avg_loss = total_loss / n_samples
                acc      = self.evaluate(test_loader)
                print(f"[Epoch {epoch}/{epochs}] loss={avg_loss:.4f}, test_acc={acc:.4f}")

        if self.dp_method == 'G':
            final_eps = self.accountant.get_epsilon(
                delta=self.dp_delta,
                poisson=False
            )
            print(f"=== Total DP budget used: (ε={final_eps:.4f}, δ={self.dp_delta}) ===")

    def fit_fullbatch(
        self,
        train_loader,
        val_loader,
        epochs,
        p,
        lam,
        log_interval=5
    ):
        X_train, y_train = next(iter(train_loader))
        X_val,   y_val   = next(iter(val_loader))
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_val,   y_val   = X_val.to(self.device),   y_val.to(self.device)
        n = X_train.size(0)

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.opt.zero_grad()

            logits = self.model(X_train)
            loss   = self.loss_fn(self.model, n, logits, y_train, p, lam)
            loss.backward()

            if self.dp_method == 'G':
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.dp_clip)
                # sigma = calibrateAnalyticGaussianMechanism(
                    # self.dp_eps, self.dp_delta, self.dp_clip
                # )
                sigma = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad += torch.randn_like(param.grad) * sigma

            self.opt.step()

            if epoch % log_interval == 0 or epoch == epochs:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val)
                    preds      = val_logits.argmax(dim=1)
                    labels     = y_val.argmax(dim=1)
                    acc        = (preds == labels).float().mean().item()
                print(f"[FullBatch {epoch}/{epochs}] loss={loss.item():.4f}, val_acc={acc:.4f}")

        if self.dp_method == 'G':
            final_eps = self.accountant.get_epsilon(
                delta=self.dp_delta,
                poisson=False
            )
            print(f"=== Total DP budget used: (ε={final_eps:.4f}, δ={self.dp_delta}) ===")

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for x, y in loader:
                x, y   = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                preds  = logits.argmax(dim=1)
                labels = y.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += x.size(0)
        return correct / total


if __name__=="__main__":
    # working dir to script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1) parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str,   default='HHAR')
    parser.add_argument('--test_size',  type=float, default=0.2)
    parser.add_argument('--scale',      action='store_true')
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--lr',         type=float, default=0.01)
    parser.add_argument('--lam',        type=float, default=0.1)
    parser.add_argument('--p',          type=int,   default=2)
    parser.add_argument('--seed',       type=int,   default=42)
    # DP args
    parser.add_argument('--dp_method',  choices=['G','W'], default='G')
    parser.add_argument('--dp_eps',     type=float, default=10)
    parser.add_argument('--dp_delta',   type=float, default=1e-5)
    parser.add_argument('--dp_clip',    type=float, default=1.0)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 2) load data
    train_loader, test_loader, n_samples, in_dim, num_cls = load_dataset(
        f'./dataset/{args.data}.mat',
        test_size=args.test_size,
        scale=args.scale,
        batch_size=args.batch_size,
        random_state=args.seed
    )

    # 3) build model & optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M3SVM(in_dim, num_cls)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) train & eval
    trainer = Trainer(
        model, optimizer, regularized_loss, device,
        dp_method=args.dp_method,
        dp_eps=args.dp_eps,
        dp_delta=args.dp_delta,
        dp_clip=args.dp_clip
    )

    if args.batch_size is None:
        trainer.fit_fullbatch(train_loader, test_loader,
                              epochs=args.epochs,
                              p=args.p, lam=args.lam)
    else:
        trainer.fit(train_loader, test_loader,
                    epochs=args.epochs,
                    p=args.p, lam=args.lam)

    final_acc = trainer.evaluate(test_loader)
    print(f"Final test accuracy: {final_acc:.4f}")
