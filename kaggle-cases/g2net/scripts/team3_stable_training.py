"""Numerically stable GeM and audited BF16/FP32 training for the final campaign.

Adapters only: contestant files and earlier artifacts remain untouched.
"""
import json
import math
import time
from pathlib import Path

import torch


def log_gem(self, value, p=3, eps=1e-6):
    # log(mean(x**p))/p is the same generalized mean, evaluated without
    # underflowing x**p and then differentiating a fractional power at zero.
    with torch.cuda.amp.autocast(enabled=False):
        x = value.float().clamp(min=eps)
        exponent = torch.as_tensor(p, device=x.device, dtype=x.dtype)
        windows = x.log().unfold(-1, self.kernel_size, self.kernel_size)
        return ((torch.logsumexp(windows * exponent, dim=-1)
                 - math.log(self.kernel_size)) / exponent).exp()


def patch_gem(module):
    module.GeM.gem = log_gem


def optimizer_moment_source(path, parameter_names):
    """Recover a skipped-update run's nearest unchanged-weight ancestor.

An empty optimizer from a zero-update run contains no moments to resume. Its
parent can supply them only when every trainable parameter is exactly equal.
BN buffers may have changed during the unsuccessful run and are not replaced.
"""
    first = torch.load(path, map_location='cpu')
    reference = first['model_state_dict']
    candidate, visited = first, []
    for _ in range(10):
        if str(path) in visited: raise ValueError('Checkpoint initialization cycle')
        visited.append(str(path))
        if any(not torch.equal(reference[n], candidate['model_state_dict'][n]) for n in parameter_names):
            raise ValueError('Ancestor optimizer moments do not match current trainable weights')
        state = candidate.get('optimizer_state_dict', {})
        if state.get('state'):
            return state, dict(source=str(path), checked_lineage=visited,
                               trainable_weights_exactly_equal=True,
                               scope='Moments/step only; current BN buffers retained; new LR schedule')
        launch = json.loads((path.parent/'launch.json').read_text())
        parent = launch.get('checkpoint_dir')
        if not parent: break
        path = Path(parent)/('Fold_%d_%s_model.pth' % (launch['fold'], launch.get('checkpoint_selection','best')))
        candidate = torch.load(path, map_location='cpu')
    raise ValueError('No nonempty optimizer state on the identical-weight initialization lineage')


def install_training(train_helper, args):
    original_init = train_helper.Trainer.__init__

    def init(self, *pos, **kw):
        original_init(self, *pos, **kw)
        if self.gradient_accumulation_steps != 1 or self.use_mixup or self.swa_model is not None:
            raise ValueError('Audited final trainer supports non-mixup, single-step, non-SWA configs')
        self.actual_optimizer_updates = 0
        self.audit_batches = 0
        self.audit_started = time.time()
        self.optimizer_resume_provenance = None
        if args.resume_optimizer_moments:
            path = args.checkpoint_dir / ('Fold_%d_%s_model.pth' % (args.fold, args.checkpoint_selection))
            names = [n for n,p in self.model.named_parameters() if p.requires_grad]
            state, self.optimizer_resume_provenance = optimizer_moment_source(path, names)
            settings = [{k:v for k,v in g.items() if k != 'params'} for g in self.optimizer.param_groups]
            self.optimizer.load_state_dict(state)
            for group, saved in zip(self.optimizer.param_groups, settings):
                group.update(saved)
            # Intentionally retain moments/step only; use this trial's LR and
            # fresh scheduler. Old artifacts do not contain exact RNG state.

    def audit(self, **extra):
        value = dict(actual_optimizer_updates=self.actual_optimizer_updates,
                     batches=self.audit_batches, completed_epochs=getattr(self, 'completed_local_epochs', 0)+(1 if extra.get('state')=='epoch_finished' else 0),
                     elapsed_seconds=time.time()-self.audit_started,
                     skipped_optimizer_updates=0, finite_gradient_checks='Every backward pass; fail immediately',
                     loss_scaler_enabled=False, sam_rho=args.sam_rho,
                     resume_optimizer_moments=args.resume_optimizer_moments,
                     optimizer_resume_provenance=self.optimizer_resume_provenance, **extra)
        path = Path(self.audit_output) / 'optimizer_audit.json'
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)

    def train_epoch(self, loader):
        self.model.train()
        if args.freeze_bn:
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        params = [p for p in self.model.parameters() if p.requires_grad]
        total = torch.zeros((), device=self.device)
        start_updates = self.actual_optimizer_updates
        for step, batch in enumerate(loader, 1):
            if args.stop_at is not None and time.time() >= args.stop_at:
                # Finish a partial epoch with normal validation/checkpointing.
                if step > 1:
                    break
            self.optimizer.zero_grad(set_to_none=True)
            x, target = batch[0].to(self.device), batch[1].to(self.device)
            if args.gain_jitter:
                x = x * torch.exp(torch.randn(len(x), 3, 1, device=x.device) * args.gain_jitter)
            cpu_rng = torch.get_rng_state() if args.sam_rho else None
            cuda_rng = torch.cuda.get_rng_state(self.device) if args.sam_rho else None

            def backward():
                with train_helper.autocast(enabled=self.use_autocast):
                    output = self.model(x).reshape(-1)
                    if not torch.isfinite(output).all():
                        raise FloatingPointError('Nonfinite training output')
                    loss = self.criterion(output, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=True)
                return loss, norm

            try:
                loss, norm = backward()
                if args.sam_rho:
                    # Same stochastic masks and batch statistics at both SAM
                    # points; restore running BN buffers after the second pass.
                    saved = []
                    with torch.no_grad():
                        for p in params:
                            if p.grad is not None:
                                saved.append((p, p.detach().clone()))
                                p.add_(p.grad, alpha=args.sam_rho/(float(norm)+1e-12))
                    bn_buffers = [(b, b.clone()) for n,b in self.model.named_buffers()
                                  if 'running_' in n or 'num_batches_tracked' in n]
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng, self.device)
                    try:
                        _, norm = backward()
                    finally:
                        with torch.no_grad():
                            for p, old in saved: p.copy_(old)
                            for b, old in bn_buffers: b.copy_(old)
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.gradient_clip, error_if_nonfinite=True)
                self.optimizer.step()
                self.actual_optimizer_updates += 1
                self.audit_batches += 1
                self.step += 1
                self.scheduler.step()
                total += loss.detach()
                if step == 1 or step % 100 == 0:
                    audit(self, last_loss=float(loss), last_gradient_norm=float(norm), state='training')
                    print('AUDIT updates=%d loss=%.6f grad_norm=%.6f' %
                          (self.actual_optimizer_updates, float(loss), float(norm)), flush=True)
            except Exception as error:
                audit(self, state='failed', error=str(error))
                raise
        updates = self.actual_optimizer_updates-start_updates
        if updates <= 0:
            raise FloatingPointError('Epoch made zero optimizer updates')
        audit(self, state='epoch_finished', epoch_updates=updates)
        return total / updates

    def output_init(self, *pos, **kw):
        init(self, *pos, **kw)
        self.audit_output = str(Path(train_helper.__file__).parents[3])  # overwritten below
        config = pos[7] if len(pos) > 7 else kw['Config']
        self.audit_output = config.model_output_folder

    train_helper.Trainer.__init__ = output_init
    train_helper.Trainer.train_epoch = train_epoch
