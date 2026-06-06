import torch

class ScaleScheduler:
    '''
    Scheduler for managing scale parameter resets during training.

    The scale reset has the same rythm as opacity resets, occurring every `n * opacity_reset_interval` epochs.
    '''
    def __init__(self,
                 opacity_reset_interval: int,
                 total_epochs: int,
                 until_epoch: int,
                 iterations_per_epoch: int,
                 scale_reset_factor: float):
        self.opacity_reset_interval = opacity_reset_interval
        self.total_epochs = total_epochs
        self.until_epoch = until_epoch
        self.iterations_per_epoch = iterations_per_epoch
        self.scale_reset_factor = scale_reset_factor

        # Calculate the last epoch where reset occurs (key for all period calculations)
        # This is the largest multiple of opacity_reset_interval that is < until_epoch
        self.last_reset_epoch = ((self.until_epoch - 1) // self.opacity_reset_interval) * self.opacity_reset_interval

        # Track current epoch
        self.current_epoch = 0

        # Print period structure for debugging
        self._print_period_structure()
    
    def _print_period_structure(self) -> None:
        print("=" * 80)
        print("Period Structure based on Opacity Reset:")
        print(f"  opacity_reset_interval: {self.opacity_reset_interval:>5}")
        print(f"  total_epochs:         {self.total_epochs:>5}")
        print(f"  until_epoch:          {self.until_epoch:>5}")
        print(f"  iterations_per_epoch: {self.iterations_per_epoch:>5}")
        print(f"  total_iters:          {self.total_epochs*self.iterations_per_epoch:>5}")
        print()

        print(f"  Last reset epoch: {self.last_reset_epoch}")
        print(f"  Reset epochs: ", end="")

        # List all reset epochs
        reset_epochs = []
        epoch = self.opacity_reset_interval
        while epoch < self.until_epoch:
            reset_epochs.append(epoch)
            epoch += self.opacity_reset_interval
        print(", ".join(map(str, reset_epochs)))
        print()
        
        # Print period structure
        print("  Periods:")
        current_epoch = 0
        period_num = 0

        while current_epoch < self.total_epochs:
            if period_num == 0:
                # Period 0 is special: includes epoch 0 and goes to opacity_reset_interval
                training_start = 0
                training_end = self.opacity_reset_interval
            elif current_epoch < self.last_reset_epoch:
                # Normal periods
                training_start = current_epoch + 1
                training_end = current_epoch + self.opacity_reset_interval
            else:
                # Final period
                training_start = current_epoch + 1
                training_end = self.total_epochs - 1

            training_length = training_end - training_start + 1
            iteration_count = training_length * self.iterations_per_epoch
            start_iter = training_start * self.iterations_per_epoch
            print(f"    Period {period_num:2d}: Serves training epochs {training_start:3d} - {training_end:3d} ({training_length:3d} epochs, {iteration_count:5d} iters, starts from iter {start_iter:5d})")

            # Move to next period
            if current_epoch < self.last_reset_epoch:
                current_epoch += self.opacity_reset_interval
            else:
                break  # Final period covers rest
            period_num += 1

        print("=" * 80)
        
    def calculate_period_info(self, epoch: int) -> tuple[int, int, int, float, bool]:
        # Handle Period 0 specially (epochs 0-10, has 11 epochs)
        if epoch <= self.opacity_reset_interval:
            period_number = 0
            epoch_in_period = epoch
            period_length = self.opacity_reset_interval + 1  # Period 0 has 11 epochs (0-10)
            is_final_period = False
        # Check if we're in the final period (starting from last reset)
        elif epoch > self.last_reset_epoch:
            # Final period: from last_reset_epoch+1 to total_epochs-1
            period_number = self.last_reset_epoch // self.opacity_reset_interval
            epoch_in_period = epoch - (self.last_reset_epoch + 1)
            period_length = self.total_epochs - 1 - self.last_reset_epoch
            is_final_period = True
        else:
            # Normal periods (1, 2, 3, etc.): each has exactly opacity_reset_interval epochs
            # Adjust epoch to account for Period 0 having 11 epochs instead of 10
            adjusted_epoch = epoch - (self.opacity_reset_interval + 1)
            period_number = (adjusted_epoch // self.opacity_reset_interval) + 1
            epoch_in_period = adjusted_epoch % self.opacity_reset_interval
            period_length = self.opacity_reset_interval
            is_final_period = False

        assert period_length >= self.opacity_reset_interval, \
            f"period_length ({period_length}) < opacity_reset_interval ({self.opacity_reset_interval})"

        # Calculate progress ratio: should be 1.0 at the final epoch of the period
        progress_ratio = epoch_in_period / (period_length - 1)
        is_final_reset = (epoch == self.last_reset_epoch)
        # print(f"Period info at epoch {epoch}: period_number={period_number}, epoch_in_period={epoch_in_period}, period_length={period_length}, progress_ratio={progress_ratio:.3f}, is_final_period={is_final_period}, is_final_reset={is_final_reset}")
        return period_number, epoch_in_period, period_length, progress_ratio, is_final_period, epoch, is_final_reset

    def _get_reset_factor_default(self, period_info: tuple) -> float:
        """Default reset factor logic"""
        period_number, epoch_in_period, period_length, progress_ratio, is_final_period, _, is_final_reset = period_info

        if is_final_period:
            raise ValueError("No scale reset in final period")

        # Start with base reset factor
        reset_factor = self.scale_reset_factor

        if period_number <= 2:
            reset_factor = 0.0
        if 3 <= period_number <= 4:
            reset_factor += 0.1
        if 5 <= period_number:
            reset_factor += 0.0

        return reset_factor

    def _get_reset_factor_with_entropy(self, period_info: tuple) -> float:
        """Reset factor logic when entropy strategy is enabled"""
        period_number, epoch_in_period, period_length, progress_ratio, is_final_period, _, is_final_reset = period_info

        if is_final_period:
            raise ValueError("No scale reset in final period")

        # Start with base reset factor
        reset_factor = self.scale_reset_factor

        if period_number <= 2:
            reset_factor = 0.0
        if 3 <= period_number <= 4:
            reset_factor += 0.1
        if 5 <= period_number:
            reset_factor += 0.0

        return reset_factor

    @torch.no_grad()
    def step(self, optimizer: torch.optim.Optimizer, epoch: int, lambda_entropy: float) -> None:
        """Perform scale reset if needed at the current epoch

        Args:
            optimizer: The optimizer to reset
            epoch: Current training epoch
            lambda_entropy: Entropy loss weight. If > 0.0, uses entropy-specific reset logic
        """
        if self.scale_reset_factor <= 0.0:
            return

        # Check if we should reset at this epoch
        if epoch >= self.until_epoch:
            return

        scale_reset_interval = 2 * self.opacity_reset_interval
        '''
        Here we skip the last scale reset for quality reasons. For small model, we will need it for speedup.
        '''
        if epoch % scale_reset_interval != 0 or epoch == 0 or (epoch > self.last_reset_epoch - scale_reset_interval):
            return

        # Calculate period info
        period_info = self.calculate_period_info(epoch)

        # Get reset factor using appropriate logic based on entropy strategy
        if lambda_entropy > 0.0:
            reset_factor = self._get_reset_factor_with_entropy(period_info)
        else:
            reset_factor = self._get_reset_factor_default(period_info)

        if reset_factor <= 0.0:
            return

        # Clear optimizer state
        optimizer.state.clear()

        # Get scale parameter from optimizer
        scale = None
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'scale':
                scale = param_group['params'][0]
                break

        # Apply scale reduction
        actived_scale = scale.exp()
        scale.data = (actived_scale * reset_factor).log()

        strategy_str = "with_entropy" if lambda_entropy > 0.0 else "default"
        print(f"ScaleScheduler: Epoch {epoch}: Applied scale reset with factor {reset_factor:.3f} (strategy: {strategy_str})")
