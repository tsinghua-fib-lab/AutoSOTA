import os
from dotenv import load_dotenv

load_dotenv()

from src.utils import find_keyword_positions, preprocess_text, tokenize_string
import numpy as np
from transformers import AutoTokenizer
from scipy import stats
from typing import List, Optional

class StoppingRule:
    """
    Abstract class for stopping rules.
    The stopping rule monitors the decoded tokens so far and decides when to stop decoding.
    """
    def __init__(self, model_name, alpha = 0.05, **kwargs):
        """
        Initialize the stopping rule with a lazy interval.
        """
        self.tokens_so_far = []
        self.alpha = alpha
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def calibrate(self, reasoning_traces: List[str]):
        """
        Train the stopping rule based on the provided tokenized reasoning traces.
        :param tokenized_reasoning_traces: A list of tokenized reasoning traces to calibrate the stopping rule.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def test(self, reasoning_traces: List[str]):
        raise NotImplementedError("Subclasses should implement this method.")


class DecodingSimulator:
    def __init__(self, tokenizer: AutoTokenizer, reasoning_trace: str):
        self.reasoning_trace = reasoning_trace
        self.tokenizer = tokenizer
        self.encoded_trace = tokenize_string(reasoning_trace, tokenizer)
        self.current_position = 0
    
    def next_token_position(self, n: int = 1):
        if self.current_position + n < len(self.encoded_trace):
            self.current_position += n
        else:
            self.current_position = len(self.encoded_trace)
            return None
        
        text = self.tokenizer.decode(self.encoded_trace[:self.current_position], skip_special_tokens=True)
        clean_text = preprocess_text(text).strip()
        token_position = len(clean_text.split()) - 1
        return token_position
        
    def calculate_savings(self):
        """
        Calculate the savings based on the current position in the reasoning trace.
        Return two values:
        - The number of tokens saved.
        - The percentage of tokens saved compared to the total length of the reasoning trace.
        """
        number_of_tokens = len(self.encoded_trace)
        tokens_saved = number_of_tokens - self.current_position 
        percentage_saved = (tokens_saved / number_of_tokens) * 100 if number_of_tokens > 0 else 0
        return tokens_saved, percentage_saved


class LengthStoppingRule(StoppingRule):
    """
    A simple stopping rule that stops decoding after a certain number of tokens.
    """
    def __init__(self, model_name, alpha: float = 0.05):
        super().__init__(model_name, alpha)
        # set this to integer inf
        self.length_threshold = int(1e9)  # effectively infinity for practical purposes
        self.cal_length_quantiles = None
        self.test_length_quantiles = []

    def calibrate(self, reasoning_traces):
        lengths = [len(tokenize_string(trace, self.tokenizer)) for trace in reasoning_traces]
        n = len(lengths)
        self.length_threshold = int(np.quantile(lengths, (1 - self.alpha) * (n+1) / n))
        self.cal_length_quantiles = tuple(np.quantile(lengths, [0, 0.25, 0.5, 0.75, 1]))

    def test(self, reasoning_traces) -> bool:
        lengths = [len(tokenize_string(trace, self.tokenizer)) for trace in reasoning_traces]
        tokens_saved_list = [max(0, length - self.length_threshold) for length in lengths]
        percentage_saved_list = [saved / length * 100 if length > 0 else 0 for saved, length in zip(tokens_saved_list, lengths)]
        self.test_length_quantiles.append(tuple(np.quantile(lengths, [0, 0.25, 0.5, 0.75, 1])))
        return tokens_saved_list, percentage_saved_list

class UncertaintyStoppingRule(StoppingRule):
    """
    A stopping rule based on the uncertainty of the model's predictions.
    This is a placeholder for a more complex implementation.
    """
    def __init__(self, model_name, lazy_interval: int = 100, alpha: float = 0.05, uncertainty_keywords: list = None, **kwargs):
        super().__init__(model_name, alpha)
        self.uncertainty_threshold = 1
        self.uncertainty_keywords = uncertainty_keywords if uncertainty_keywords is not None else []
        self.lazy_interval = lazy_interval
    
    def _should_stop(self, X_t, t) -> bool:
        raise NotImplementedError("Subclasses should implement this method.")

    def calibrate(self, reasoning_traces):
        lengths = [len(tokenize_string(trace, self.tokenizer)) for trace in reasoning_traces]
        self.lazy_interval = min(int(np.median(lengths) / 3), self.lazy_interval)

    def test(self, reasoning_traces) -> bool:
        tokens_saved_list = []
        percentage_saved_list = []
        for trace in reasoning_traces:
            tokens_saved, percentage_saved = self._test_a_single_trace(trace)
            tokens_saved_list.append(tokens_saved)
            percentage_saved_list.append(percentage_saved)

        return tokens_saved_list, percentage_saved_list
    
    def _test_a_single_trace(self, trace) -> bool:
        arrivals, _ = find_keyword_positions(trace, self.uncertainty_keywords)
        simulator = DecodingSimulator(self.tokenizer, trace)
        X_t = 0
        t = 0
        while True:
            next_pos = simulator.next_token_position(self.lazy_interval)
            if next_pos is None:
                break
            while X_t < len(arrivals) and arrivals[X_t] <= next_pos:
                X_t += 1
            t = next_pos
            if X_t >= 0 and t > 0:
                if self._should_stop(X_t, t):
                    break
            
        return simulator.calculate_savings()

class UncertaintyArrivalStoppingRule(UncertaintyStoppingRule):
    """
    A stopping rule based on the renewal process theory for uncertainty keywords.
    It uses the Central Limit Theorem for renewal processes to decide when to stop decoding.
    """
    def __init__(self, model_name, lazy_interval: int = 100, alpha: float = 0.05, uncertainty_keywords: list = None, alpha_spending="Sidak"):
        super().__init__(model_name, lazy_interval, alpha, uncertainty_keywords)
        self.mu_hat = None
        self.sigma_hat_sq = None
        self.alpha_spending = alpha_spending
        self.max_num_test = None

    def _adjust_critical_value(self, alpha: float, alpha_spending = "Sidak", max_num_test: int = 320) -> float:
        """
        Adjust alpha and critical value based on the spending function for multiple comparison correction.
        """
        if alpha_spending is None:
            return stats.norm.ppf(1 - alpha)
        elif alpha_spending == 'Bonferroni':
            return stats.norm.ppf(1 - alpha / max_num_test)
        elif alpha_spending == 'Sidak':
            return stats.norm.ppf((1 - alpha) ** (1 / max_num_test))
        else:
            raise ValueError(f"Unknown alpha spending function: {alpha_spending}")

    def calibrate(self, reasoning_traces):
        """
        Train the stopping rule by estimating the mean and variance of inter-arrival times.
        :param reasoning_traces: A list of reasoning traces.
        """
        super().calibrate(reasoning_traces)
        all_inter_arrival_times = []

        for trace in reasoning_traces:
            arrivals, max_len = find_keyword_positions(trace, self.uncertainty_keywords)
            
            if len(arrivals) > 1:
                inter_arrivals = np.diff(np.insert(arrivals, 0, 0))
                all_inter_arrival_times.extend(inter_arrivals)
            else:
                all_inter_arrival_times.append(max_len)

        if not all_inter_arrival_times:
            raise ValueError("No uncertainty keywords found in the training data.")
        else:
            self.mu_hat = np.mean(all_inter_arrival_times)
            self.sigma_hat_sq = np.var(all_inter_arrival_times)
        
        tokenized_lengths = [tokenize_string(trace, self.tokenizer) for trace in reasoning_traces]

        self.max_num_test = np.ceil(np.quantile([len(trace) for trace in tokenized_lengths], 0.5) / self.lazy_interval)
        self.critical_value = self._adjust_critical_value(self.alpha, self.alpha_spending, self.max_num_test)            

    def _should_stop(self, X_t, t) -> bool:
        """
        Decide whether to stop decoding based on the hypothesis test.
        :return: True if decoding should stop, False otherwise.
        """

        if X_t == 0:
            return False

        numerator = X_t - (t / self.mu_hat)
        denominator = np.sqrt(t * self.sigma_hat_sq / (self.mu_hat**3))
        Z_t = numerator / denominator
        
        critical_value = self.critical_value
        if t > self.max_num_test * self.lazy_interval:
            max_num_test = np.ceil(t / self.lazy_interval)
            critical_value = self._adjust_critical_value(self.alpha, self.alpha_spending, max_num_test)

        if Z_t > critical_value:
            return True
        return False


class MaxUncertaintyStoppingRule(UncertaintyStoppingRule):
    """
    Group-Conditional Conformal Stopping (Maxwise calibration).

    - During training, for each calibration trace we compute the *maximum* uncertainty
      across all eligible bin prefixes and set a single global threshold tau* as the
      (1 - alpha)-quantile of these maxima.
    - At inference, stop whenever the current uncertainty exceeds tau*.

    This controls the probability of *ever* crossing the boundary across the planned
    sequential looks at approximately alpha (finite-sample via conformal quantile).
    """

    def __init__(self, model_name: str, lazy_interval: int = 100, alpha: float = 0.05, uncertainty_keywords: list = None):
        super().__init__(model_name=model_name, lazy_interval=lazy_interval, alpha=alpha, uncertainty_keywords=uncertainty_keywords)

        # Learned global threshold from maxwise calibration
        self.global_threshold: Optional[float] = None

    
    def calibrate(self, reasoning_traces):
        """
        Maxwise calibration:
          For each calibration trace i, compute M_i = max_j u_i(L_j) over all
          bin prefixes L_j that are <= len(trace_i).
          Set global_threshold = (1 - alpha)-quantile of {M_i}.
        """
        super().calibrate(reasoning_traces)
        
        Ms: List[float] = []

        for trace in reasoning_traces:
            arrivals, _ = find_keyword_positions(trace, self.uncertainty_keywords)
            simulator = DecodingSimulator(self.tokenizer, trace)
            X_t = 0
            t = 0
            mmax = -np.inf
            while True:
                next_pos = simulator.next_token_position(self.lazy_interval)
                if next_pos is None:
                    break
                while X_t < len(arrivals) and arrivals[X_t] <= next_pos:
                    X_t += 1
                t = next_pos
                if X_t >= 0 and t > 0:
                    mmax = max(mmax, X_t / t)
            
            if np.isfinite(mmax):
                Ms.append(mmax)

        assert Ms, "No usable traces reached the first boundary; increase data or reduce lazy_interval."

        n = len(Ms)
        self.global_threshold = float(np.quantile(Ms, min((1.0 - self.alpha) * (n+1) / n, 1.0)))
        if self.global_threshold == 0.0:
            self.global_threshold = max(max(Ms), 1e-6)  # avoid degenerate zero threshold

    def _should_stop(self, X_t, t) -> bool:
        return X_t / t > self.global_threshold
    