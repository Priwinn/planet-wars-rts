import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution, Categorical
from torch.distributions.transforms import SigmoidTransform
import math

# Adapted from SEED RL
class SigmoidTransformedDistribution(TransformedDistribution):
    """Normal distribution followed by sigmoid transformation."""

    def __init__(self, loc, scale, threshold=0.99, validate_args=False):
        """Initialize the distribution.

        Args:
            loc: Mean of the underlying normal distribution.
            scale: Standard deviation of the underlying normal distribution.
            threshold: Clipping value of the action when computing the logprob.
            validate_args: Passed to super class.
        """
        base_distribution = Normal(loc, scale, validate_args=validate_args)
        transforms = [SigmoidTransform()]
        
        super().__init__(
            base_distribution=base_distribution,
            transforms=transforms,
            validate_args=validate_args
        )
        
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -logit(threshold)] for
        # log_prob_left and [logit(threshold), inf] for log_prob_right.
        self._threshold = threshold
        
        # For sigmoid, the inverse is logit
        inverse_threshold = torch.logit(torch.tensor(threshold))
        
        # Let epsilon = 1 - threshold
        # average(pdf) on [threshold, 1] = probability([threshold, 1])/epsilon
        # So log(average(pdf)) = log(probability) - log(epsilon)
        log_epsilon = torch.log(torch.tensor(1.0 - threshold))
        
        # These values are differentiable w.r.t. model parameters
        self._log_prob_left = torch.log(torch.clamp(self.base_dist.cdf(-inverse_threshold), min=1e-6)) - log_epsilon
        self._log_prob_right = torch.log(torch.clamp(1-self.base_dist.cdf(inverse_threshold), min=1e-6)) - log_epsilon


    def log_prob(self, value):
        """Compute log probability with clipping for numerical stability."""
        # Clip the value to avoid NaNs
        value = torch.clamp(value, min=1.0 - self._threshold, max=self._threshold)
        
        # We don't need to handle jacobian because we use ratios in PPO (see SEED RL paper)
        result = super().log_prob(value)
        result = torch.where(value <= (1.0 - self._threshold), self._log_prob_left, result)
        result = torch.where(value >= self._threshold, self._log_prob_right, result)
        return result

    def entropy(self):
        """Entropy estimation using a single sample."""
        # Sample from base distribution
        # print(f"Base distribution stddev: {self.base_dist.stddev}")
        sample = self.base_dist.rsample()
        # Compute log det jacobian at the sample point
        sigmoid_sample = torch.sigmoid(sample)

        log_det_jacobian = torch.log(torch.clamp(sigmoid_sample * (1 - sigmoid_sample), min=1e-6))


        # Entropy approximation
        return self.base_dist.entropy() + log_det_jacobian


class MaskedCategorical(Categorical):
    """Categorical distribution with action masking"""
    
    def __init__(self, logits=None, probs=None, mask=None):
        if mask is not None:
            # Set logits of invalid actions to very negative values
            if logits is not None:
                logits = torch.where(mask.bool(), logits, torch.tensor(torch.finfo(torch.float32).min, device=logits.device))
            elif probs is not None:
                probs = torch.where(mask.bool(), probs, torch.tensor(1e-8, device=probs.device))
        
        super().__init__(logits=logits, probs=probs)
        self.mask = mask
    
    def entropy(self):
        # Only calculate entropy for valid actions
        if self.mask is not None:
            # Get probabilities and zero out invalid actions
            p_log_p = self.logits * self.probs
            p_log_p = torch.where(self.mask.bool(), p_log_p, torch.tensor(0.0, device=p_log_p.device))
            return -p_log_p.sum(-1)
        return super().entropy()