import torch
import torch.nn.functional as F

class Bernoulli:

  def __init__(self, dist=None):
    super().__init__()
    self._dist = dist
    self.mean = dist.mean

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def entropy(self):
    return self._dist.entropy()

  def mode(self):
    _mode = torch.round(self._dist.mean)
    return _mode.detach() +self._dist.mean - self._dist.mean.detach()

  def sample(self, sample_shape=()):
    return self._dist.rsample(sample_shape)

  def log_prob(self, x):
    _logits = self._dist.base_dist.logits
    log_probs0 = -F.softplus(_logits)
    log_probs1 = -F.softplus(-_logits)

    return log_probs0 * (1-x) + log_probs1 * x


class CategoricalDist():
  def __init__(self, dist, dim_to_reduce=[-1, -2]):
    super().__init__()
    self._dist = dist
    self._dim_to_reduce = dim_to_reduce

  def log_prob(self, x):
    log_prob = self._dist.log_prob(x)
    # aggregate.
    return log_prob.sum(self._dim_to_reduce)
  
  def mode(self):
    return self._dist.mode

  def entropy(self):
    return self._dist.entropy()

  def __getattr__(self, name):
    return getattr(self._dist, name)

class OneHotDist(torch.distributions.one_hot_categorical.OneHotCategorical):

  def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
    if logits is not None and unimix_ratio > 0.0:
      probs = F.softmax(logits, dim=-1)
      probs = probs * (1.0-unimix_ratio) + unimix_ratio / probs.shape[-1]
      logits = torch.log(probs)
      super().__init__(logits=logits, probs=None)
    else:
      super().__init__(logits=logits, probs=probs)

  def mode(self):
    _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
    return _mode.detach() + super().logits - super().logits.detach()

  def sample(self, sample_shape=(), seed=None):
    if seed is not None:
      raise ValueError('need to check')
    sample = super().sample(sample_shape)
    probs = super().probs
    while len(probs.shape) < len(sample.shape):
      probs = probs[None]
    sample += probs - probs.detach()
    return sample


class ContDist:

  def __init__(self, dist=None):
    super().__init__()
    self._dist = dist
    self.mean = dist.mean

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def entropy(self):
    return self._dist.entropy()

  def mode(self):
    return self._dist.mean

  def sample(self, sample_shape=()):
    return self._dist.rsample(sample_shape)

  def log_prob(self, x):
    return self._dist.log_prob(x)

    
class SafeTruncatedNormal(torch.distributions.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event
