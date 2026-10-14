from .conditional import *
from . import time

from .sampler import Sampler, System

from .parse import *

__std_library__ = {
  'const': const,

  'id': identity,
  'identity': identity,

  'uniform': uniform,
  'U': uniform,

  'normal': normal,
  'gaussian': normal,
  'N': normal,
  'glorot-normal': glorot_normal,

  'exponential': exponential,
  'Exp': exponential,

  'log-normal': log_normal,
  'logN': log_normal,
  'biased-log-normal': biased_log_normal,
  'logit-normal': logit_normal,

  'log-uniform': log_uniform,
  'logU': log_uniform,

  'NPD': normal_positive_definite,
  'NND': normal_negative_definite,
  'NPDB': normal_positive_definite_bound,
  'NNDB': normal_negative_definite_bound,
  'NSS': normal_skew_symmetric,

  'grid': time.grid,
  # 'uniform': time.uniform,
  'grid-uniform': time.grid_uniform,
  # 'uniform-delta': time.uniform_delta,
}