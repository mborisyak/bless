from typing import Self

__all__ = [
  'Distribution'
]

class Distribution(object):
  @classmethod
  def parse(cls: type[Self], *args, **kwargs) -> Self:
    raise NotImplementedError()
