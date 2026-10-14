import yaml

import bless

def test_expand():
  config = yaml.safe_load('''
  __name__: lib1
  ground-truth:
    model: 1
    args: 2
  modelll:
    __template__:
      model: 2
      args: $value1
      argss: $value2
    __values__:
      value1: {a: 1, b: 2}
      value2: [10, 20]
    __postfix__: '{value1}-{value2}-{value1}'
  ''')

  expanded = bless.utils.config.expand_library(config)

  assert expanded['__name__'] == 'lib1'
  assert expanded['ground-truth'] == {'model': 1, 'args': 2}
  assert set(expanded) == {'__name__', 'ground-truth', 'modelll-a-0-a', 'modelll-b-0-b', 'modelll-a-1-a', 'modelll-b-1-b'}
  assert expanded['modelll-b-1-b'] == {'model': 2, 'args': 2, 'argss': 20}

def test_expand_library():
  with open('config/library/regulatory.yaml') as f:
    library = bless.utils.config.expand_library(yaml.safe_load(f))

  assert library['__name__'] == 'regulatory'
  assert library['bless-4w12-reg3']['model']['blessed-monod']['latent'] == 4
  assert library['bless-4w12-reg3']['model']['blessed-monod']['units'] == [12]
  assert library['bless-4w12-reg3']['regularization'] == 1.0e-3
  assert library['sequential-77-reg3']['model']['sequential-monod']['units'] == [77]
