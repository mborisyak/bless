"""
The results table of a campaign, everything read from its config (`config/snakemake.yaml`: root, libraries, model
patterns, seeds), the libraries' configs and the protocol (`config/fit.yaml`). Rows are the fitted entries of the
libraries in library order, labelled by their top-level key (by the entry names where a key expands to several
entries); columns are the libraries. A cell is the mean NLL per observation point +- its standard error over the
(seed, system) pairs; a system scores the mean over its finite experiments, a system without one (every restart
diverged) is excluded and counted in the counts table.

    python scripts/results_table.py [--config config/snakemake.yaml] [--out results.txt]
"""
import argparse
import os

import numpy as np
import yaml

import bless
from fit import seed_root

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAMPAIGN = os.path.join(ROOT, 'config', 'snakemake.yaml')
PROTOCOL = os.path.join(ROOT, 'config', 'fit.yaml')
LIBRARIES = os.path.join(ROOT, 'config', 'library')

SPLITS = ('test', 'train')

def load_yaml(path):
  with open(path) as f:
    return yaml.safe_load(f)

def rows_of(library, patterns):
  """`[(label, entry)]` of the fitted entries matching `patterns`, in library order."""
  rows = []
  for key, config in library.items():
    if key in (bless.utils.config.NAME, bless.utils.config.GROUND_TRUTH):
      continue
    entries = bless.utils.config.models(bless.utils.config.expand_library({key: config}), patterns)
    rows.extend((key if len(entries) == 1 else entry, entry) for entry in entries)
  return rows

def model_name(entry):
  """The registry name of an entry's model."""
  name, = entry['model']
  return name

def base_name(entry):
  """The registry name of the mechanistic model an entry's model extends (the model itself if not a hybrid)."""
  model = bless.mm.__models__[model_name(entry)]
  base = getattr(model, 'Base', model)
  return next(name for name, m in bless.mm.__models__.items() if m is base)

def solver_name(entry):
  name, = entry['solver']
  arguments = ', '.join(f'{k}={v}' for k, v in entry['solver'][name].items())
  return f'{name}({arguments})'

def unique(xs):
  return ', '.join(dict.fromkeys(xs))

def per_system(root, library, model, seeds, split):
  """`({(seed, system): score}, n_diverged)` over the fits that exist; a system with no finite experiment is diverged."""
  out, diverged = {}, 0
  for seed in seeds:
    path = os.path.join(seed_root(root, seed), library, model, 'mle.npz')
    if not os.path.exists(path):
      continue
    with np.load(path) as data:
      for system, row in enumerate(data[f'metadata.{split}']):
        if not np.isfinite(row).any():
          diverged += 1
          continue
        out[(seed, system)] = float(np.nanmean(row[np.isfinite(row)]))
  return out, diverged

def cell(root, library, entry, seeds, split):
  """`(text, counted, diverged)` of one (library, entry) on `split`."""
  scores, diverged = per_system(root, library, entry, seeds, split)
  x = np.array(list(scores.values()))
  if x.size == 0:
    return 'n/a', 0, diverged
  se = x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else float('nan')
  return f'{x.mean():.3f} +- {se:.3f}', x.size, diverged

def render(header, rows, numeric=True):
  """Text table, two spaces between columns: the first column left-aligned, the others right-aligned when `numeric`."""
  widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
  just = str.rjust if numeric else str.ljust
  def line(r):
    cells = [str(r[0]).ljust(widths[0])] + [just(str(v), w) for v, w in zip(r[1:], widths[1:])]
    return '  '.join(cells).rstrip()
  return [line(header), '  '.join('-' * w for w in widths)] + [line(r) for r in rows]

def flow(mapping):
  return yaml.safe_dump(mapping, default_flow_style=True, sort_keys=False, width=1 << 16).strip()

def table(campaign, protocol):
  root, seeds, patterns = campaign['root'], campaign['seeds'], campaign['models']
  configs = {name: load_yaml(os.path.join(LIBRARIES, f'{name}.yaml')) for name in campaign['libraries']}
  libraries = {name: bless.utils.config.expand_library(config) for name, config in configs.items()}
  rows = {name: rows_of(config, patterns) for name, config in configs.items()}
  labels = list(dict.fromkeys(label for name in libraries for label, _ in rows[name]))

  n_systems = protocol['sampling']['samples']
  lines = [
    f'{root}: NLL per observation point, mean +- standard error over the (seed, system) pairs',
    f'of {len(seeds)} seeds x {n_systems} systems.',
    f'sampling: {flow(protocol["sampling"]["experiments"])}',
    f'fitting: {flow(protocol["fitting"])}',
    '',
  ]

  legend = []
  for name, library in libraries.items():
    truth = library[bless.utils.config.GROUND_TRUTH]
    fitted = [library[entry] for _, entry in rows[name]]
    legend.append([
      name, model_name(truth), solver_name(truth),
      unique(base_name(e) for e in fitted), unique(solver_name(e) for e in fitted),
    ])
  header = ['library', 'ground truth', 'its solver', 'fitted bases', 'fitted solvers']
  lines.extend(render(header, legend, numeric=False))
  lines.append('')

  columns = ['model', *libraries]
  for split in SPLITS:
    scores, counts = [], []
    for label in labels:
      score_row, count_row = [label], [label]
      for name in libraries:
        entries = dict(rows[name])
        if label not in entries:
          score_row.append('n/a')
          count_row.append('n/a')
          continue
        text, counted, diverged = cell(root, name, entries[label], seeds, split)
        score_row.append(text)
        count_row.append(f'{counted}' + (f' ({diverged} diverged)' if diverged else ''))
      scores.append(score_row)
      counts.append(count_row)

    lines.append(f'{split} NLL')
    lines.extend(render(columns, scores))
    lines.append('')
    lines.append(f'{split}: systems counted of {len(seeds) * n_systems}')
    lines.extend(render(columns, counts))
    lines.append('')

  return '\n'.join(lines)

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--config', default=CAMPAIGN, help='the campaign config')
  parser.add_argument('--out', default=None, help='also write the table here')
  args = parser.parse_args()

  text = table(load_yaml(args.config), load_yaml(PROTOCOL))
  print(text)
  if args.out is not None:
    with open(args.out, 'w') as f:
      f.write(text + '\n')

if __name__ == '__main__':
  main()
