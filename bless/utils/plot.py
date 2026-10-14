import numpy as np

import bokeh.plotting as plt
import bokeh.models as mdl
import bokeh.palettes as palettes

__all__ = [
  'get_rows_cols',
  'get_range',
  'losses',
  'plot_distribution'
]


def get_rows_cols(n: int):
  import math

  best_error = n * n
  best_cols = None
  best_rows = None

  for n_cols in range(1, int(math.sqrt(n)) + 1):
    n_rows = n // n_cols + (0 if n % n_cols == 0 else 1)
    error = abs(n_rows * n_cols - n) + 0.1 * abs(n_rows * n_rows - n)
    if error < best_error:
      best_error = error
      best_cols = n_cols
      best_rows = n_rows

  return best_rows, best_cols

def get_range(arr, *arrs, axis=None, q=0.99):
  low = []
  high = []

  arrs = [arr, *arrs]

  for arr in arrs:
    l, h = np.quantile(arr, q=(1 - q, q))
    low.append(l)
    high.append(h)

  low, high = np.min(low), np.max(high)
  delta = high - low
  alpha = max(4 * (1 - q), 0.1)
  low, high = low - alpha * delta, high + alpha * delta

  return low, high

def plot_distribution(fig, xs, ys, n_q=4, color=palettes.Dark2_8[0]):
  qs = np.quantile(ys, axis=0, q=np.linspace(0, 1, num=2 * n_q + 3)[1:-1])

  for i in range(n_q):
    fig.varea(xs, qs[i], qs[-i - 1], fill_alpha=1.0 / n_q, fill_color=color)

  fig.line(xs, qs[n_q], color=color)

  return fig

def plot_loss(loss, fig=None):
  n_q = 4

  if isinstance(loss, (list, tuple)):
    loss = {
      f'{i}': ls
      for i, ls in enumerate(loss)
    }
    no_label = len(loss) == 1

  elif isinstance(loss, dict):
    no_label = False
  else:
    loss = {'_': loss}
    no_label = True

  quantiles = {
    k: np.quantile(ls, axis=1, q=np.linspace(0, 1, num=2 * n_q + 3)[1:-1])
    for k, ls in loss.items()
  }

  if fig is None:
    max_y = max(np.max(quantiles[k]) for k in quantiles)
    fig = plt.figure(sizing_mode="scale_width", aspect_ratio=2, y_range=(0, 1.05 * max_y))

  for i, key in enumerate(loss):
    ls = loss[key]
    ls = np.reshape(ls, shape=(ls.shape[0], -1))

    median = np.median(ls, axis=1)
    mean = np.mean(ls, axis=1)

    mean_error = np.std(ls, axis=1) / np.sqrt(1 + ls.shape[1])
    iters = np.arange(ls.shape[0])

    prefix = '' if no_label else f'{key} '
    for j in range(n_q):
      fig.varea(iters, quantiles[key][j], quantiles[key][-j - 1], fill_alpha=1.0 / n_q, fill_color=palettes.Dark2[8][i])
    fig.line(
      iters, median, line_dash='dashed',
      line_color=palettes.Dark2[8][i]
    )
    fig.line(
      iters, mean,
      legend_label=f'{prefix}mean ({mean[-1]:.3f} +- {mean_error[-1]:.3f})', line_color=palettes.Dark2[8][i]
    )

  return fig

def plot_samples(samples):
  if 'predictions' in samples:
    n_fields = len(samples['predictions'])
    (n_batch, n_experiments), = set([x.shape[:2] for _, x in samples['predictions'].items()])
    fields = tuple(samples['predictions'])

  elif 'observations' in samples:
    n_fields = len(samples['observations'])
    (n_batch, n_experiments), = set([x.shape[:2] for _, x in samples['observations'].items()])
    fields = tuple(samples['observations'])

  else:
    raise ValueError('nothing to plot')

  fig_grid = [
    [plt.figure(sizing_mode="scale_height", aspect_ratio=1) for _ in range(n_batch)]
    for _ in range(n_fields)
  ]

  for j, key in enumerate(fields):
    heap = []
    if 'predictions' in samples:
      heap.append(samples['predictions'][key].ravel())

    if 'observations' in samples:
      heap.append(samples['observations'][key].ravel())

    y_low, y_high = np.quantile(np.concatenate(heap), q=(0.01, 0.99))
    y_delta = y_high - y_low
    y_low, y_high = y_low - 0.1 * y_delta, y_high + 0.1 * y_delta

    for k in range(n_batch):
      fig = fig_grid[j][k]
      for l in range(min(n_experiments, 8)):
        if samples.get('mask', None) is None:
          training_sample = False
        else:
          training_sample = (samples['mask'][k, l] > 0.5)

        if k == 0:
          fig.title.text = key
          fig.title.align = 'center'
        fig.y_range = mdl.Range1d(y_low, y_high)

        if 'observations' in samples:
          ts = samples['observation_ts'][k, l]
          obs = samples['observations'][key][k, l]
          if training_sample:
            fig.scatter(x=ts, y=obs, name=key, line_color=palettes.Dark2[8][l], marker="x")
          else:
            fig.scatter(
              x=ts, y=obs, name=key, line_color=palettes.Dark2[8][l], fill_color=palettes.Dark2[8][l],
              marker="circle"
            )

        if 'predictions' in samples:
          grid = samples['prediction_ts'][k, l]
          pred = samples['predictions'][key][k, l]
          if pred.ndim == 1:
            fig.line(x=grid, y=pred, line_color=palettes.Dark2[8][l], line_dash='dashed' if training_sample else 'solid')
          else:
            n_q = 3
            quantiles = np.linspace(0, 1, num=2 * n_q + 3)[1:-1]
            qs = np.quantile(pred, axis=0, q=quantiles)
            for q_i in range(n_q):
              fig.varea(grid, qs[q_i], qs[-q_i - 1], fill_alpha=1.0 / n_q, fill_color=palettes.Dark2[8][l])

            med = qs[n_q]
            fig.line(x=grid, y=med, line_color=palettes.Dark2[8][l], line_dash='dashed' if training_sample else 'solid')

  return plt.gridplot(fig_grid, sizing_mode='scale_height')

def _plot_losses(fname: str | bytes, loss_infos, samples=None):
  try:
    figures = list()

    for i, loss_info in enumerate(loss_infos):
      fig = plot_loss(loss_info['losses'])
      fig.title.text = loss_info['title']
      fig.title.align = 'center'
      fig.yaxis.axis_label = loss_info['label']
      figures.append(fig)

    loss_figs = plt.row(figures, sizing_mode='scale_height')

    if samples is not None:
      sample_grid = plot_samples(samples)
      layout = plt.column([loss_figs, sample_grid], sizing_mode='scale_height')
    else:
      layout = loss_figs
    plt.output_file(fname)
    plt.save(layout, fname, title='losses')
  except:
    import traceback
    traceback.print_exc()

def losses(fname: str | bytes, loss_infos, samples):
  import threading
  thread = threading.Thread(target=_plot_losses, args=(fname, loss_infos, samples))
  thread.start()
  return thread