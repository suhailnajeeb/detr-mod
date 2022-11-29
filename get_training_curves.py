from util.plot_utils import plot_logs
from pathlib import Path
import matplotlib.pyplot as plt

outDir = 'outputs/combined_00'
figDir = 'outputs/combined_00'

# ensure figdir exists
Path(figDir).mkdir(parents=True, exist_ok=True)

log_directory = [Path(outDir)]

fields_of_interest = (
    'loss',
    'mAP',
)

fig, axs = plot_logs(log_directory, fields_of_interest)
# save figure
fig.savefig(Path(figDir) / 'loss_map.png')
plt.show()

fields_of_interest = (
     'loss_ce',
     'loss_bbox',
     'loss_giou',
)

fig, axs = plot_logs(log_directory,
          fields_of_interest)

fig.savefig(Path(figDir) / 'losses.png')
plt.show()

fields_of_interest = (
    'class_error',
    'cardinality_error_unscaled',
    )

fig, axs = plot_logs(log_directory,
          fields_of_interest)

fig.savefig(Path(figDir) / 'class_cardinality_error.png')
plt.show()