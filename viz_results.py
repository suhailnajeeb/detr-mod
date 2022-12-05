import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

log_paths = {
    'model_0': 'outputs/combined_00/log.txt',
#    'model_1': 'outputs/combined_01/log.txt',
#    'model_2': 'outputs/combined_02/log.txt',
#    'model_3': 'outputs/combined_03/log.txt',
#    'model_4': 'outputs/combined_04/log.txt',
    'model_5': 'outputs/combined_05/log.txt',
}

fig, axs = plt.subplots(ncols=2, figsize=(16, 5))

colors = sns.color_palette(n_colors=len(log_paths))

for model, log_path in log_paths.items():
    df = pd.read_json(log_path, lines=True)
    field = 'mAP'
    ewm_col = 0

    if field == 'mAP':
        coco_eval = pd.DataFrame(
            np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
        ).ewm(com=ewm_col).mean()
        #plt.plot(coco_eval, c=colors.pop(), label=model)
        axs[0].plot(coco_eval, c=colors.pop(), label=model)

axs[0].set_title(field)
axs[0].legend()

colors = sns.color_palette(n_colors=len(log_paths))

for model, log_path in log_paths.items():
    df = pd.read_json(log_path, lines=True)
    field = 'loss'
    ewm_col = 0

    df.interpolate().ewm(com=ewm_col).mean().plot(
        y=[f'train_{field}', f'test_{field}'],
        ax=axs[1],
        color=colors.pop(),
        style=['-', '--'],
    )

axs[1].set_title(field)
axs[1].legend()

plt.title('mAP and loss')

plt.show()