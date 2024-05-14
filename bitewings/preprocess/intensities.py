from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


if __name__ == '__main__':
    roots = [
        Path('/home/mkaailab/.darwin/datasets/mucoaid/bitewingchartfiling_train/images'),
        Path('/home/mkaailab/.darwin/datasets/mucoaid/bitewingchartfiling_test/images'),
        Path('/home/mkaailab/.darwin/datasets/mucoaid/bitewingchartfiling_extra/images')
    ]

    bins = np.zeros((len(roots), 256))
    for i, root in enumerate(roots):
        for img_path in tqdm(list(root.glob('*'))):
            img = cv2.imread(str(img_path))

            counts = np.bincount(img[..., 0].flatten(), minlength=256)
            bins[i] += counts

    clip_bins = bins

    total_count = 1_000_000
    clip_bins = clip_bins / clip_bins.sum(1, keepdims=True)
    repeats = (clip_bins * 1_000_000).astype(int)

    numbers_list = [
        np.repeat(np.arange(0, 256), reps)
        for reps in repeats
    ]

    df = pd.DataFrame({
        'Intensity value': np.concatenate(numbers_list),
        'Source': np.array(['Germany', 'Netherlands', 'Slovakia']).repeat([len(n) for n in numbers_list]),
    })

    sns.kdeplot(data=df, x='Intensity value', hue='Source', bw_adjust=1)
    plt.gca().get_yaxis().set_ticks([])
    plt.savefig('kdes.png', dpi=800, bbox_inches='tight', pad_inches=None)
    plt.show()

