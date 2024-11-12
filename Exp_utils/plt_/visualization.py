import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plt_histogram(series, bins=10, color='#0066cc', col_width=0.9, save_plt_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n, bins, patches = ax.hist(x=series, bins=bins, color=color,
                               alpha=0.7, rwidth=col_width)
    plt.grid(axis='y', alpha=0.75)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 11 if maxfreq % 10 else maxfreq + 10)
    plt.xlabel('Insurance', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    for i in ax.patches:
        if i.get_height() > 9:
            ax.text(i.get_x() + 0.025, i.get_height() - 1.7, \
                    str(int((i.get_height()))), fontsize=10,
                    color='White')
        else:
            ax.text(i.get_x() + 0.045, i.get_height() - 1.5, \
                    str(int((i.get_height()))), fontsize=10,
                    color='White')
    if save_plt_path:
        plt.savefig(save_plt_path+'.pdf', bbox_inches='tight')
    plt.show()


def plt_corr_matrix(matrix_dataframe,
                    axis_fontsize=12, axis_rotation=45,
                    save_plt_path=None):
    sns.set_theme(style="white")

    # Generate a large random dataset
    rs = np.random.RandomState(33)

    # Compute the correlation matrix
    corr = matrix_dataframe.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.xticks(fontsize=axis_fontsize, rotation=axis_rotation)
    plt.yticks(fontsize=axis_fontsize, rotation=axis_rotation)

    if save_plt_path:
        plt.savefig(save_plt_path+'.pdf', bbox_inches='tight')

    plt.show()
