import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch

from Analysis.Auxiliary import *


def plotTimeseries(df, sets, titles, suptitle, imaging_interval, every_nth=100):

    n = len(sets)

    fig, ax = plt.subplots(1, n, sharey=True, figsize=(20, 12))

    df['time [min]'] = df['frame'] * imaging_interval

    for ii, set in enumerate(sets):
        sns.lineplot(ax=ax[ii], data=df.loc[(df['set'] == sets[ii]) & (df['UID'] % every_nth == 0)], x='time [min]',
                 y='fq_delta', hue='UID', linewidth=0.5, alpha=0.8, legend=False, palette=sns.color_palette('colorblind'))
        ax[ii].set_title(titles[ii])

    plt.suptitle(suptitle)

    return fig


def plotAdmix(df, mt_sets, title):

    for set in df['set'].unique():
        if set.startswith('WT'):
            plt.scatter(0, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'MT')]) / len(df.loc[df['set'] == set]), 5), color='r')
        elif set in mt_sets:
            plt.scatter(100, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'MT')]) / len(df.loc[df['set'] == set]), 5), color='r')
            print(len(df.loc[(df['set'] == set) & (df['cluster'] == 'MT')]) / len(df.loc[df['set'] == set]), 5)

        elif set.startswith('MT_10'):
            plt.scatter(10, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'MT')]) / len(df.loc[df['set'] == set]), 5), color='r')
        elif set.startswith('MT_25'):
            plt.scatter(25, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'MT')]) / len(df.loc[df['set'] == set]), 5), color='r')
        elif set.startswith('MT_50'):
            plt.scatter(50, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'MT')]) / len(df.loc[df['set'] == set]), 5), color='r')
        elif set.startswith('MT_75'):
            plt.scatter(75, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'MT')]) / len(df.loc[df['set'] == set]), 5), color='r')
        elif set.startswith('MT_90'):
            plt.scatter(90, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'MT')]) / len(df.loc[df['set'] == set]), 5), color='r')

    for set in df['set'].unique():
        if set.startswith('WT'):
            plt.scatter(0, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'WT')]) / len(df.loc[df['set'] == set]), 5), color='b')
            print(len(df.loc[(df['set'] == set) & (df['cluster'] == 'WT')]) / len(df.loc[df['set'] == set]), 5)
        elif set in mt_sets:
            plt.scatter(100, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'WT')]) / len(df.loc[df['set'] == set]), 5), color='b')
        elif set.startswith('MT_10'):
            plt.scatter(10, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'WT')]) / len(df.loc[df['set'] == set]), 5), color='b')
        elif set.startswith('MT_25'):
            plt.scatter(25, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'WT')]) / len(df.loc[df['set'] == set]), 5), color='b')
        elif set.startswith('MT_50'):
            plt.scatter(50, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'WT')]) / len(df.loc[df['set'] == set]), 5), color='b')
        elif set.startswith('MT_75'):
            plt.scatter(75, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'WT')]) / len(df.loc[df['set'] == set]), 5), color='b')
        elif set.startswith('MT_90'):
            plt.scatter(90, get_concentration_estimate(len(df.loc[(df['set'] == set) & (df['cluster'] == 'WT')]) / len(df.loc[df['set'] == set]), 5), color='b')

    legend_elements_1 = [Patch(facecolor='b', label='wildtype'),
                           Patch(facecolor='r', label='mutant')]
    plt.gca().add_artist(plt.legend(handles=legend_elements_1, loc='upper center'))

    #plt.ylim(0, 10)
    plt.xlabel('mutant AF [%]')
    plt.ylabel('measured allele concentration [fM]')
    plt.title(title)

    fig = plt.gcf()

    return fig