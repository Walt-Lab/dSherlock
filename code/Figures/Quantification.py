"""
functions for basic quantification figure plotting

author: Anton Thieme <anton@thiemenet.de>
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import Auxiliary as aux
from matplotlib.patches import Patch
from scipy.optimize import leastsq
import pandas as pd
import seaborn as sns
import scienceplots
import re

from Analysis.Auxiliary import *

DILUTION_FACTOR = 5

FOLDER_PATH_DSHERLOCK_ACC_PANEL = r""
FOLDER_PATH_DSHERLOCK_CALCURVE = r""


def dSherlock_vs_ddPCR(x, y):
    custom_palette = {}
    for ii in range(len(Clades)):
        colorcode = []
        if Clades[ii] == '1':  # purple
            colorcode = [152, 78, 163]
        elif Clades[ii] == '2':  # red
            colorcode = [228, 26, 28]
        elif Clades[ii] == '3':  # green
            colorcode = [77, 175, 74]
        elif Clades[ii] == '4':  # blue
            colorcode = [55, 126, 184]
        else:  # gray
            continue

        custom_palette[ii + 1] = tuple([x / 255 for x in colorcode[0:3]])

    # get PCC
    pearsonsr, pval = stats.pearsonr(x=x, y=y)

    # get curve fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(x), np.log10(y))

    fit_x = np.linspace(min(x), max(x), 100)

    plt.scatter(x=x, y=y, color=list(custom_palette.values()))

    plt.plot(fit_x, aux.line(x=fit_x, m=slope, y0=intercept), linestyle='dashed', color='lightsteelblue')
    plt.plot([1, 10, 100, 1000, 2000], [1, 10, 100, 1000, 2000], linestyle='dashed', color='orange')

    plt.text(x=1.8, y=500, s="PCC: {:.2f}".format(pearsonsr))
    plt.text(x=1.8, y=250, s="Slope: {:.2f}".format(slope))
    plt.text(x=1.8, y=125, s="Average Efficiency: {:.2f}".format(np.mean(np.divide(y, x))))

    plt.ylabel("dSherlock [aM]")
    plt.xlabel("ddPCR [aM]")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1, 2000)
    plt.ylim(1, 2000)

    return plt.gcf()


def dSherlock_vs_qPCR(x, y):

    custom_palette = {}
    for ii in range(len(Clades)):
        colorcode = []
        if Clades[ii] == '1':  # purple
            colorcode = [152, 78, 163]
        elif Clades[ii] == '2':  # red
            colorcode = [228, 26, 28]
        elif Clades[ii] == '3':  # green
            colorcode = [77, 175, 74]
        elif Clades[ii] == '4':  # blue
            colorcode = [55, 126, 184]
        else:  # gray
            continue

        custom_palette[ii + 1] = tuple([x / 255 for x in colorcode[0:3]])

    # get PCC
    pearsonsr, pval = stats.pearsonr(x=x, y=y)

    # get curve fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(x), np.log10(y))

    fit_x = np.linspace(min(x), max(x), 100)

    plt.scatter(x=x, y=y, color=list(custom_palette.values()))

    plt.plot(fit_x, aux.line(x=fit_x, m=slope, y0=intercept), linestyle='dashed', color='lightsteelblue')

    plt.text(x=2e-9, y=500, s="PCC: {:.2f}".format(pearsonsr))
    plt.text(x=2e-9, y=250, s="Slope: {:.2f}".format(slope))

    plt.ylabel("dSherlock [aM]")
    plt.xlabel(r"qPCR $[\frac{1}{2^{Ct}}]$")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1e-9, 1e-5)
    plt.ylim(1, 1000)

    return plt.gcf()


def dSherlock_Acc_Panel_Bars(df):

    sorting_dict_conc = {'H': 1, 'M': 2, 'L': 3}
    sorting_dict_clade = {'1':1, '2': 2, '3': 3, '4': 4, '*': 5}
    df = df.sort_values(by=['clade', 'concentration'], key= lambda col: [sorting_dict_conc[x] if x in sorting_dict_conc.keys() else sorting_dict_clade[x] for x in col])
    Clades = list(df['clade'])
    Concentrations = list(df['concentration'])
    Samples = list(df['sample'])
    df['sample_new'] = range(1, 16)
    print(df)
    df.reset_index(inplace=True)

    fig, ax = plt.subplots()

    hatches = []

    custom_palette = {}
    for ii in range(len(Clades)):
        colorcode = []
        if Clades[ii] == '1':   # purple
            colorcode = [152, 78, 163]
        elif Clades[ii] == '2':   # red
            colorcode = [228, 26, 28]
        elif Clades[ii] == '3':   # green
            colorcode = [77, 175, 74]
        elif Clades[ii] == '4':   # blue
            colorcode = [55, 126, 184]
        else:                      # gray
            colorcode = [153, 153, 153]
        """
        if Concentrations[ii] == 'L':
            colorcode.append(0.3)
        if Concentrations[ii] == 'M':
            colorcode.append(0.7)
        if Concentrations[ii] == 'H':
            colorcode.append(1.0)
        """
        if Concentrations[ii] == 'L':
            hatches.append('/')
        if Concentrations[ii] == 'M':
            hatches.append('///')
        if Concentrations[ii] == 'H':
            hatches.append('/////')

        #custom_palette[ii+1] = tuple([x / 255 for x in colorcode[0:3]] + [colorcode[3]])

        custom_palette[ii + 1] = tuple([x / 255 for x in colorcode[0:3]])

    df.loc[df['fractionPos'] == 0.0, 'concentrationfM'] = 0.0001

    plt.bar(x=df['sample_new'], height=1000*df['concentrationfM'], color=list(custom_palette.values()), yerr=1000*df['poissonNoisefM'], capsize=5)#, hatch=hatches)
    plt.yscale("log")

    plt.table(cellText=list(map(list, zip(*df[['sample', 'concentration', 'clade']].values.tolist()))),
              rowLabels=['sample #', 'concentration', 'clade'],
              loc='bottom', cellLoc='center', edges='open')

    plt.xlim((0.5,15.5))
    plt.ylim(0.08, 1000)

    plt.tick_params(labelbottom=False, bottom=False)

    legend_elements_1 = [Patch(facecolor=(153/255, 153/255, 153/255, 0.3), label='5     CFU/chip'),
                       Patch(facecolor=(153/255, 153/255, 153/255, 0.7), label='50   CFU/chip'),
                       Patch(facecolor=(153/255, 153/255, 153/255, 1.0), label='500 CFU/chip')]
    legend_elements_2 = [Patch(facecolor=(152/255, 78/255, 163/255), label='Clade 1'),
                       Patch(facecolor=(228/255, 26/255, 28/255), label='Clade 2'),
                       Patch(facecolor=(77/255, 175/255, 74/255), label='Clade 3'),
                       Patch(facecolor=(55/255, 126/255, 184/255), label='Clade 4')]

    #plt.gca().add_artist(plt.legend(handles=legend_elements_1, loc='upper right'))
    #plt.gca().add_artist(plt.legend(handles=legend_elements_2, loc='upper right', bbox_to_anchor=(1, 0.80)))

    plt.ylabel('Concentration [aM]')

    ax.tick_params(axis='both', which='both', direction='in')

    return fig


def dSherlock_CalCurve(df):

    df['concentration'] = df['concentration']

    df['positive'] = df['positive'] * 100

    data = df.groupby(by="concentration").agg(list)
    average = df.groupby(by='concentration').mean()
    stdev = df.groupby(by="concentration").std()
    data['average'] = average
    data['stdev'] = stdev
    data_0 = data.loc[data.index == 0.0]
    data = data.loc[data.index != 0.0]

    print(data['average'])

    background = data_0.loc[0.0, 'average'] + (3 * np.std(list(data_0['positive'])))
    print("BACKGROUND =\t{}".format(background))

    # Initial guess for parameters
    Min_0 = 0.001#0.0
    Max_0 = 0.9#0.8
    X50_0 = 0.1#500
    Hill_0 = -2#-1
    #Min_0 = 0.001#0.0
    #Max_0 = 0.6#0.8
    #X50_0 = 21097#500
    #Hill_0 = -2#-1

    w = 0.1
    width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)

    optimized = leastsq(func=aux.residuals, x0=np.array([Min_0, Max_0, X50_0, Hill_0]), args=(np.array(data.index), np.array(data.average)))
    print("OPTIMIZED:\t{}".format(optimized))
    Min = optimized[0][0]
    Max = optimized[0][1]
    X50 = optimized[0][2]
    Hill = optimized[0][3]

    X = np.linspace(0.1, 5000, 1000000)
    #X = np.linspace(10, 500000, 1000000)
    Y = aux.curve(X, Min, Max, X50, Hill)

    residuals = np.subtract(np.array(data.average), np.array(aux.curve(np.array(data.index), Min, Max, X50, Hill)))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.array(data.average)-np.mean(np.array(data.average)))**2)
    r_squared = 1 - (ss_res / ss_tot)

    X_LOD = np.linspace(0.0, 1.0, 1000000)
    Y_LOD = aux.curve(X_LOD, Min, Max, X50, Hill)
    LOD_signal = min(filter(lambda i: i > background, Y_LOD))
    print("LOD signal =\t{}".format(LOD_signal))
    LOD = X_LOD[np.where(Y_LOD == LOD_signal)[0][0]]
    print("LOD = \t{}".format(LOD))

    fig, ax = plt.subplots()

    # Plot results
    plt.plot(X, Y)

    plt.scatter(x=data.index, y=data.average, color='orange')
    plt.errorbar(x=data.index, y=data.average, yerr=data.stdev, linestyle='None', color='orange', capsize=2.0)

    plt.axhline(y=background, color='b', linestyle='--', alpha=0.5)
    plt.text(x=0.1, y=50, s='LOD = {:.2f}aM'.format(1000*LOD))
    plt.text(x=0.1, y=20, s='$R^2$ = {:.2f}'.format(r_squared))
    #plt.text(x=0.1, y=50, s='LOD = {:.2f}fM'.format(LOD))
    plt.xlabel('Target concentration [fM]')
    plt.ylabel('Positive partitions [%]')
    plt.xscale('log')
    plt.yscale('log')

    ax.tick_params(axis='both', which='both', direction='in')

    return fig


if __name__ == "__main__":

    plt.style.use(['science', 'no-latex', 'nature'])
    plt.rcParams['svg.fonttype'] = 'none'