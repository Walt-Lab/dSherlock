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

from Analysis_PlateReader import *
from Analysis.Auxiliary import *


# accuracy panel measurements of the 12 samples containing C. Auris with qPCR, dSherlock and ddPCR
Acc_Panel_qPCR_CA_Ct = [23.4, 22.7, 20.7, 18.5, 19.5, 28.2, 26.9, 26.3, 28.4, 19.2, 22.6, 24.2]
Acc_Panel_dSherlock_CA_aM = [29.6, 45.7, 70.5, 836.7, 150.2, 2.5, 4.5, 2.4, 1.9, 258.4, 2.5, 5.6]
Acc_Panel_ddPCR_CA_cpul = [[3.75, 5.03], [6.49, 9.59], [12.7, 12.6], [51.8, 68.1], [48.1, 42.4], [0.619, 0.807], [1.15, 1.35], [0.561, 0.539], [0.138, 0.129], [63.8, 68.7], [0.633, 0.246], [1.47, 0.645]]

# auxiliary global variables
Samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Concentrations = ['M', 'M', 'H', 'H', 'H', 'H', 'L', 'H', 'L', 'L', 'L', 'H', 'M', 'H', 'M']
Clades = ['1', '4', '1', '2', '4', '*', '1', '*', '4', '2', '3', '3', '2', '*', '3']
DILUTION_FACTOR = 5

CA_IDT_C_FACTOR = 3.101574114

FOLDER_PATH_DSHERLOCK_ACC_PANEL = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Accuracy Panel\dSherlock\Images\rox exclude up"
FOLDER_PATH_DSHERLOCK_CALCURVE = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Calibration Curve\Images\Resultsss"


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
    #afont = {'fontname': 'Arial', 'size':7}

    # dSherlock vs ddPCR
    """
    # calculating means of the duplicate ddPCR measurements
    Acc_Panel_ddPCR_mean = np.average([l for l in Acc_Panel_ddPCR_CA_cpul], axis=1)
    # calculating aM from cp/ul and accounting for dilution factor in ddPCR
    Acc_Panel_ddPCR_mean_aM = []
    for meas in Acc_Panel_ddPCR_mean:
        Acc_Panel_ddPCR_mean_aM.append(10 * (meas * 10) / 6.022)

    fig = dSherlock_vs_ddPCR(x=Acc_Panel_ddPCR_mean_aM, y=Acc_Panel_dSherlock_CA_aM)
    fig.set_size_inches(5.6/2.54, 3.4/2.54)
    """
    #plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\AccPanel_dSherlock_ddPCR.svg', format='svg', dpi=450, transparent=True)


    # dSherlock vs qPCR
    """
    fig = dSherlock_vs_qPCR(x=1 / np.power(2, Acc_Panel_qPCR_CA_Ct), y=Acc_Panel_dSherlock_CA_aM)
    fig.set_size_inches(5.6/2.54, 3.4/2.54)
    """
    #plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\AccPanel_dSherlock_qPCR.svg', format='svg', dpi=450, transparent=True)


    # dSherlock bars

    df = aux.load_dSherlock_Acc_Panel(FOLDER_PATH_DSHERLOCK_ACC_PANEL)
    print(df.columns)
    print(df)
    print(df['fractionPos'])
    #df['concentrationfM'] = df['fractionPos'].apply(aux.get_conc_fm_calcurve_cauris)

    print(df['concentrationfM'])
    fig = dSherlock_Acc_Panel_Bars(df)
    #fig.set_size_inches(10/2.54, 7.2/2.54)
    fig.set_size_inches(8.2 / 2.54, 6.5 / 2.54)

    plt.show()

    #plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\AccPanel_smaller_8.svg', format='svg', dpi=450, transparent=True)


    # dSherlock Calibration Curve

    """
    df = aux.load_dSherlock_CalCurve(FOLDER_PATH_DSHERLOCK_CALCURVE)
    fig = dSherlock_CalCurve(df)
    print(df)
    #fig.set_size_inches(8.6/2.54, 6/2.54)
    fig.set_size_inches(5.6 / 2.54, 3.4 / 2.54)
    """
    #plt.show()
    #plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\Thesis\Schematic\CAuris_CalCurve.svg', format='svg', dpi=450, transparent=True)

    #plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\1_CAurisCalcurve\CAuris_CalCurve.svg', format='svg', dpi=450, transparent=True)


    # Representative Timeseries

    #df = pd.read_csv(r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\figure\analyze\image\Results_50fM_03_cutout2\timeseries_processed.csv")
    """
    df = df.loc[df['frame'] >= 2]
    df['Time [min]'] = (df['frame']-2) * 2
    df['Fluorescence [RFU]'] = df['fq_delta']

    fig, ax = plt.subplots()
    sns.lineplot(ax=ax, data=df.loc[df['Time [min]'] <= 120], x='Time [min]', y='Fluorescence [RFU]', hue='UID', linewidth=0.5, alpha=0.8, legend=False,
                     palette=sns.color_palette('colorblind'))
    plt.hlines(y=3000, xmin=0, xmax=120, linestyle='dashed', color='orange')
    plt.xlabel('Time [min]')
    plt.ylabel('Fluorescence [RFU]')
    ax.tick_params(axis='both', which='both', direction='in')
    #fig.set_size_inches(5.6/2.54, 3.4/2.54)
    fig.set_size_inches(8.6 / 2.54, 6 / 2.54)
    """
    #plt.show()
    #plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\Thesis\Schematic\Timeseries.svg', format='svg', dpi=450, transparent=True)

    #plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\Timeseries.svg', format='svg', dpi=450, transparent=True)


    # Amies/Human bulk


    #PATH1 = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\5_NonSpecificDNAInterference\Bulk\2023_04_20_CAuris_Inhibition_Control_3.xlsx"
    #PATH2 = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\5_NonSpecificDNAInterference\Bulk\2023_04_21_CAuris_Inhibition_Control_4.xlsx"
    #PATH3 = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\5_NonSpecificDNAInterference\Bulk\2023_04_21_CAuris_Inhibition_Control_5.xlsx"

    """
    HEADER_ROWS_OFFSET = 1
    HEADER_ROWS = [32 - HEADER_ROWS_OFFSET, 112 - HEADER_ROWS_OFFSET, 192 - HEADER_ROWS_OFFSET,
                   272 - HEADER_ROWS_OFFSET]  # 1 in excel = 0 here
    REPLICATES = 1  # replicates have same number/column (1) or same letter/row (2)

    LABELS1 = ["CA_H 1", "H 1", "CA_A 1", "A 1", "CA 1", "CA_H 2", "CA_A 2", "CA 2", 'empty']
    df1 = getData(PATH1)

    LABELS2 = ["CA_H 4", "H 4", "CA_A 4", "A 4", "CA 4", "H 2_2", "A 2_2", "CA 2_2", 'empty']
    df2 = getData(PATH2)

    LABELS3 = ["CA_H 3", "H 3", "CA_A 3", "A 3", "CA 3", 'empty']
    df3 = getData(PATH3)

    fig, ax = plt.subplots()

    df1.dropna(axis='columns', inplace=True)
    df2.dropna(axis='columns', inplace=True)
    df3.dropna(axis='columns', inplace=True)

    d1 = calcStats(df1)
    d1.columns = LABELS1
    d2 = calcStats(df2)
    d2.columns = LABELS2
    d3 = calcStats(df3)
    d3.columns = LABELS3

    d1 = d1.reindex(columns=["CA_H 1", "H 1", "CA_A 1", "A 1", "CA 1", "CA_H 2", "CA_A 2", "CA 2"])
    d2 = d2.reindex(columns=["H 2_2", "A 2_2", "CA 2_2"])
    d3 = d3.reindex(columns=["CA_H 3", "H 3", "CA_A 3", "A 3", "CA 3"])

    d = pd.concat([d1, d2, d3])
    print(d.columns)

    long_mean_d = pd.melt(d.xs('mean', level='stats').reset_index(), id_vars=['time'], value_name='mean', var_name=VARIABLE)
    long_data_d = pd.melt(d.xs('data', level='stats').reset_index(), id_vars=['time'], value_name='data', var_name=VARIABLE)
    long_sem_d = pd.melt(d.xs('sem', level='stats').reset_index(), id_vars=['time'], value_name='sem', var_name=VARIABLE)

    print(long_mean_d)
    print(long_sem_d)
    print(long_data_d)

    long = long_mean_d.merge(long_data_d, on=['time', VARIABLE])
    long = long.merge(long_sem_d, on=['time',VARIABLE])
    long.time = long.time.apply(lambda x: (datetime.datetime.combine(datetime.date.today(), x) - datetime.datetime.combine(datetime.date.today(), long.loc[0, 'time'])).total_seconds() / 60)
    print(long)

    long = long.groupby(['time', 'condition']).first().reset_index()
    print(long)

    long['data1'] = long.data.apply(lambda x: x[0])
    long['data2'] = long.data.apply(lambda x: x[1])
    long['data3'] = long.data.apply(lambda x: x[2])

    #plotFluorTimeAndEndpoint(long)

    long['set'] = long.condition.apply(lambda x: x[x.find(" "):])
    long['condition'] = long.condition.apply(lambda x: x[:x.find(" ")])
    print(long)

    temp_ca = long.loc[(long['condition'] == 'CA')][['time', 'set', 'mean']]
    temp_ca = temp_ca.rename({'mean':'mean_ca'}, axis=1)
    long = long.merge(temp_ca, on=['time', 'set'])
    long['Norm. fl. Intensity'] = long['mean'] / long['mean_ca']
    group_mean = long.groupby(['time', 'condition', 'set'])['Norm. fl. Intensity'].mean()
    group_mean = group_mean.reset_index()
    print(group_mean)
    print(temp_ca)
    print(long)

    names = {'A': 'Amies', 'H': 'hgDNA', 'CA_A': 'C.auris\n+Amies', 'CA_H':'C.auris\n+hgDNA', 'CA':'CA'}
    order_id = {'CA_A':1, 'A':2, 'CA_H':3, 'H':4, 'CA':5}
    group_mean['order_id'] = [order_id[x] for x in group_mean['condition']]
    group_mean['condition'] = [names[x] for x in group_mean['condition']]

    plt.axhline(y=1.0, xmin=0, xmax=1, linestyle='dashed', color='black')
    #sns.barplot(data=group_mean.loc[(group_mean['time'] == 150.0) & (group_mean['condition'] != 'CA')], x='condition', y='normalized', capsize=0.2)
    sns.barplot(data=group_mean.loc[group_mean['condition'] != 'CA'], x='condition', y='Norm. fl. Intensity', capsize=0.2, order=['C.auris\n+Amies', 'Amies', 'C.auris\n+hgDNA', 'hgDNA'])
    #plt.hlines(y=1.0)
    plt.xlabel('')
    fig.set_size_inches(5.6 / 2.54, 3.4 / 2.54)
    """
    #plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\Amies_Human_Bulk.svg', format='svg', dpi=450, transparent=True)
    #plt.show()



    #Amies/Human Digital

    #PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\5_NonSpecificDNAInterference\Digital"
    """
    THRESHOLD_POSITIVE = 3000
    SHORT_SETS = ['CA_1', 'CA_2', 'CA_2_2']

    df = batch_load_and_combine(PATH, "features_{}.csv".format(76))
    print(df['set'].unique())
    print(df)

    #df = clean(df, dropna=False)
    print(df)

    fracs = pd.DataFrame({'set':[], 'frac':[], 'Condition':[], 'batch':[]})

    for set in df['set'].unique():

        if set in SHORT_SETS:
            #plt.hist(df.loc[df['set'] == set]['feat_fq_initial'])
            frac = len(df.loc[(df['set'] == set) & (df['feat_fq_initial'] > 15000)]) / len(df.loc[df['set'] == set])
        else:
            frac = len(df.loc[(df['set'] == set) & (df['feat_fq_delta_max'] > THRESHOLD_POSITIVE)]) / len(df.loc[df['set'] == set])

        condition = set[:re.search(r"\d", set).start()-1]
        batch = set[re.search(r"\d", set).start():]
        print(condition)
        print(batch)
        fracs.loc[len(fracs.index)] = [set, frac, condition, batch]
    #plt.show()
    print(fracs)

    temp_ca = fracs.loc[(fracs['Condition'] == 'CA')][['frac', 'batch']]
    temp_ca = temp_ca.rename({'frac':'frac_ca'}, axis=1)
    fracs = fracs.merge(temp_ca, on=['batch'])
    fracs['Norm.\npositive partitions'] = fracs['frac'] / fracs['frac_ca']

    names = {'A': 'Amies', 'H': 'hgDNA', 'CA_A': 'C.auris\n+Amies', 'CA_H':'C.auris\n+hgDNA', 'CA':'CA'}
    order_id = {'CA_A':1, 'A':2, 'CA_H':3, 'H':4, 'CA':5}
    fracs['order_id'] = [order_id[x] for x in fracs['Condition']]
    fracs['Condition'] = [names[x] for x in fracs['Condition']]


    print(fracs)

    fig, ax = plt.subplots()
    sns.barplot(data=fracs.loc[fracs['Condition'] != 'CA'], x='Condition', y='Norm.\npositive partitions', capsize=0.2, order=['C.auris\n+Amies', 'Amies', 'C.auris\n+hgDNA', 'hgDNA'])
    #plt.hlines(y=1.0)
    #plt.ylim(-0.2, 2.0)
    plt.xlabel('')
    plt.axhline(y=1.0, xmin=0, xmax=1, linestyle='dashed', color='black')
    #plt.yscale('log')
    fig.set_size_inches(5.6 / 2.54, 3.4 / 2.54)
    """
    #plt.savefig(r'C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\2_Quantification\Amies_Human.svg', format='svg', dpi=450, transparent=True)
    #plt.show()


    """
    put Acc panel dotplots into fM range
    Shift inhibition before acc panel stuff
    change y-axis of timeseries to 10^3
    """