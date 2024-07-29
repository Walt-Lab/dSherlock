"""
auxiliary functions for figure creation

author: Anton Thieme <anton@thiemenet.de>
"""

import numpy as np
import pandas as pd
import os
import math

Concentrations = ['M', 'M', 'H', 'H', 'H', 'H', 'L', 'H', 'L', 'L', 'L', 'H', 'M', 'H', 'M']
Clades = ['1', '4', '1', '2', '4', '*', '1', '*', '4', '2', '3', '3', '2', '*', '3']
DILUTION_FACTOR = 5

# auxiliary function to return y-values for linear function
def line(x, m, y0):

    print("function called with:")
    print("x = {}".format(x))
    print("m = {}".format(m))
    print("y0 = {}".format(y0))

    y = []

    for x_val in x:
        print(x_val)
        y.append(np.power(10, np.log10(x_val)*m +y0))

    print(y)
    return y

# calculate concentration via poisson
def get_concentration_estimate(frac):

    molecules_per_liter = - 1e6 * DILUTION_FACTOR / 0.000755 * math.log(1-frac, math.e)
    molar = molecules_per_liter / (6.0221408e+23)
    femtomolar = molar * 1e15
    print("Estimated Concentration:  {:.2f}fM".format(femtomolar))

    return femtomolar


def get_conc_fm_calcurve_cauris(frac):

    optimized = [1.75619916e-02, 5.98922690e+01, 2.19975771e+02, -1.31581983e+00]

    Min = optimized[0]
    Max = optimized[1]
    X50 = optimized[2]
    Hill = optimized[3]

    X = np.linspace(0.0005, 50, 100000000)
    Y = curve(X, Min, Max, X50, Hill)

    difference_array = np.absolute(Y - frac)
    print(Y)
    print(difference_array)
    index = difference_array.argmin()
    print(index)

    conc_fm = X[index]
    print(frac)
    print(conc_fm)
    return conc_fm

# auxiliary function to return y-values for 4 parameter logistic regression function
def curve(x, Min, Max, X50, Hill):

    y = Min + (Max-Min)/(1+(x/X50)**Hill)

    return y

# auxiliary function to calculate residuals in fitting of 4 parameter logistic regression
def residuals(params, x, y):

    Min_c, Max_c, X50_c, Hill_c = params
    error = ((y - curve(x, Min=Min_c, Max=Max_c, X50=X50_c, Hill=Hill_c)) / y)

    return error

# auxiliary function to load the dSherlock accuracy panel data
def load_dSherlock_Acc_Panel(FOLDER_PATH):

    df = pd.DataFrame(
        columns=['sample', 'totalTrackedN', 'positiveN', 'fractionPos', 'concentrationfM', 'poissonNoiseN',
                 'poissonNoisefM', 'experiment'])

    for file_or_folder in os.listdir(FOLDER_PATH):
        path = os.path.join(FOLDER_PATH, file_or_folder)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('meta.txt'):

                    sample = int(path[path.find("Results_") + len("Results_AccPanel_"):])

                    with open(os.path.join(path, 'meta.txt')) as meta_file:
                        lines = meta_file.readlines()
                        if float(lines[1][lines[1].find("\t") + 1:lines[1].find("\n")]) != 0:
                            positiveN = int(float(lines[0][lines[0].find("\t") + 1:]) * float(
                                lines[1][lines[1].find("\t") + 1:lines[1].find("\n")]))
                        else:
                            positiveN = 0

                        totalTrackedN = float(lines[0][lines[0].find("\t") + 1:])
                        fractionPos = float(lines[1][lines[1].find("\t") + 1:lines[1].find("\n")])
                        concentrationfM = get_concentration_estimate(fractionPos)
                        poissonNoiseN = math.sqrt(positiveN)
                        poissonNoisefM = get_concentration_estimate(poissonNoiseN / totalTrackedN)

                    df.loc[len(df.index)] = [sample, totalTrackedN, positiveN, fractionPos, concentrationfM,
                                             poissonNoiseN, poissonNoisefM, path]

    df.sort_values(by='sample', inplace=True)
    df['concentration'] = Concentrations
    df['clade'] = Clades

    return df


def load_dSherlock_CalCurve(FOLDER_PATH):

    df = pd.DataFrame(columns=['concentration', 'positive', 'experiment'])

    for file_or_folder in os.listdir(FOLDER_PATH):
        path = os.path.join(FOLDER_PATH, file_or_folder)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('meta.txt'):

                    concentration = float(path[path.find("Results_") + len("Results_"):path.find("M") - 1])
                    if path[path.find("M") - 1] == "a":
                        concentration = concentration / 1000
                    elif path[path.find("M") - 1] == "p":
                        concentration = concentration * 1000

                    positive = 0
                    with open(os.path.join(path, 'meta.txt')) as meta_file:
                        lines = meta_file.readlines()
                        positive = float(lines[1][lines[1].find("\t") + 1:lines[1].find("\n")])

                    df.loc[len(df.index)] = [concentration, positive, path]

    return df