import pandas as pd
import math
import os
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

DILUTION_FACTOR = 5

# used
def set_smooth(df, column):
    df[column] = df.groupby('UID')[column].rolling(window=3, center=True, min_periods=1, win_type='Gaussian').mean().reset_index(0, drop=True)

    return df


# used
# velocity as difference per minute
def set_velocity(df, column):
    df['{}_velocity'.format(column)] = df.groupby('UID')[column].diff() / IMAGING_INTERVAL_MIN

    return df


# initial smoothened measurement
def set_initial(df):
    initial = df.where(df['frame'] <= 3).groupby('UID')['mean'].mean()
    df = df.merge(initial, how='outer', on='UID', suffixes=(None, '_initial'))
    df.rename({'mean_initial': 'initial'}, axis=1, inplace=True)

    return df


# final smoothened measurement
def set_final(df):
    final = df.where(df['frame'].max() - df['frame'] < 3).groupby('UID')['mean'].mean()
    df = df.merge(final, how='outer', on='UID', suffixes=(None, '_final'))
    df.rename({'mean_final': 'final'}, axis=1, inplace=True)

    return df

# used
# cumulative delta
def set_delta(df, column):
    initial = df.where(df['frame'] == 1).groupby('UID')[column].mean()
    df = df.merge(initial, how='outer', on='UID')
    df.rename({"{}_x".format(column): column, "{}_y".format(column): 'initial'}, axis=1, inplace=True)
    df['{}_delta'.format(column)] = df[column] - df['initial']
    df.drop(['initial'], axis=1, inplace=True)

    return df


# cumulative delta reaches threshold to be classified as positive
def set_positive(df):
    positive = df.groupby('UID')['delta'].max() > 1500
    df = df.merge(positive, how='outer', on='UID', suffixes=(None, '_positive'))
    df.rename({'delta_positive': 'positive'}, axis=1, inplace=True)

    return df


# fraction of wells classified as positive
def get_fraction_positive(df):
    total = df['UID'].max()
    positive = df.where(df['frame'] == 1)['positive'].sum()

    print("Total # Wells:            {}".format(total))
    print("Total # Positive Wells:   {}".format(positive))
    print("\nPercent Positive:         {:.2f}%".format(100 * positive / total))
    print("\n")

    return positive / total


def get_concentration_estimate(frac):
    molecules_per_liter = - 1e6 * DILUTION_FACTOR / 0.000755 * math.log(1 - frac, math.e)
    molar = molecules_per_liter / (6.0221408e+23)
    femtomolar = molar * 1e15
    print("Estimated Concentration:  {:.2f}fM".format(femtomolar))

    return femtomolar


def set_max_velocity(df):
    df['max_velocity'] = df.groupby('UID')['velocity'].transform('max')

    return df


def set_area(df):
    area = df[df['delta'].notna()].groupby('UID').apply(lambda g: integrate.trapz(g.delta, x=g.frame))
    area.name = 'area'
    df = df.merge(area, on='UID', how='outer')

    return df


def set_max_delta(df):
    df['max_delta'] = df.groupby('UID')['delta'].transform('max')

    return df


def set_tp_max_delta(df):
    tp_max_delta = df.loc[df['delta'] == df['max_delta']]['frame']
    tp_max_delta = tp_max_delta.reset_index(drop=True)
    # tp_max_velocity.index = pd.RangeIndex(start=1, stop=tp_max_velocity.stop + 1, step=1)
    tp_max_delta = tp_max_delta.loc[tp_max_delta.index.repeat(MAX_FRAME)]

    df['tp_max_delta'] = tp_max_delta.values

    return df


def set_final_delta(df):
    df['final_delta'] = df['final'] - df['initial']

    return df


def set_final_velocity(df):
    final_velocity = df.loc[df['frame'] == df['frame'].max() - 1].groupby('UID')['velocity'].mean()
    df = df.merge(final_velocity, how='outer', on='UID', suffixes=(None, '_final'))
    df.rename({'velocity': 'final_velocity'}, axis=1, inplace=True)

    return df


def set_rox_initial_norm_factor(df, df_rox):
    mean_firstN = \
    (df_rox.loc[df_rox.frame == 1]['mean'].reset_index().add(df_rox.loc[df_rox.frame == 2]['mean'].reset_index()) / 2)[
        'mean']
    max_rox = mean_firstN.max()
    rox_initial_norm_factor = mean_firstN / max_rox
    rox_initial_norm_factor.index.rename('UID', inplace=True)
    rox_initial_norm_factor.index = rox_initial_norm_factor.index + 1
    rox_initial_norm_factor.name = 'norm_factor_initial'
    df = df.merge(right=rox_initial_norm_factor, how='left', on='UID')

    return df


def set_normalized(df):
    df['delta_normalized'] = df['delta'] * df['norm_factor']

    return df


def set_tp_max_velocity(df):
    # df['max_velocity'] = df.groupby('UID')['velocity'].transform('max')

    tp_max_velocity = df.loc[df['velocity'] == df['max_velocity']]['frame']
    tp_max_velocity = tp_max_velocity.reset_index(drop=True)
    # tp_max_velocity.index = pd.RangeIndex(start=1, stop=tp_max_velocity.stop + 1, step=1)
    tp_max_velocity = tp_max_velocity.loc[tp_max_velocity.index.repeat(MAX_FRAME)]

    df['tp_max_velocity'] = tp_max_velocity.values

    return df

# used
def set_rox(df, df_rox):
    df = df.merge(right=df_rox[['mean', 'UID', 'frame']], how='left', on=['UID', 'frame'])
    df.rename({'mean_x': 'fq', 'mean_y': 'rox'}, axis=1, inplace=True)

    return df


def set_class(df):
    fig, ax = plt.subplots(2, 6)

    ax[0] = plt.hist(df.loc[df['frame'] == 1]['max_velocity'], bins=100)
    plt.yscale("log")
    plt.show()

    return


def set_rox_concurrent_norm_factor(df):
    max_rox = df.groupby('UID')['rox'].max()
    df = df.merge(right=max_rox, on='UID')
    df.rename({'rox_x': 'rox', 'rox_y': 'rox_max'}, axis=1, inplace=True)
    df['norm_factor_concurrent'] = df['rox'] / df['rox_max']
    df.drop(['rox_max'], axis=1, inplace=True)

    return df

# used
def set_initial_intensity(df, column):
    initial = df.loc[df.groupby('UID')[column].apply(lambda x: x.first_valid_index()), :][['UID', column]]
    df = df.merge(initial, how='outer', on='UID')
    df.rename({"{}_x".format(column): column, "{}_y".format(column): "feat_{}_initial".format(column)}, axis=1,
              inplace=True)

    return df

# used
def set_final_intensity(df, column):
    final = df.loc[df.groupby('UID')[column].apply(lambda x: x.last_valid_index()), :][['UID', column]]
    # print(final)
    df = df.merge(final, how='outer', on='UID')
    df.rename({"{}_x".format(column): column, "{}_y".format(column): "feat_{}_final".format(column)}, axis=1,
              inplace=True)

    return df

# used
def set_max_intensity(df, column):
    maxx = df.groupby('UID')[column].max()
    df = df.merge(maxx, how='outer', on='UID')
    df.rename({"{}_x".format(column): column, "{}_y".format(column): "feat_{}_max".format(column)}, axis=1,
              inplace=True)

    return df

# used
def set_min_intensity(df, column):
    minn = df.groupby('UID')[column].min()
    df = df.merge(minn, how='outer', on='UID')
    df.rename({"{}_x".format(column): column, "{}_y".format(column): "feat_{}_min".format(column)}, axis=1,
              inplace=True)

    return df

# used
def set_avg_intensity(df, column):
    avgg = df.groupby('UID')[column].mean()
    df = df.merge(avgg, how='outer', on='UID')
    df.rename({"{}_x".format(column): column, "{}_y".format(column): "feat_{}_avg".format(column)}, axis=1,
              inplace=True)

    return df

# used
def set_timepoint_equal(df, column, attribute):
    timepoint = df.loc[df[column] == df["feat_{}_{}".format(column, attribute)]][['UID', 'frame']]
    df = df.merge(timepoint, how='outer', on='UID')
    df.rename({"frame_x": 'frame', "frame_y": "feat_{}_tp_{}".format(column, attribute)}, axis=1, inplace=True)
    df["feat_{}_tp_{}".format(column, attribute)] = (df["feat_{}_tp_{}".format(column, attribute)]-1) * IMAGING_INTERVAL_MIN

    return df

# used
def set_timepoint_bigger(df, column, attribute):
    timepoint = df.loc[df[column] >= df["feat_{}_{}".format(column, attribute)]][['UID', 'frame']].groupby('UID').head(
        1)
    df = df.merge(timepoint, how='outer', on='UID')
    df.rename({"frame_x": 'frame', "frame_y": "feat_{}_tp_{}".format(column, attribute)}, axis=1, inplace=True)
    df["feat_{}_tp_{}".format(column, attribute)] = (df["feat_{}_tp_{}".format(column, attribute)]-1) * IMAGING_INTERVAL_MIN

    return df

# used
def set_final_area(df, column):
    area = df.groupby('UID')[column].apply(np.trapz, dx=IMAGING_INTERVAL_MIN)
    df = df.merge(area, how='outer', on='UID')
    df.rename({"{}_x".format(column): column, "{}_y".format(column): "feat_{}_area_final".format(column)}, axis=1,
              inplace=True)

    return df

# used
def set_active_velocity(df, final_frame):
    # identify timepoint where velocity falls below half of the maximum velocity after timepoint max velocity
    # take intensity at that timepoint divided by the timepoint

    timepoint = df.loc[(df['fq_delta_velocity'] <= 0.5 * df['feat_fq_delta_velocity_max']) & (
                df['time'] >= df['feat_fq_delta_velocity_tp_max'])][['UID', 'time']].groupby('UID').head(1)
    df = df.merge(timepoint, how='outer', on='UID')
    df.rename({"time_x": 'time', "time_y": "feat_tp_end_active"}, axis=1, inplace=True)
    timepoint = df.loc[(df['fq_delta_velocity'] >= 0.5 * df['feat_fq_delta_velocity_max']) & (
            df['time'] <= df['feat_fq_delta_velocity_tp_max'])][['UID', 'time']].groupby('UID').head(1)
    df = df.merge(timepoint, how='outer', on='UID')
    df.rename({"time_x": 'time', "time_y": "feat_tp_start_active"}, axis=1, inplace=True)

    df['feat_tp_end_active'].fillna(value=(final_frame-1)*IMAGING_INTERVAL_MIN, inplace=True)
    df['feat_tp_end_active'].fillna(value=0, inplace=True)

    end = df.loc[df['time'] == df['feat_tp_end_active']][['UID', 'fq_delta', 'feat_tp_end_active']]
    #end.drop(['feat_tp_end_active', 'fq_delta'], axis=1, inplace=True)

    start = df.loc[df['time'] == df['feat_tp_start_active']][['UID', 'fq_delta', 'feat_tp_start_active']]
    #start.drop(['feat_tp_start_active', 'fq_delta'], axis=1, inplace=True)

    active_velocity = end.merge(start, how='inner', on='UID')
    active_velocity['feat_active_velocity'] = (active_velocity['fq_delta_x'] - active_velocity['fq_delta_y']) / (active_velocity['feat_tp_end_active'] - active_velocity['feat_tp_start_active'])
    active_velocity.drop(['fq_delta_x', 'fq_delta_y', 'feat_tp_end_active', 'feat_tp_start_active'], axis=1, inplace=True)

    df = df.merge(active_velocity, how='outer', on='UID')

    return df

# used
def set_intensities(df, timepoints):
    for time in timepoints:
        intensities = df.loc[df['frame'] == time][['UID', 'fq_delta']]
        df = df.merge(intensities, how='outer', on='UID')
        df.rename({"fq_delta_x": 'fq_delta', "fq_delta_y": 'feat_fq_delta_tp_{}'.format(time)}, axis=1, inplace=True)

    print(df)
    return df


def process(path, path_rox):
    ######## DATA LOADING ##########

    # pull fq dataframe and sort its values differently for easier debugging
    d = pd.read_csv(path)
    d['UID'] = d['label']  ############## replace with batch
    d.sort_values(by=['UID', 'frame'], inplace=True)
    print(d)
    d = d.loc[:, ['UID', 'frame', 'mean', 'y', 'x']]

    # pull rox dataframe and add it to the fq dataframe
    d_rox = pd.read_csv(path_rox)
    d_rox['UID'] = d_rox['label']
    d = set_rox(df=d, df_rox=d_rox)

    print(d['frame'].max())

    d['time'] = (d['frame'] - 1)*IMAGING_INTERVAL_MIN
    print(d)


    ######### SMOOTHENING ##########

    # replace each timeseries with a rolling average over 3 time points
    if(SMOOTHEN):
        d = set_smooth(d, 'fq') ##################


    ######## BACKGROUND SUBTRACTION #########

    # set the cumulative delta for each smoothened timeseries
    d = set_delta(d, 'fq')


    ######## VELOCITY ESTIMATION #########

    # set the current velocity for each smoothened timeseries and their cumulative delta timeseries
    d = set_velocity(d, 'fq_delta')

    print(d)
    d.to_csv(r"{}\{}".format(path[0:path.rfind('\\')], 'timeseries_processed.csv'))


    timeseries = d.columns

    timeseries = [s for s in timeseries if 'fq_delta' in s or 'UID' in s or 'frame' in s or 'time' in s]
    print(timeseries)

    d_stable = d

    for final_frame in FINAL_FRAMES_TO_TRY:

        d = d_stable
        d = d.loc[d['frame'] <= final_frame]

        ######### FEATURE EXTRACTION #########

        for series in ['fq', 'rox']:

            # initial intensity
            d = set_initial_intensity(d, series)

        for series in [s for s in timeseries if 'fq_delta' in s]:

            ## intensities ##

            # final intensity
            d = set_final_intensity(d, series)

            # max intensity
            d = set_max_intensity(d, series)

            # min intensity
            d = set_min_intensity(d, series)

            # average intensity
            d = set_avg_intensity(d, series)

            ## intensity at time points ##

            # intensity at time point area bigger average area

            ## intensity time points ##

            # time point max intensity
            d = set_timepoint_equal(d, series, 'max')

            # time point min intensity
            d = set_timepoint_equal(d, series, 'min')

            # time point intensity bigger average intensity
            d = set_timepoint_bigger(d, series, 'avg')  # need to find minimal frame where true

            ## areas ##

            # final area / max area
            d = set_final_area(d, series)

        print(d)
        # active velocity
        d = set_active_velocity(d, final_frame)

        # intensity at certain timepoints
        #d = set_intensities(d, range(1, 92))

        print(d)
        d = d.loc[d['frame'] == final_frame].drop(['frame', 'fq', 'rox', 'fq_delta', 'fq_delta_velocity'], axis=1)

        d.to_csv(r"{}\{}_{}.csv".format(path[0:path.rfind('\\')], 'features', final_frame))

    return


def batch(experiment_path):
    for file_or_folder in os.listdir(experiment_path):

        path = os.path.join(experiment_path, file_or_folder)

        if os.path.isdir(path):

            for file in os.listdir(path):

                #if file.startswith("features"):
                    #os.remove(os.path.join(path, file))

                if file.endswith("timeseries.csv"):
                    print(path)
                    print("\n")

                    process(path=os.path.join(path, file), path_rox=os.path.join(path, "timeseries_rox.csv"))

                    print("\n\n\n")


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)





    #IMAGING_INTERVAL_MIN = 2
    #MAX_FRAME = 76
    #FINAL_FRAMES_TO_TRY = range(6, 77, 5)
    #SMOOTHEN = 1
    #FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\3_Kinetics\3_FKS1Admix\Results"

    #batch(FOLDER_PATH)

    #IMAGING_INTERVAL_MIN = 10

    #MAX_FRAME = 145
    #FINAL_FRAMES_TO_TRY = range(7, 146, 6)
    #SMOOTHEN = 1

    #FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\3_Kinetics\1_FKS1CalCurve\20230324_FKS1_WT_CalCurve_1_Results"

    #batch(FOLDER_PATH)

    #IMAGING_INTERVAL_MIN = 10

    #MAX_FRAME = 145
    #FINAL_FRAMES_TO_TRY = range(7, 146, 6)
    #SMOOTHEN = 1

    #FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\3_Kinetics\1_FKS1CalCurve\20230325_FKS1_WT_CalCurve_2_Results"

    #batch(FOLDER_PATH)

    #IMAGING_INTERVAL_MIN = 10

    #MAX_FRAME = 145
    #FINAL_FRAMES_TO_TRY = range(7, 146, 6)
    #SMOOTHEN = 1

    #FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\3_Kinetics\1_FKS1CalCurve\20230326_FKS1_WT_CalCurve_3_Results"

    #batch(FOLDER_PATH)

    #IMAGING_INTERVAL_MIN = 2

    #MAX_FRAME = 151
    #FINAL_FRAMES_TO_TRY = range(31, 152, 30)
    #SMOOTHEN = 1

    #FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\3_Kinetics\1_FKS1CalCurve\20230324_FKS1_MT_CalCurve_3_Results"

    #batch(FOLDER_PATH)

    #IMAGING_INTERVAL_MIN = 10

    #MAX_FRAME = 91
    #FINAL_FRAMES_TO_TRY = range(3, 92, 1)
    #SMOOTHEN = 1

    #FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\Experiments\Digital\20230407_FKS1_Admix_15h_10fM"

    #batch(FOLDER_PATH)

    #IMAGING_INTERVAL_MIN = 10

    #MAX_FRAME = 144
    #FINAL_FRAMES_TO_TRY = range(7, 145, 6)
    #SMOOTHEN = 1

    #FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\3_Kinetics\4_Scrambled"

    #batch(FOLDER_PATH)


    #IMAGING_INTERVAL_MIN = 10

    #MAX_FRAME = 144
    #FINAL_FRAMES_TO_TRY = range(3, 145, 1)
    #SMOOTHEN = 1

    #FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\CAuris\Figures\3_Kinetics\2_FKS1Timecourse\Results\Mixed Population"

    #batch(FOLDER_PATH)

    #IMAGING_INTERVAL_MIN = 2

    #MAX_FRAME = 76
    #FINAL_FRAMES_TO_TRY = range(3, 77, 1)
    #SMOOTHEN = 1

    #FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\Experiments\Digital\20230404_CAuris_Inhibition_Control"

    #batch(FOLDER_PATH)

    IMAGING_INTERVAL_MIN = 2

    MAX_FRAME = 76
    FINAL_FRAMES_TO_TRY = [76]
    SMOOTHEN = 1

    FOLDER_PATH = r"C:\Users\Wyss User\OneDrive - Harvard University\Experiments\Digital\20230207_CAuris_CalCurve_11_GuidePrimerScreen\new analysis"

    batch(FOLDER_PATH)