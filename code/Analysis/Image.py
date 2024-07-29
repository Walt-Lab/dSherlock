"""
functions for the image analysis of dSHERLOCK data

author: Anton Thieme <anton@thiemenet.de>
"""

import numpy as np
import skimage.io as skio
from skimage.filters import threshold_yen
from skimage import measure
import math
from skimage import transform
import pandas as pd
import matplotlib.pyplot as plt
import os

LOWER_AREABOUND = 80
UPPER_AREABOUND = 120
MAXIMUM_ALLOWED_MOVEMENT = 10

FQ_CHANNEL_INDEX = 0
ROX_CHANNEL_INDEX = 1


# give the path to the folder that contains the images you want to analyze
FOLDER_PATH = r""


"""
DESCRIPTION: Image processing pipeline, called by batch()

PRE: result folder created, path leads to tif file, path to field flatness correcting NORM_FACTORS.csv is correct
POST: number of wells that were trackable, reporter fluorescence intensity over time for these wells, and rox fluorescence intensity over time for these wells are returned,
result_folder contains image of the master mask, image of the masked reporter fluorescence intensity at t=0 and at t=max(t)
"""
def imageProcessing(path, result_folder):

    print('  ---LOADING IMAGES---')

    # load tif file
    tiff = skio.imread(path, plugin="tifffile")

    print('          done')

    # change order of dimensions in image to enable faster processing
    tiff = np.transpose(tiff, (1, 0, 2, 3))

    # load and apply field flatness correction factors to reporter channel
    flatness_factors = np.loadtxt(r'C:\Users\Wyss User\OneDrive - Harvard University\Experiments\Digital\20230303_FieldFlatness\NORM_FACTORS.csv', delimiter=',', dtype=float)
    tiff = [np.divide(tiff[0], flatness_factors),tiff[1]]

    print('   ---THRESHOLDING---')

    # set up datastructure for thresholding
    masks = {'mask': [], 'labels': [], 'props': []}
    img_count = len(tiff[ROX_CHANNEL_INDEX])

    # iterate through all timepoints (frames)
    for ii, imgRox in enumerate(tiff[ROX_CHANNEL_INDEX]):

        # find yen threshold and (large) range around it for rox image, range is required because of heterogeneous rox illumination
        # and might well be better in a different optical system. Probably space for performance improvement here.
        thresh = threshold_yen(imgRox)
        threshRange = np.arange(thresh-thresh*0.9, thresh+thresh*0.5, thresh*0.05)

        masks_temp = {'mask':[], 'labels':[], 'props':[], 'nOfWells':[]}

        # for each value in the range of thresholds
        for thresh in threshRange:

            # apply threshold
            mask = imgRox > thresh
            # get connected components
            labels = measure.label(mask)
            # measure properties of each connected component
            props = measure.regionprops(labels, imgRox)
            # find and discard connected components that are too small or too large (not microarray partitions)
            false_cc_coords = [cc.coords for cc in props if (cc.area < LOWER_AREABOUND) or (cc.area > UPPER_AREABOUND)]
            false_coords = [coords for cc in false_cc_coords for coords in cc]
            y = [coord[0] for coord in false_coords]
            x = [coord[1] for coord in false_coords]
            mask[y, x] = 0

            # save mask
            masks_temp['mask'].append(mask)

        # sum up masks generated with each threshold to get mask with all identifiable partitions, combination of all masks
        mask = np.sum(masks_temp['mask'], axis=0) > 0
        # get connected components
        labels = measure.label(mask)
        # measure properties of each connected component
        props = measure.regionprops(labels, imgRox)

        # save mask, connected components and their properties
        masks['mask'].append(mask)
        masks['labels'].append(labels)
        masks['props'].append(props)

        print('\r           {:.0f}%'.format(100*(ii+1)/img_count), end='')

    print('\n---IMAGE REGISTRATION---')

    # set up datastructure for image registration
    master_mask = masks['mask'][0]
    warped_images = [tiff[FQ_CHANNEL_INDEX][0]]
    warped_rox_images = [tiff[ROX_CHANNEL_INDEX][0]]
    T = np.eye(3,3)

    # iterate through all timepoints (frames)
    for ii in range(1,len(masks['props'])):

        # get centroids of all connected components (partitions) from current frame and previous frame
        centroids_prev = [(cc.centroid[1], cc.centroid[0]) for cc in masks['props'][ii-1]]
        centroids_this = [(cc.centroid[1], cc.centroid[0]) for cc in masks['props'][ii]]

        # get centroid of one partition in ~ middle of the chip from previous frame
        ind_center_prev = int(len(centroids_prev)/2)
        center_prev = np.asarray(centroids_prev[ind_center_prev])

        # get centroid of partition closes to that in current frame
        ind_center_this = centroids_this.index(min(centroids_this, key=lambda x: math.dist(x, center_prev)))
        center_this = np.asarray(centroids_this[ind_center_this])

        # if partitions are too far away from each other, assume that partition became untrackable between the two frames and try a different one
        while math.dist(center_prev, center_this) > MAXIMUM_ALLOWED_MOVEMENT:
            ind_center_prev += 1
            center_prev = np.asarray(centroids_prev[ind_center_prev])
            ind_center_this = centroids_this.index(min(centroids_this, key=lambda x: math.dist(x, center_prev)))
            center_this = np.asarray(centroids_this[ind_center_this])

        # get centroid of one partition in ~ upper half of the chip from previous frame
        ind_upper_prev = int(len(centroids_prev)/4)
        upper_prev = np.asarray(centroids_prev[ind_upper_prev])

        # get centroid of partition closes to that in current frame
        ind_upper_this = centroids_this.index(min(centroids_this, key=lambda x: math.dist(x, upper_prev)))
        upper_this = np.asarray(centroids_this[ind_upper_this])

        # if partitions are too far away from each other, assume that partition became untrackable between the two frames and try a different one
        while math.dist(upper_prev, upper_this) > MAXIMUM_ALLOWED_MOVEMENT:
            ind_upper_prev += 1
            upper_prev = np.asarray(centroids_prev[ind_upper_prev])
            ind_upper_this = centroids_this.index(min(centroids_this, key=lambda x: math.dist(x, upper_prev)))
            upper_this = np.asarray(centroids_this[ind_upper_this])

        # get centroid of one partition in ~ lower half of the chip from previous frame
        ind_lower_prev = int(3*len(centroids_prev)/4)
        lower_prev = np.asarray(centroids_prev[ind_lower_prev])

        # get centroid of partition closes to that in current frame
        ind_lower_this = centroids_this.index(min(centroids_this, key=lambda x: math.dist(x, lower_prev)))
        lower_this = np.asarray(centroids_this[ind_lower_this])

        # if partitions are too far away from each other, assume that partition became untrackable between the two frames and try a different one
        while math.dist(lower_prev, lower_this) > MAXIMUM_ALLOWED_MOVEMENT:
            ind_lower_prev += 1
            lower_prev = np.asarray(centroids_prev[ind_lower_prev])
            ind_lower_this = centroids_this.index(min(centroids_this, key=lambda x: math.dist(x, lower_prev)))
            lower_this = np.asarray(centroids_this[ind_lower_this])

        # estimate spatial shift between last and current frame and apply the inverse to correct for it
        T_temp = transform.estimate_transform('euclidean', np.array([upper_prev, center_prev, lower_prev]), np.array([upper_this, center_this, lower_this]))
        T = np.matmul(T, T_temp)
        warped_mask = transform.warp(masks['mask'][ii], T, preserve_range=True)

        # save shifted images for both channels
        warped_images.append(transform.warp(tiff[0][ii], T, preserve_range=True))
        warped_rox_images.append(transform.warp(tiff[ROX_CHANNEL_INDEX][ii], T, preserve_range=True))

        # build master mask that is all masks overlayed such that partitions that become untrackable are ignored from the start
        master_mask = master_mask*warped_mask

        print('\r           {:.0f}%'.format(100*(ii+1)/img_count), end='')

    # save an image of master mask into result folder
    plt.imshow(master_mask)
    plt.savefig(r"{}\{}".format(result_folder, 'mask'), dpi=1200)
    #plt.savefig(r"{}\{}".format(result_folder, 'mask'), format='svg', dpi=450, transparent=True)

    # create masked images from master mask and the shifted images in both channels
    masked_images = [warped_image*master_mask for warped_image in warped_images]
    masked_rox_images = [warped_rox_image * master_mask for warped_rox_image in warped_rox_images]

    # save images of masked reporter fluorescence intensity at t=0 and t=max(t)
    plt.imshow(masked_images[0])
    plt.savefig(r"{}\{}".format(result_folder, 'masked_image0'), dpi=1200)
    #plt.savefig(r"{}\{}".format(result_folder, 'masked_image0'), format='svg', dpi=450, transparent=True)
    plt.imshow(masked_images[len(masked_images)-1])
    plt.savefig(r"{}\{}{}".format(result_folder, 'masked_image', len(masked_images)-1), dpi=1200)
    #plt.savefig(r"{}\{}{}".format(result_folder, 'masked_image', len(masked_images) - 1), format='svg', dpi=450, transparent=True)

    print("\n  ---DATA EXTRACTION---")

    # get connected components and measure their properties in both channels
    labels = measure.label(master_mask)
    nOfWells = len(measure.regionprops(labels, masked_images[0]))
    props = [measure.regionprops(labels, masked_image) for masked_image in masked_images]
    props_rox = [measure.regionprops(labels, masked_rox_image) for masked_rox_image in masked_rox_images]

    # extract mean intensity and position for each partition at first timepoint in both of the channels
    timeseries = pd.DataFrame({'frame': [1]*len(props[0]), 'mean' : [cc.intensity_mean for cc in props[0]], 'label' : [cc.label for cc in props[0]], 'y' : [cc.centroid[0] for cc in props[0]], 'x' : [cc.centroid[1] for cc in props[0]]})
    timeseries_rox = pd.DataFrame({'frame': [1] * len(props_rox[0]), 'mean': [cc.intensity_mean for cc in props_rox[0]], 'label': [cc.label for cc in props_rox[0]]})


    coordinates = [cc.centroid for cc in props[0]]

    # extract mean intensity and position for each partition at following timepoints in both of the channels
    for ii in range(1,len(props)):

        timeseries = pd.concat([timeseries, pd.DataFrame({'frame': [ii+1]*len(props[ii]), 'mean' : [cc.intensity_mean for cc in props[ii]], 'label' : [cc.label for cc in props[ii]], 'y' : [cc.centroid[0] for cc in props[ii]], 'x' : [cc.centroid[1] for cc in props[ii]]})])
        timeseries_rox = pd.concat([timeseries_rox, pd.DataFrame({'frame': [ii+1]*len(props_rox[ii]), 'mean' : [cc.intensity_mean for cc in props_rox[ii]], 'label' : [cc.label for cc in props_rox[ii]], 'y' : [cc.centroid[0] for cc in props[ii]], 'x' : [cc.centroid[1] for cc in props[ii]]})])

        print('\r           {:.0f}%'.format(100*(ii+1)/img_count), end='')

    return(nOfWells, timeseries, timeseries_rox)


"""
DESCRIPTION: function to apply the image processing pipeline to all tif files in the folder specified by FOLDER_PATH

PRE: all tif files to be processed are in FOLDER_PATH
POST: all tif files in FOLDER_PATH have run through the image processing pipine imageProcessing(), result folders have been created in FOLDER_PATH,
result folders contain csv files with reporter fluorescence intensity over time for each tracked well and rox fluorescence intensity over time for each tracked well
"""
def batch():

    for file in os.listdir(FOLDER_PATH):

        if file.endswith(".tif"):

            # make result folder
            print("WORKING ON FILE: {}".format(file))
            result_folder = r"{}\Results_{}".format(FOLDER_PATH, file[0:file.find('.')])
            os.mkdir(result_folder)

            # call image processing pipeline and save the extracted mean intensities for each partition at each timepoint to csv files in the result folder
            nOfWells, timeseries, timeseries_rox = imageProcessing(r"{}\{}".format(FOLDER_PATH, file), result_folder)
            timeseries.to_csv(r"{}\{}".format(result_folder, 'timeseries.csv'))
            timeseries_rox.to_csv(r"{}\{}".format(result_folder, 'timeseries_rox.csv'))

            print("\nWells Tracked: {}".format(nOfWells))
            print("\n\n\n")


if __name__ == "__main__":

    batch()