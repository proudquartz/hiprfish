import os
import re
import sys
import glob
import joblib
import skimage
import argparse
import numpy as np
import pandas as pd
from skimage import color
from ete3 import NCBITaxa
import bioformats
import javabridge
from skimage import measure
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import filters
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from neighbor2d import line_profile_2d_v2
from matplotlib import collections
from sklearn.cluster import KMeans
from skimage.morphology import disk
from skimage.draw import line
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline


ncbi = NCBITaxa()
javabridge.start_vm(class_path=bioformats.JARS)

def cm_to_inches(x):
    return(x/2.54)

def save_lachno_image(image_lachno, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(4),cm_to_inches(4))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_lachno)
    # scalebar = ScaleBar(0.0675, 'um', frameon = True, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
    # plt.gca().add_artist(scalebar)
    image_lachno_filename = '{}_lachno.pdf'.format(sample)
    fig.savefig(image_lachno_filename, dpi = 300)
    plt.close()
    return

def save_straightened_image(straightened_image, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(straightened_image.shape[1]/straightened_image.shape[0]*cm_to_inches(1),cm_to_inches(1))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(straightened_image)
    # scalebar = ScaleBar(0.0675, 'um', frameon = True, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
    # plt.gca().add_artist(scalebar)
    image_filename = '{}.pdf'.format(sample)
    fig.savefig(image_filename, dpi = 300)
    plt.close()
    return

def get_interpolated_pixel_value(image, x, y):
    image_neighborhood = np.zeros((2,2))
    x = [int(np.floor(x)),int(np.ceil(x)),np.mod(x,1)]
    y = [int(np.floor(y)),int(np.ceil(y)),np.mod(y,1)]
    if x[1] > 0 and x[1] < image.shape[0] and y[1] > 0 and y[1] < image.shape[1] and x[0] > 0 and x[0] < image.shape[0] and y[0] > 0 and y[0] < image.shape[1]:
        image_neighborhood[0,0] = image[x[0],y[0]]*(1-x[2])*(1-y[2])
        image_neighborhood[1,0] = image[x[0],y[1]]*(1-x[2])*(0+y[2])
        image_neighborhood[0,1] = image[x[1],y[0]]*(0+x[2])*(1-y[2])
        image_neighborhood[1,1] = image[x[1],y[1]]*(0+x[2])*(0+y[2])
        interpolated_value = np.sum(image_neighborhood)
    else:
        interpolated_value = 0
    return(interpolated_value)

def interpolate_spline(x, y, d):
    interpolated_x = []
    segment_distances = np.sqrt(np.diff(x)**2+np.diff(y)**2)
    for ct in range(x.shape[0]-1):
        new_x = np.arange(x[ct],x[ct+1],1/(np.round(segment_distances[ct]/d)))
        interpolated_x.append(new_x)
    return(np.concatenate(interpolated_x))

def fit_spline(x,y):
    spline = InterpolatedUnivariateSpline(x,y)
    x_isp = interpolate_spline(x,y,0.5)
    y_isp = spline(x_isp)
    l = 0
    x_spl = np.zeros(x_isp.shape[0])
    y_spl = np.zeros(y_isp.shape[0])
    x_spl[0] = x[0]
    y_spl[0] = y[0]
    ptw = 0
    for ct in range(1, x_isp.shape[0]):
        dx = x_isp[ct] - x_isp[ct - 1]
        dy = y_isp[ct] - y_isp[ct - 1]
        d = np.sqrt(dx**2 + dy**2)
        l = l + d
        overshoot = l - (ptw + 1)
        if overshoot > 0:
            ptw = ptw + 1
            frac = overshoot/d
            x_spl[ptw] = x_isp[ct] - frac*dx
            y_spl[ptw] = y_isp[ct] - frac*dy
    x_spl = x_spl[0:ptw]
    y_spl = y_spl[0:ptw]
    return(x_spl, y_spl)

def get_medial_line(cell_seg_mask, g, n, delta):
    seg_mask_gaussian = skimage.filters.gaussian(cell_seg_mask, g)
    seg_mask_gaussian = (seg_mask_gaussian-np.min(seg_mask_gaussian))/(np.max(seg_mask_gaussian)-np.min(seg_mask_gaussian))
    segmentation_gaussian = KMeans(n_clusters = 3, random_state = 0).fit_predict(seg_mask_gaussian.reshape(np.prod(cell_seg_mask.shape), 1)).reshape(cell_seg_mask.shape)
    intensity = [np.average((segmentation_gaussian == i)*seg_mask_gaussian) for i in range(3)]
    seg_mask_2 = segmentation_gaussian == np.argmax(intensity)
    structuring_element = disk(5)
    seg_mask_2 = skimage.morphology.binary_dilation(seg_mask_2, selem = structuring_element)
    seg_mask_medial = skimage.morphology.medial_axis(seg_mask_2)
    medial_line_coordinates = np.where(seg_mask_medial == 1)
    medial_line_coordinates_sorted = []
    if medial_line_coordinates[1][0] < medial_line_coordinates[1][-1]:
        medial_line_coordinates_sorted.append(np.flip(medial_line_coordinates[0], axis = 0))
        medial_line_coordinates_sorted.append(np.flip(medial_line_coordinates[1], axis = 0))
    else:
        medial_line_coordinates_sorted = medial_line_coordinates
    start_dx = np.gradient(medial_line_coordinates_sorted[0][0:5])
    start_dy = np.gradient(medial_line_coordinates_sorted[1][0:5])
    start_dx = np.average(start_dx)
    start_dy = np.average(start_dy)
    start_dx_norm = start_dx/np.sqrt(start_dx**2 + start_dy**2)
    start_dy_norm = start_dy/np.sqrt(start_dx**2 + start_dy**2)
    cell_start_row = int(medial_line_coordinates_sorted[0][0] - delta*start_dx_norm)
    if cell_start_row < 0:
        cell_start_row = 0
    elif cell_start_row > cell_seg_mask.shape[1]-1:
        cell_start_row = cell_seg_mask.shape[1]-1
    cell_start_col = int(medial_line_coordinates_sorted[1][0] - delta*start_dy_norm)
    if cell_start_col < 0:
        cell_start_col = 0
    elif cell_start_col > cell_seg_mask.shape[1]-1:
        cell_start_col = cell_seg_mask.shape[1]-1
    end_dx = np.gradient(medial_line_coordinates_sorted[0][-5:])
    end_dy = np.gradient(medial_line_coordinates_sorted[1][-5:])
    end_dx = np.average(end_dx)
    end_dy = np.average(end_dy)
    end_dx_norm = end_dx/np.sqrt(end_dx**2 + end_dy**2)
    end_dy_norm = end_dy/np.sqrt(end_dx**2 + end_dy**2)
    cell_end_row = int(medial_line_coordinates_sorted[0][-1] + delta*end_dx_norm)
    if cell_end_row < 0:
        cell_end_row = 0
    elif cell_end_row > cell_seg_mask.shape[1]-1:
        cell_end_row = cell_seg_mask.shape[1]-1
    cell_end_col = int(medial_line_coordinates_sorted[1][-1] + delta*end_dy_norm)
    if cell_end_col < 0:
        cell_end_col = 0
    elif cell_end_col > cell_seg_mask.shape[1]-1:
        cell_end_col = cell_seg_mask.shape[1]-1
    rr_start, cc_start = line(medial_line_coordinates_sorted[0][0], medial_line_coordinates_sorted[1][0], cell_start_row, cell_start_col)
    rr_end, cc_end = line(medial_line_coordinates_sorted[0][-1], medial_line_coordinates_sorted[1][-1], cell_end_row, cell_end_col)
    seg_mask_medial[rr_start,cc_start] = True
    seg_mask_medial[rr_end,cc_end] = True
    medial_line_coordinates = np.where(seg_mask_medial == 1)
    medial_line = np.stack([medial_line_coordinates[0][0::n], medial_line_coordinates[1][0::n]], axis = 1)
    if (medial_line_coordinates[0][-1] != medial_line[-1,0]) and (medial_line_coordinates[1][-1] != medial_line[-1,1]):
        medial_line = np.append(medial_line, np.array([medial_line_coordinates[0][-1], medial_line_coordinates[1][-1]])[None,:], axis = 0)
    return(medial_line, seg_mask_2)

def get_straightened_image(image, medial_line, width):
    try:
        x, y = fit_spline(medial_line[:,0], medial_line[:,1])
        straigtened_image = np.zeros((width, x.shape[0]))
        for ct in range(x.shape[0]):
            if ct == 0:
                dx = x[1] - x[0]
                dy = y[0] - y[1]
            else:
                dx = x[ct] - x[ct-1]
                dy = y[ct-1] - y[ct]
            l = np.sqrt(dx**2+dy**2)
            dx = dx/l
            dy = dy/l
            xStart = x[ct]-(dy*width)/2
            yStart = y[ct]-(dx*width)/2
            for ct2 in range(width):
                straigtened_image[ct2,ct] = get_interpolated_pixel_value(image, xStart, yStart)
                xStart = xStart+dy
                yStart = yStart+dx
    except:
        center_row = int(np.average(medial_line[:,0]))
        col_start = np.min(medial_line[:,1])
        col_end = np.max(medial_line[:,1])
        straigtened_image = image[int(center_row - width/2):int(center_row + width/2),col_start-50:col_end+50]
    return(straigtened_image)

def arc_length(x, y):
    npts = len(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)
    return(arc)

data_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging'
image_tab_filename = '{}/images_table_microbiome_2.csv'.format(data_folder)
image_tab = pd.read_csv(image_tab_filename)
taxa_barcode_sciname = pd.read_csv('/workdir/hs673/Runs/V1/Samples/HIPRFISH_7/simulation/DSGN0673/DSGN0673_primerset_C_barcode_selection_MostSimple_taxa_barcode_sciname.csv', dtype = {'cell_barcode':str})
lachno_barcode = taxa_barcode_sciname.loc[taxa_barcode_sciname.sci_name == 'Lachnoanaerobaculum', 'cell_barcode'].values[0]

def generate_lachno_images(data_folder, image_tab):
    lachno_table = []
    for i in range(image_tab.shape[0]):
        sample = image_tab.loc[i,'SAMPLE']
        image_name = image_tab.loc[i,'IMAGES']
        cell_info = pd.read_csv('{}/{}/{}_cell_information_consensus_filtered.csv'.format(data_folder, sample, image_name), dtype = {'cell_barcode':str})
        lachno_cell_info = cell_info.loc[cell_info.cell_barcode.values == lachno_barcode,:].reset_index().drop(columns = 'index')
        image_seg = np.load('{}/{}/{}_seg.npy'.format(data_folder, sample, image_name))
        image_registered = np.load('{}/{}/{}_registered.npy'.format(data_folder, sample, image_name))
        image_registered_sum = np.sum(image_registered, axis = 2)
        image_lachno_seg = np.zeros(image_seg.shape)
        if not lachno_cell_info.empty:
            print(i)
            image_lachno = np.zeros((image_seg.shape[0], image_seg.shape[1], 3))
            for k in range(lachno_cell_info.shape[0]):
                try:
                    cell_label = lachno_cell_info.loc[k, 'label']
                    image_lachno[image_seg == cell_label] = (0,0.25,1)
                    image_lachno_seg[image_seg == cell_label] = 1
                    cell_seg_mask = image_lachno_seg == 1
                    medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 5, 5, 50)
                    seg_mask_2_be = skimage.morphology.binary_erosion(seg_mask_2, disk(3))
                    seg_mask_2_gaussian = skimage.filters.gaussian(seg_mask_2_be, 3)
                    straightened_image = get_straightened_image(image_registered_sum*seg_mask_2_gaussian, medial_line, 50)
                    save_straightened_image(straightened_image, '{}/{}/{}_cell_{}'.format(data_folder, sample, image_name, k))
                except:
                    pass
            save_lachno_image(image_lachno, '{}/{}/{}'.format(data_folder, sample, image_name))

#####

def main():
    parser = argparse.ArgumentParser('Classify single cell spectra')
    parser.add_argument('-c', '--cell_info', dest = 'cell_info_filename', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-p', '--probe_design_filename', dest = 'probe_design_filename', type = str, default = '', help = 'Input normalized single cell spectra filename')
    args = parser.parse_args()
    sample = re.sub('_cell_information.csv', '', args.cell_info_filename)
    segmentation_filename = '{}_seg.npy'.format(sample)
    segmentation = np.load(segmentation_filename)
    cell_info = pd.read_csv(args.cell_info_filename, dtype = {'cell_barcode':str})
    taxon_lookup = get_taxon_lookup(args.probe_design_filename)
    taxa_barcode_sciname = get_taxa_barcode_sciname(args.probe_design_filename)
    cell_info_filtered = analyze_cell_info(args.cell_info_filename, taxa_barcode_sciname)
    generate_identification_image(segmentation, cell_info_filtered, sample, taxon_lookup)
    return


if __name__ == '__main__':
    main()
