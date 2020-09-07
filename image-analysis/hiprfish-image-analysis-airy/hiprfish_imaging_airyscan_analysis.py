
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import sys
import glob
import argparse
import javabridge
import bioformats
import numpy as np
import pandas as pd
from skimage.draw import line
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################

javabridge.start_vm(class_path=bioformats.JARS)

def cm_to_inches(length):
    return(length/2.54)

def get_stage_position(filename):
    image_xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(image_xml)
    pos_x = ome.image(0).Pixels.Plane().PositionX
    pos_y = ome.image(0).Pixels.Plane().PositionY
    return(np.array([pos_x, pos_y]))

def get_pixel_physical_size(filename):
    image_xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(image_xml)
    delta_x = ome.image(0).Pixels.PhysicalSizeX
    delta_y = ome.image(0).Pixels.PhysicalSizeY
    return(np.array([delta_x, delta_y]))

def save_segmentation(segmentation, sample):
    seg_color = color.label2rgb(segmentation, bg_label = 0, bg_color = (0,0,0))
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(seg_color)
    scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', box_color = 'white')
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    segfilename = sample + '_seg.pdf'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    np.save(sample + '_seg', segmentation)
    return

def save_identification(image_identification, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_identification)
    scalebar = ScaleBar(0.0675, 'um', frameon = True, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
    plt.gca().add_artist(scalebar)
    segfilename = sample + '_identification.pdf'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    return

def save_sum_images(image_final, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(8),cm_to_inches(8))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_final, cmap = 'inferno')
    segfilename = sample + '_sum.png'
    fig.savefig(segfilename, dpi = 1000)
    plt.close()
    return

def save_enhanced_images(image_final, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(4),cm_to_inches(4))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_final, cmap = 'inferno')
    plt.axis('off')
    segfilename = sample + '_enhanced.pdf'
    fig.savefig(segfilename, dpi = 300)
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
    structuring_element = disk(20)
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

def generate_stitched_images(sample):
    sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_11'
    subfield_list = [3, 5, 6, 7, 8, 1, 9, 4, 10]
    image_list = ['{}_{}_633_Out.czi'.format(sample, sf) for sf in subfield_list]
    images = [bioformats.load_image(f) for f in image_list]
    image_shape_y = images[0].shape[0]
    image_shape_x = images[0].shape[1]
    pixel_size = get_pixel_physical_size(image_list[0])
    position_list = np.stack([get_stage_position(f) for f in image_list])
    min_x = np.min(position_list[:,0])
    min_y = np.min(position_list[:,1])
    image_size_x = np.max(position_list[:,0]) - np.min(position_list[:,0])
    image_size_y = np.max(position_list[:,1]) - np.min(position_list[:,1])
    position_pixel_list = np.zeros(position_list.shape)
    for i in range(0, len(image_list)):
        position_pixel_list[i,0] = int(np.ceil((position_list[i,1] - min_y)/pixel_size[1]))
        position_pixel_list[i,1] = int(np.ceil((position_list[i,0] - min_x)/pixel_size[1]))
    delta_position_list = np.zeros(adjusted_position_list.shape)
    delta_position_list[1,0] = 25
    delta_position_list[3:,0] += 50
    adjusted_position_list = position_pixel_list + delta_position_list
    full_image = np.zeros((int(np.ceil(image_size_y/pixel_size[1])) + images[0].shape[0] + 100,int(np.ceil(image_size_x/pixel_size[0])) + images[0].shape[0] + 100))
    full_image_mask = np.zeros(full_image.shape)
    for i in range(len(image_list)):
        x_start = int(adjusted_position_list[i, 0] + 50)
        x_end = int(adjusted_position_list[i, 0] + image_shape_x + 50)
        y_start = int(adjusted_position_list[i, 1] + 50)
        y_end = int(adjusted_position_list[i, 1] + image_shape_y + 50)
        full_image[x_start:x_end, y_start:y_end] += images[i][:,:,0]
        full_image_mask[x_start:x_end, y_start:y_end] += 1

    full_image_mask[full_image_mask == 0] = 1
    full_image_scaled = full_image/full_image_mask
    full_image_scaled = full_image_scaled/np.max(full_image_scaled)
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5*full_image_scaled.shape[0]/full_image_scaled.shape[1]))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(full_image_scaled, cmap = 'inferno')
    scalebar = ScaleBar(0.01686, 'um', frameon = False, color = 'white', box_color = 'white')
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    full_image_filename = sample + '_stiched.pdf'
    fig.savefig(full_image_filename, dpi = 300)
    plt.close()
    np.save('{}_full.npy'.format(sample), full_image_scaled)
    segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(full_image_scaled.reshape(np.prod(full_image_scaled.shape), 1)).reshape(full_image_scaled.shape)
    seg_mask = segmentation != 0
    seg_mask_filtered = skimage.morphology.remove_small_objects(seg_mask, 2500)
    seg_mask_gaussian = skimage.filters.gaussian(seg_mask_filtered, 10)
    segmentation_gaussian = KMeans(n_clusters = 3, random_state = 0).fit_predict(seg_mask_gaussian.reshape(np.prod(full_image_scaled.shape), 1)).reshape(full_image_scaled.shape)
    seg_mask_2 = segmentation_gaussian != 0
    seg_mask_medial = skimage.morphology.medial_axis(seg_mask_2)
    medial_line_coordinates = np.where(seg_mask_medial == 1)
    arc = arc_length(medial_line_coordinates[0],medial_line_coordinates[1])
    start_dx = np.gradient(medial_line_coordinates[0][0:5])
    start_dy = np.gradient(medial_line_coordinates[1][0:5])
    cell_start_row = int(medial_line_coordinates[0][0] - 50*np.average(start_dx))
    cell_start_col = int(medial_line_coordinates[1][0] - 50*np.average(start_dy))
    end_dx = np.gradient(medial_line_coordinates[0][-5:])
    end_dy = np.gradient(medial_line_coordinates[1][-5:])
    cell_end_row = int(medial_line_coordinates[0][-1] + 50*np.average(end_dx))
    cell_end_col = int(medial_line_coordinates[1][-1] + 50*np.average(end_dy))
    rr_start, cc_start = line(medial_line_coordinates[0][0], medial_line_coordinates[1][0], cell_start_row, cell_start_col)
    rr_end, cc_end = line(medial_line_coordinates[0][-1], medial_line_coordinates[1][-1], cell_end_row, cell_end_col)
    seg_mask_medial[rr_start,cc_start] = True
    seg_mask_medial[rr_end,cc_end] = True
    medial_line_coordinates = np.where(seg_mask_medial == 1)
    medial_line = np.stack([medial_line_coordinates[0][0::100], medial_line_coordinates[1][0::100]], axis = 1)
    medial_line = np.append(medial_line, np.array([medial_line_coordinates[0][-1], medial_line_coordinates[1][-1]])[None,:], axis = 0)
    straightened_image = get_straightened_image(full_image_scaled*seg_mask_2, medial_line, 200)

    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(17.75),cm_to_inches(17.75*straightened_image.shape[0]/straightened_image.shape[1]))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(straightened_image, cmap = 'inferno')
    # scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
    # plt.gca().add_artist(scalebar)
    plt.axis('off')
    image_filename = '{}_straightened.pdf'.format(sample)
    plt.savefig(image_filename, dpi = 300)
    plt.close()
    intensity_profile = np.sum(straightened_image, axis = 0)
    intensity_profile = intensity_profile/np.max(intensity_profile)

    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(18),cm_to_inches(4))
    gs = GridSpec(2,1)
    ax = plt.subplot(gs[0,0])
    ax.imshow(straightened_image, cmap = 'inferno')
    scalebar = ScaleBar(0.01686, 'um', frameon = False, color = 'white', box_color = 'white', height_fraction = 0.05)
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    ax = plt.subplot(gs[1,0])
    ax.plot(0.016*np.arange(0,intensity_profile.shape[0]), intensity_profile, color = (0,0.5,1))
    plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
    plt.xlabel(r'Length [$\mu$m]', fontsize = 8, color = theme_color)
    plt.ylabel('Intensity', fontsize = 8, color = theme_color)
    ax.spines['left'].set_color(theme_color)
    ax.spines['bottom'].set_color(theme_color)
    ax.spines['right'].set_color(theme_color)
    ax.spines['top'].set_color(theme_color)
    ax.arrow(1876*0.016, 0.9, 0, -0.2, head_width = 0.5, head_length = 0.1, color = theme_color)
    ax.arrow(2772*0.016, 0.9, 0, -0.2, head_width = 0.5, head_length = 0.1, color = theme_color)
    plt.subplots_adjust(left = 0.05, bottom = 0.25, right = 0.98, top = 0.98)
    straightened_image_filename = '{}_straightened.pdf'.format(sample)
    fig.savefig(straightened_image_filename, dpi = 300, transparent = True)
    plt.close()
    zoom_image_filename_1 = sample + '_zoom_1.pdf'
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    plt.imshow(straightened_image[50:150,1830:1930], cmap = 'inferno')
    scalebar = ScaleBar(0.01686, 'um', frameon = False, color = 'white', box_color = 'white', height_fraction = 0.02)
    plt.axis('off')
    plt.gca().add_artist(scalebar)
    fig.savefig(zoom_image_filename_1, dpi = 300, transparent = True)
    plt.close()
    zoom_image_filename_2 = sample + '_zoom_2.pdf'
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5),cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    plt.imshow(straightened_image[50:150,2720:2820], cmap = 'inferno')
    scalebar = ScaleBar(0.01686, 'um', frameon = False, color = 'white', box_color = 'white', height_fraction = 0.02)
    plt.axis('off')
    plt.gca().add_artist(scalebar)
    fig.savefig(zoom_image_filename_2, dpi = 300, transparent = True)
    plt.close()
    return(full_image_scaled)

def save_airy_image(image, sample, subfield):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(3),cm_to_inches(3))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image, cmap = 'inferno')
    plt.axis('off')
    plt.text(50, 50, subfield, fontsize = 8, color = 'white')
    image_filename = '{}_intensity.pdf'.format(sample)
    fig.savefig(image_filename, dpi = 300)
    plt.close()
    return

def analyze_airy(sample):
    image_filename = '{}.czi'.format(sample)
    subfield = re.search('fov_.*_Out',sample).group(0).split('_')[2]
    image = bioformats.load_image(image_filename)
    save_airy_image(image[:,:,0], sample, subfield)
    return()

def main():
    parser = argparse.ArgumentParser('Mesure environmental microbial community spectral images')
    parser.add_argument('image_filename', type = str, help = 'Input folder containing spectral images')
    args = parser.parse_args()
    sample = re.sub('.czi', '', args.image_filename)
    analyze_airy(sample)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
