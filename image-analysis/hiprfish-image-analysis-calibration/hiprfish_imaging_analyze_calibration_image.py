import re
import glob
import argparse
import javabridge
import bioformats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

javabridge.start_vm(class_path=bioformats.JARS)

def cm_to_inches(length):
    return(length/2.54)

def load_tslice(filename, t_index):
    image = bioformats.load_image(filename, t = t_index)
    return(image)

def get_t_range(filename):
    xml = bioformats.get_omexml_metadata(filename)
    ome = bioformats.OMEXML(xml)
    t_range = ome.image(0).Pixels.get_SizeT()
    return(t_range)

def load_image_tstack(filename):
    t_range = get_t_range(filename)
    image = np.stack([load_tslice(filename, t) for t in range(0, t_range)], axis = 3)
    return(image)

def save_calibration_image(image, sample):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(image_final, cmap = 'inferno')
    scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black')
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    segfilename = '{}_spectral_average.pdf'
    fig.savefig(segfilename, dpi = 300, transparent = True)
    plt.close()
    return

def fit_calibration_image(calibration_integrated_norm, calibration_filename):
    x_center_initial = 1300
    y_center_initial = 800
    sigma_initial = 6300
    magnitude_initial = 0.875
    xx, yy = np.meshgrid(np.arange(0,2000), np.arange(0,2000))
    cost = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            x_center = x_center_initial - 100 + i*10
            y_center = y_center_initial - 100 + j*10
            calibration_calculated = magnitude*np.exp(-((xx-x_center)**2 + (yy-y_center)**2)/(sigma**2))
            residuals = calibration_calculated - calibration_integrated_norm
            cost[i,j] = np.sum(residuals**2)
    center_index = np.unravel_index(np.argmin(cost), cost.shape)
    x_center = x_center_initial - 100 + center_index[0]*10
    y_center = y_center_initial - 100 + center_index[1]*10
    cost = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            sigma = sigma_initial - 100 + i*10
            magnitude = magnitude_initial - 0.01 + j*0.001
            calibration_calculated = magnitude*np.exp(-((xx-x_center)**2 + (yy-y_center)**2)/(sigma**2))
            residuals = calibration_calculated - calibration_integrated_norm
            cost[i,j] = np.sum(residuals**2)
    center_index = np.unravel_index(np.argmin(cost), cost.shape)
    sigma = sigma_initial - 100 + center_index[0]*10
    magnitude = magnitude_initial - 0.01 + j*0.001
    calibration_calculated = magnitude*np.exp(-((xx-x_center)**2 + (yy-y_center)**2)/(sigma**2))
    residuals = calibration_calculated - calibration_integrated_norm
    calibration_calculated_norm = calibration_calculated/np.max(calibration_calculated)
    calibration_calculated_filename = re.sub('.czi', '_calculated.npy', calibration_filename)
    np.save(calibration_calculated_filename, calibration_calculated_norm)
    return

def measure_calibration_image(calibration_filename):
    calibration = load_image_tstack(calibration_filename)
    calibration_average = np.average(calibration, axis = 3)
    calibration_integrated = np.average(calibration_average, axis = 2)
    calibration_integrated_norm = calibration_integrated/np.max(calibration_integrated)
    calibration_np_filename = re.sub('.czi', '.npy', calibration_filename)
    np.save(calibration_np_filename, calibration_integrated)
    return

def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('-c', '--caliobration_filename', dest = 'calibration_filename', type = str, default = '', help = 'Input folder containing images of biological samples')
    args = parser.parse_args()
    measure_calibration_image(args.calibration_filename)

if __name__ == '__main__':
    main()

javabridge.kill_vm()
