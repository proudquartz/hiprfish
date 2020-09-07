import pandas as pd
import numpy as np
import glob
import argparse
import re
from matplotlib import pyplot as plt
import os
from ete3 import NCBITaxa
from matplotlib.gridspec import GridSpec
import skimage
from skimage import measure, segmentation
import astropy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.future import graph
import networkx as nx
import matplotlib
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

ncbi = NCBITaxa()

def cm_to_inches(length):
    return(length/2.54)

def pair_correlation(distance, dr, nr, A):
    n_cell_total = distance.shape[0]
    g = np.zeros(nr)
    for k in range(1, nr+1):
        total_count = 0
        for i in range(distance.shape[0]):
            total_count += distance[i, (distance[i, :] > (k-1)*dr) & (distance[i, :] < k*dr)].shape[0]
        g[k-1] = A*total_count/(n_cell_total*n_cell_total*2*np.pi*k*dr*dr)
    return(g)


sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0557/03_21_2019_DSGN0557_B_205_colon_1_fov_1'
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
image_seg = np.load('{}_seg.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))

image_identification_filtered = image_identification.copy()
for i in range(cell_info.shape[0]):
    x = cell_info.iloc[i, 70]
    y = cell_info.iloc[i, 71]
    cell_label = cell_info.iloc[i, 69]
    cell_area = cell_info.iloc[i, 76]
    if (760*y + 2000*x - 1980*2000 > 0):
        image_identification_filtered[image_seg == cell_label] = [0.5,0.5,0.5]
    elif (cell_area > 1000) or (cell_area < 20):
        image_identification_filtered[image_seg == cell_label] = [0,0,0]

hespellia_cells = cell_info.iloc[cell_info[67].values == '0000110', :]
hespellia_epithelial_distance = np.zeros(hespellia_cells.shape[0])
for i in range(hespellia_cells.shape[0]):
    x = hespellia_cells.iloc[i, 70]
    y = hespellia_cells.iloc[i, 71]
    hespellia_epithelial_distance[i] = np.abs(760*y + 2000*x - 1980*2000)/np.sqrt(760**2+2000**2)

bacteroides_cells = cell_info.iloc[cell_info[67].values == '1000001', :]
bacteroides_epithelial_distance = np.zeros(bacteroides_cells.shape[0])
for i in range(bacteroides_cells.shape[0]):
    x = bacteroides_cells.iloc[i, 70]
    y = bacteroides_cells.iloc[i, 71]
    bacteroides_epithelial_distance[i] = np.abs(760*y + 2000*x - 1980*2000)/np.sqrt(760**2+2000**2)

bacteroides_cells = cell_info.iloc[cell_info[67].values == '1000001', :]
bacteroides_distance = np.zeros((bacteroides_cells.shape[0], bacteroides_cells.shape[0]))
for i in range(bacteroides_cells.shape[0]):
    for j in range(bacteroides_cells.shape[0]):
        x1 = bacteroides_cells.iloc[i, 70]
        y1 = bacteroides_cells.iloc[i, 71]
        x2 = bacteroides_cells.iloc[j, 70]
        y2 = bacteroides_cells.iloc[j, 71]
        bacteroides_distance[i,j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

hespellia_cells = cell_info.iloc[cell_info[67].values == '0000110', :]
hespellia_distance = np.zeros((hespellia_cells.shape[0], hespellia_cells.shape[0]))
for i in range(hespellia_cells.shape[0]):
    for j in range(hespellia_cells.shape[0]):
        x1 = hespellia_cells.iloc[i, 70]
        y1 = hespellia_cells.iloc[i, 71]
        x2 = hespellia_cells.iloc[j, 70]
        y2 = hespellia_cells.iloc[j, 71]
        hespellia_distance[i,j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

dr = 2
nr = 200
g_bacteroides = pair_correlation(bacteroides_distance, dr, nr, image_seg.shape[0]*image_seg.shape[0])
g_hespellia = pair_correlation(hespellia_distance, dr, nr, image_seg.shape[0]*image_seg.shape[0])

fig = plt.figure()
color_list = ['darkorange', 'dodgerblue']
# color_list = ['maroon', 'darkorange', 'dodgerblue']
fig.set_size_inches(cm_to_inches(3), cm_to_inches(1.75))
# plt.plot((np.arange(0, dr*nr, dr)/macellibacteroides_cells[73].mean()), g_macellibacteroides, color = color_list[0], alpha = 0.8)
plt.plot((np.arange(0, dr*nr, dr)/hespellia_cells[73].mean()), g_hespellia, color = color_list[0], alpha = 0.8)
plt.plot((np.arange(0, dr*nr, dr)/bacteroides_cells[73].mean()), g_bacteroides, color = color_list[1], alpha = 0.8)
plt.xlabel(r'$\it{r}$/$a_{minor}$ [-]', fontsize = 5, color = 'black', labelpad = 0)
plt.ylabel('Pair correlation', fontsize = 5, color = 'black', labelpad = 0)
plt.tick_params(direction = 'in', labelsize = 5, colors = 'black', pad = 2, width = 0.5)
plt.subplots_adjust(left = 0.14, right = 0.95, bottom = 0.25, top = 0.9)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.axes().spines['bottom'].set_color('black')
plt.axes().spines['left'].set_color('black')
l = plt.legend([r'$\it{Hespellia}$', r'$\it{Bacteroides}$'], fontsize = 5, loc = 2, frameon = False, bbox_to_anchor=(0.7, 1.05))
# l = plt.legend(['Macellibacteroides', 'Hespellia', 'Bacteroides'], fontsize = 8, loc = 2, frameon = False, bbox_to_anchor=(0.8, 1.05))
for i in range(2):
    l.get_texts()[i].set_color(color_list[i])
    l.get_texts()[i].set_ha('right')
    l.get_lines()[i].set_alpha(0)

plt.xlim(0,22)
plt.savefig('{}_gr.pdf'.format(sample), dpi = 300, transparent = True)
plt.close()

fig = plt.figure()
fig.set_size_inches(cm_to_inches(3), cm_to_inches(1.75))
# plt.hist(macellibacteroides_epithelial_distance*0.07, bins = 30, density = True, histtype = 'step', color = color_list[0], alpha = 0.8)
plt.hist(hespellia_epithelial_distance*0.07, bins = 30, density = True, histtype = 'step', color = color_list[0], alpha = 0.8)
plt.hist(bacteroides_epithelial_distance*0.07, bins = 30, density = True, histtype = 'step', color = color_list[1], alpha = 0.8)
plt.xlabel(r'Distance [$\mu$m]', fontsize = 5, color = 'black', labelpad = 0)
plt.ylabel('Density [$\mu$m$^{-1}$]', fontsize = 5, color = 'black', labelpad = 0)
plt.tick_params(labelsize = 5, direction = 'in', colors = 'black', pad = 2, width = 0.5)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.ylim(0,0.028)
plt.yticks([0,0.01,0.02], [0,1,2])
l = plt.legend([r'$\it{Hespellia}$', r'$\it{Bacteroides}$'], fontsize = 5, loc = 2, frameon = False, bbox_to_anchor=(0.7, 1.05))
for i in range(2):
    l.get_texts()[i].set_color(color_list[i])
    l.get_texts()[i].set_ha('right')
    l.get_patches()[i].set_alpha(0)

plt.axes().spines['bottom'].set_color('black')
plt.axes().spines['left'].set_color('black')
plt.text(-2,0.025,r'$\times 10^{-2}$', fontsize = 5)
plt.ylim(0,0.03)
plt.subplots_adjust(left = 0.14, right = 0.95, bottom = 0.28, top = 0.9)
plt.savefig('{}_epithelial_distance.pdf'.format(sample), dpi = 300, transparent = True)
plt.close()



input_dir = '{}/09_08_2019_DSGN0567/206_colon_1'.format(data_folder)
image_registered = np.load('{}/09_08_2019_DSGN0567_206_colon_1_tile_scan_fov_8_registered.npy'.format(input_dir))

debris = image_seg*image_epithelial_area
image_identification_filtered = image_identification.copy()
image_identification_filtered[debris > 0] = [0.5,0.5,0.5]

for i in range(cell_info.shape[0]):
    cell_label = cell_info.loc[i, 'label']
    cell_area = cell_info.loc[i, 'area']
    if (cell_area > 10000):
        image_identification_filtered[segmentation == cell_label] = [0.5,0.5,0.5]

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(8.5),cm_to_inches(8.5*3800/7400))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.transpose(image_identification_filtered, axes = (1,0,2)))
scalebar = ScaleBar(0.0675, 'um', frameon = True, color = 'white', box_color = 'black', box_alpha = 0.65, location = 4)
plt.gca().add_artist(scalebar)
segfilename = sample + '_identification_filtered.pdf'
fig.savefig(segfilename, dpi = 1000)
plt.close()




def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('input_folder', type = str, help = 'Input folder containing images of biological samples')
    parser.add_argument('-p', '--probe_design_filename', dest = 'probe_design_filename', type = str, nargs = '*', help = 'Input folder containing images of biological samples')
    parser.add_argument('-f', '--sample_prefix', dest = 'sample_prefix', type = str, nargs = '*', help = 'Input folder containing images of biological samples')
    args = parser.parse_args()

    return

if __name__ == '__main__':
    main()
