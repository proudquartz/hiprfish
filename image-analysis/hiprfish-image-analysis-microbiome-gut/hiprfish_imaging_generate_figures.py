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

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0549/03_15_2019_DSGN0549_C_fov_1'
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
image_seg = np.load('{}_seg.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))

fig = plt.figure()
fig.set_size_inches(cm_to_inches(5.5), cm_to_inches(5.5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, font_properties = {'size': 8})
plt.gca().add_artist(scalebar)
plt.savefig('{}_identification.pdf'.format(sample), dpi = 300)
plt.close()

cell_id = image_seg[1620, 830]
cell_barcode =  cell_info.iloc[cell_info[69].values == cell_id, 67]
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
# plt.xlabel('Channel', fontsize = 8, labelpad = 0)
# plt.ylabel('Intensity', fontsize = 8, labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.35)
plt.savefig('{}_cell_1_spectrum.pdf'.format(sample), dpi = 300)

cell_id = image_seg[987, 868]
cell_barcode =  cell_info.iloc[cell_info[69].values == cell_id, 67]
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
# plt.xlabel('Channel', fontsize = 8, labelpad = 0)
# plt.ylabel('Intensity', fontsize = 8, labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.35)
plt.savefig('{}_cell_2_spectrum.pdf'.format(sample), dpi = 300)

cell_id = image_seg[323, 582]
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
# plt.xlabel('Channel', fontsize = 8, labelpad = 0)
# plt.ylabel('Intensity', fontsize = 8, labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.35)
plt.savefig('{}_cell_3_spectrum.pdf'.format(sample), dpi = 300)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[1500:1700, 700:900])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.axis('off')
plt.savefig('{}_cell_1_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[900:1100, 800:1000])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.axis('off')
plt.savefig('{}_cell_2_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[200:400, 500:700])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.axis('off')
plt.savefig('{}_cell_3_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

cell_id = image_seg[1928, 739]
cell_barcode = cell_info.iloc[cell_info[69].values == cell_id, 67].values
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlabel('Channel', fontsize = 8, labelpad = 0)
plt.ylabel('Intensity', fontsize = 8, labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.35)
plt.savefig('{}_prevotellamassila_spectrum.pdf'.format(sample), dpi = 300)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[1800:2000, 650:850])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.axis('off')
plt.savefig('{}_prevotellamassila_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()


fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black')
plt.gca().add_artist(scalebar)
plt.savefig('{}_identification.pdf'.format(sample), dpi = 300)
plt.close()


lautropia_cells = cell_info.iloc[cell_info[67].values == '1100000']
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3), cm_to_inches(3))
plt.plot(lautropia_cells[72].values*0.07, lautropia_cells[73].values*0.07, 'o', color = (0, 1, 0.889), alpha = 0.4, markersize = 2)
plt.xlabel('Semi-major Axis Length [um]', fontsize = 4)
plt.ylabel('Semi-minor Axis Length [um]', fontsize = 4)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(labelsize = 4, direction = 'in', length = 2)
plt.subplots_adjust(left = 0.3, right = 0.95, bottom = 0.3, top = 0.95)
plt.savefig('{}_lautropia_cell_size_scatter.pdf'.format(sample), dpi = 300)
plt.close()

distance = np.zeros((lautropia_cells.shape[0], lautropia_cells.shape[0]))
for i in range(lautropia_cells.shape[0]):
    for j in range(lautropia_cells.shape[0]):
        x1 = lautropia_cells.iloc[i, 70]
        x2 = lautropia_cells.iloc[j, 70]
        y1 = lautropia_cells.iloc[i, 71]
        y2 = lautropia_cells.iloc[j, 71]
        distance[i,j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(3), cm_to_inches(3))
plt.hist(distance[np.triu_indices(428)]*0.07, density = True, bins = 100, color = (0, 1, 0.889))
plt.xlabel('Intracellular distance [um]', fontsize = 4)
plt.ylabel('Frequency', fontsize = 4)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tight_layout()
plt.tick_params(labelsize = 4, direction = 'in', length = 2)
plt.subplots_adjust(left = 0.3, right = 0.95, bottom = 0.3, top = 0.95)
plt.savefig('{}_lautropia_cell_distance_histogram.pdf'.format(sample), dpi = 300)
plt.close()

g = pair_correlation(distance, 4, 500, image_seg.shape[0]*image_seg.shape[0])
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3), cm_to_inches(3))
plt.plot(np.arange(0, 2000, 4)*0.07, g, color = 'orange')
plt.xlabel(r'$\it{r}$ [$\mu$m]', fontsize = 6)
plt.ylabel(r'$g(r)$', fontsize = 6)
plt.tick_params(direction = 'in', labelsize = 6)
plt.subplots_adjust(left = 0.3, right = 1, bottom = 0.3, top = 1)
plt.xlim(0,3)
plt.savefig('{}_gr.pdf'.format(sample), dpi = 300, transparent = True)
plt.close()


phocaeicola_cells = cell_info.iloc[cell_info[67].values == '0011000']
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3), cm_to_inches(3))
plt.plot(phocaeicola_cells[72].values*0.07, phocaeicola_cells[73].values*0.07, 'o', color = (0.333, 1, 0), alpha = 0.4, markersize = 2)
plt.xlabel('Semi-major Axis Length [um]', fontsize = 4)
plt.ylabel('Semi-minor Axis Length [um]', fontsize = 4)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tight_layout()
plt.tick_params(labelsize = 4, direction = 'in', length = 2)
plt.subplots_adjust(left = 0.3, right = 0.95, bottom = 0.3, top = 0.95)
plt.savefig('{}_phocaeicola_size_scatter.pdf'.format(sample), dpi = 300)

distance = np.zeros((phocaeicola_cells.shape[0], phocaeicola_cells.shape[0]))
for i in range(phocaeicola_cells.shape[0]):
    for j in range(phocaeicola_cells.shape[0]):
        x1 = phocaeicola_cells.iloc[i, 70]
        x2 = phocaeicola_cells.iloc[j, 70]
        y1 = phocaeicola_cells.iloc[i, 71]
        y2 = phocaeicola_cells.iloc[j, 71]
        distance[i,j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(3), cm_to_inches(3))
plt.hist(distance[np.triu_indices(367)]*0.07, density = True, bins = 100, color = (0.333, 1, 0))
plt.xlabel('Intracellular distance [um]', fontsize = 4)
plt.ylabel('Frequency', fontsize = 4)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tight_layout()
plt.tick_params(labelsize = 4, direction = 'in', length = 2)
plt.subplots_adjust(left = 0.3, right = 0.95, bottom = 0.3, top = 0.95)
plt.savefig('{}_phocaeicola_cell_distance_histogram.pdf'.format(sample), dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0549/03_15_2019_DSGN0549_C_fov_2'
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
image_seg = np.load('{}_seg.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_registered = np.load('{}_registered.npy'.format(sample))
image_seg_centroid = np.zeros(image_seg.shape)
for i in range(cell_info.shape[0]):
    cx = int(cell_info.iloc[i, 70])
    cy = int(cell_info.iloc[i, 71])
    if cell_info.iloc[i, 67] == '1000101':
        image_seg_centroid[cx, cy] = 1

test = np.zeros(image_seg.shape)
for i in range(2000):
    x = int(np.random.random()*2000)
    y = int(np.random.random()*2000)
    test[x,y] = 1
test_cells = skimage.measure.regionprops(skimage.measure.label(test))
test_cell_info = pd.DataFrame(np.zeros((len(test_cells), 2)))
for i in range(len(test_cells)):
    test_cell_info.iloc[i, 0] = test_cells[i].centroid[0]
    test_cell_info.iloc[i, 1] = test_cells[i].centroid[1]
test_distance = np.zeros((test_cell_info.shape[0], test_cell_info.shape[0]))
for i in range(test_cell_info.shape[0]):
    for j in range(test_cell_info.shape[0]):
        x1 = test_cell_info.iloc[i, 0]
        y1 = test_cell_info.iloc[i, 1]
        x2 = test_cell_info.iloc[j, 0]
        y2 = test_cell_info.iloc[j, 1]
        test_distance[i,j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


rothia_cells = cell_info.iloc[cell_info[67].values == '0001101', :]
distance = np.zeros((rothia_cells.shape[0], rothia_cells.shape[0]))
for i in range(rothia_cells.shape[0]):
    for j in range(rothia_cells.shape[0]):
        x1 = rothia_cells.iloc[i, 70]
        y1 = rothia_cells.iloc[i, 71]
        x2 = rothia_cells.iloc[j, 70]
        y2 = rothia_cells.iloc[j, 71]
        distance[i,j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

image_seg_rothia = np.zeros(image_seg.shape)
for i in range(cell_info.shape[0]):
    if cell_info.iloc[i, 67] == '0001101':
        image_seg_rothia[image_seg == cell_info.iloc[i, 69]] = 1

g = pair_correlation(distance, 2, 200, image_seg.shape[0]*image_seg.shape[0])
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3), cm_to_inches(3))
plt.plot(np.arange(0, 40, 0.1)*0.07, g, color = 'orange')
plt.xlabel(r'$\it{r}$ [$\mu$m]', fontsize = 6)
plt.ylabel(r'$g(r)$', fontsize = 6)
plt.tick_params(direction = 'in', labelsize = 6)
plt.subplots_adjust(left = 0.3, right = 1, bottom = 0.3, top = 1)
plt.xlim(0,3)
plt.savefig('{}_gr.pdf'.format(sample), dpi = 300, transparent = True)
plt.close()

image_registered_sum = np.sum(image_registered, axis = 2)
image_registered_norm = image_registered_sum/np.max(image_registered_sum)
image_registered_norm[skimage.filters.gaussian(image_seg_centroid) > 0.05] = np.nan
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(5.5), cm_to_inches(5.5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
cmap = matplotlib.cm.inferno
cmap.set_bad('white')
ax.imshow(image_registered_norm[1400:1700, 900:1200], cmap = cmap)
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8})
# plt.gca().add_artist(scalebar)
plt.axis('off')
plt.savefig('{}_cell_lattice.pdf'.format(sample), dpi = 300)
plt.close()

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_seg_centroid[1400:1700, 900:1200], cmap = 'inferno')
plt.savefig('{}_cell_centroid.pdf'.format(sample), dpi = 300)
plt.close()

image_fft = np.fft.fftshift(np.fft.fft2(image_seg_centroid))
image_fft_log = np.log(np.abs(image_fft)+1)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(3), cm_to_inches(3))
plt.imshow(image_fft_log[950:1050, 950:1050], cmap = 'inferno')
plt.xlabel(r'$\it{q_x}$ [px$^{-1}$]', color = 'white', fontsize = 8, labelpad = 0)
plt.ylabel(r'$\it{q_y}$ [px$^{-1}$]', color = 'white', fontsize = 8, labelpad = 0)
plt.xticks([10,50,90], [-2,0,2])
plt.yticks([10,50,90], [2,0,-2])
plt.tick_params(direction = 'in', colors = 'white', labelsize = 8)
plt.axes().spines['bottom'].set_color('white')
plt.axes().spines['top'].set_color('white')
plt.axes().spines['left'].set_color('white')
plt.axes().spines['right'].set_color('white')
plt.text(0,-2, r'$\times 10^{-2}$', color = 'white', fontsize = 8)
plt.text(102,100, r'$\times 10^{-2}$',color = 'white', fontsize = 8, rotation = 90, verticalalignment = 'bottom')
plt.subplots_adjust(left = 0.27, bottom = 0.12, right = 0.88, top = 0.99)
plt.savefig('{}_cell_centroid_fft.pdf'.format(sample), dpi = 300, transparent =True)
plt.close()

image_fft_log[150,150] = np.average(image_fft_log)
plt.imshow(image_fft_log, cmap = 'inferno')
fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_fft_log, cmap = 'inferno')
plt.axis('off')
plt.savefig('{}_cell_centroid_fft_clipped.pdf'.format(sample), dpi = 300)
plt.close()

cell_id = image_seg[1536, 1100]
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(4), cm_to_inches(2))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'chartreuse', linewidth = 0.2, markersize = 1)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 0.2, markersize = 1)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'orange', linewidth = 0.2, markersize = 1)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 0.2, markersize = 1)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlabel('Channel [-]', fontsize = 4)
plt.ylabel('Intensity [-]', fontsize = 4)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 4, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=1, top=0.9, bottom=0.3)
plt.savefig('{}_cell_3_spectrum.pdf'.format(sample), dpi = 300)



sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0561/04_02_2019_DSGN0561_fov_12'
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
image_seg = np.load('{}_seg.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_registered = np.load('{}_registered.npy'.format(sample))

fig = plt.figure()
fig.set_size_inches(cm_to_inches(24), cm_to_inches(8))
spec_size = [0, 23, 43, 57, 63]
overall_ax = plt.Axes(fig, [0., 0., 1., 1.])
overall_ax.set_axis_off()
fig.add_axes(overall_ax)
emission = [int(np.round(495 + 8.9*i)) for i in range(23)]
column_labels = ['{}'.format(em) for em in emission]
row_labels = ['488', '514', '561', '633']
gs = GridSpec(4, 23)
for i in range(4):
    for j in range(spec_size[i+1] - spec_size[i]):
        ax = plt.subplot(gs[i, 23 - spec_size[i + 1] + spec_size[i] + j], frameon = False)
        ax.imshow(image_registered[:,:,spec_size[i] + j], cmap = 'inferno')
        ax.set_xticks([])
        ax.set_yticks([])

for i in range(23):
    plt.subplot(gs[0, i], frameon = False)
    plt.title(column_labels[i], fontsize = 8, rotation = 45, pad = 10, loc = 'right')

for i in range(0,4):
    ax = plt.subplot(gs[i, 22], frameon = False)
    ax.axis('on')
    ax.tick_params(length = 0, labelleft = 'off', labelbottom = 'off')
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(row_labels[i], fontsize = 8, rotation = 45, labelpad = 10)

ax = plt.subplot(gs[3,15], frameon = False)
scalebar = ScaleBar(0.0675, 'um', length_fraction = 1, height_fraction = 0.1, location = 'center')
scalebar.font_properties.set_size(8)
plt.gca().add_artist(scalebar)
ax.axis('off')
plt.annotate('Emission [nm]', xy = (0.5,1), xytext = (0.4,0.9), textcoords='figure fraction', fontsize = 8)
plt.annotate('Excitation [nm]', xy = (1,0.5), xytext = (0.97,0.5), textcoords='figure fraction', rotation = -90, fontsize = 8)
plt.subplots_adjust(left = 0.03, bottom = 0, right = 0.92, top = 0.75, wspace = 0.05, hspace = 0.05)
image_registered_spectral_filename = '{}_image_registered_spectral.pdf'.format(sample)
plt.savefig(image_registered_spectral_filename, dpi = 300, transparent = False)
plt.close()

image_registered_sum = np.sum(image_registered, axis = 2)
image_registerd_sum_norm = image_registered_sum/np.max(image_registered_sum)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registerd_sum_norm, cmap ='inferno')
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black')
plt.gca().add_artist(scalebar)
plt.savefig('{}_registered_sum_norm.pdf'.format(sample), dpi = 300)
plt.close()


fig = plt.figure()
fig.set_size_inches(cm_to_inches(5.5), cm_to_inches(5.5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification)
scalebar = ScaleBar(0.0675, 'um', frameon = False, location = 2, color = 'white', font_properties = {'size': 8})
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.savefig('{}_identification.pdf'.format(sample), dpi = 300)
plt.close()

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.axis('off')
ax.imshow(image_identification[0:200, 900:1100])
plt.savefig('{}_corncob_1.pdf'.format(sample), dpi = 300)
plt.close()

cell_id = image_seg[94, 950]
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.axes().spines['left'].set_color('black')
plt.axes().spines['bottom'].set_color('black')
# plt.xlabel('Channel', fontsize = 8, color = 'black', labelpad = 0)
# plt.ylabel('Intensity', fontsize = 8, color = 'black', labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = 'black')
plt.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.35)
plt.savefig('{}_porphyromonas_spectrum.pdf'.format(sample), dpi = 300, transparent = True)
plt.close()

cell_id = image_seg[678, 1059]
cell_barcode = cell_info.iloc[cell_info[69].values == cell_id, 67].values
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.axes().spines['left'].set_color('black')
plt.axes().spines['bottom'].set_color('black')
plt.xlabel('Channel', fontsize = 8, color = 'black', labelpad = 0)
plt.ylabel('Intensity', fontsize = 8, color = 'black', labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = 'black')
plt.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.35)
plt.savefig('{}_filifactor_spectrum.pdf'.format(sample), dpi = 300, transparent = True)
plt.close()

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[550:750, 950:1150])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.axis('off')
plt.axis('off')
plt.savefig('{}_filifactor_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

image_registered = np.load('{}_registered.npy'.format(sample))
image_registered_fft_filtered = np.zeros(image_registered.shape)
for i in range(image_registered.shape[2]):
    image_fft = np.fft.fftshift(np.fft.fft2(image_registered[:,:,i]))
    rr, cc = circle(1000,1000, 20)
    fft_mask = np.ones(image_fft.shape)
    fft_mask[rr, cc] = 0
    image_fft_masked = image_fft*fft_mask
    image_ift_masked = np.fft.ifft2(image_fft_masked)
    image_registered_fft_filtered[:,:,i] = np.abs(image_ift_masked)

spec_filtered = np.average(image_registered_fft_filtered[image_seg == 236, :], axis = 0)
spec = np.average(image_registered[image_seg == 236, :], axis = 0)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[1000:1200, 1000:1200])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.axis('off')
plt.savefig('{}_corynebacterium_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

cell_id = image_seg[1123, 1070]
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'orange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
# plt.xlabel('Channel', fontsize = 8, color = 'black', labelpad = 0)
# plt.ylabel('Intensity', fontsize = 8, color = 'black', labelpad = 0)
plt.axes().spines['left'].set_color('black')
plt.axes().spines['bottom'].set_color('black')
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2, colors = 'black')
plt.subplots_adjust(left=0.2, right = 0.98, top=0.95, bottom=0.35)
plt.savefig('{}_corynebacterium_spectrum.pdf'.format(sample), dpi = 300, transparent = True)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0561/04_02_2019_DSGN0561_fov_4'
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
image_seg = np.load('{}_seg.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))

cell_id = image_seg[1330, 1002]
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlabel('Channel', fontsize = 8, labelpad = 0)
plt.ylabel('Intensity', fontsize = 8, labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.35)
plt.savefig('{}_lautropia_spectrum.pdf'.format(sample), dpi = 300)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[1200:1400, 900:1100])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.axis('off')
plt.savefig('{}_lautropia_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0560/04_01_2019_DSGN0560_fov_4'
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
image_seg = np.load('{}_seg.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))

cell_id = image_seg[700, 1455]
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
# plt.xlabel('Channel', fontsize = 8, labelpad = 0)
# plt.ylabel('Intensity', fontsize = 8, labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.35)
plt.savefig('{}_lautropia_spectrum.pdf'.format(sample), dpi = 300, transparent = True)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[600:800, 1350:1550])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.axis('off')
plt.savefig('{}_lautropia_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0549/03_15_2019_DSGN0549_C_fov_1'
cell_info_DSGN0549 = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
lautropia_cells_DSGN0549 = cell_info_DSGN0549.iloc[cell_info_DSGN0549[67].values == '1100000',:]

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0560/04_01_2019_DSGN0560_fov_4'
cell_info_DSGN0560 = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
lautropia_cells_DSGN0560 = cell_info_DSGN0560.iloc[cell_info_DSGN0560[67].values == '0101100',:]

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0561/04_02_2019_DSGN0561_fov_4'
cell_info_DSGN0561 = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
lautropia_cells_DSGN0561 = cell_info_DSGN0561.iloc[cell_info_DSGN0561[67].values == '0110010',:]

lautropia_cells_DSGN0549[77] = 'DSGN0549'
lautropia_cells_DSGN0560[77] = 'DSGN0560'
lautropia_cells_DSGN0561[77] = 'DSGN0561'
lautropia_cells_physical_properties = pd.concat([lautropia_cells_DSGN0549.loc[:,72:77],lautropia_cells_DSGN0560.loc[:,72:77],lautropia_cells_DSGN0561.loc[:,72:77]])

embedding = umap.UMAP(n_neighbors = 25).fit_transform(lautropia_cells_physical_properties.iloc[:,0:4].values)
embedding_DSGN0549 = embedding[lautropia_cells_physical_properties[77].values == 'DSGN0549', :]
embedding_DSGN0560 = embedding[lautropia_cells_physical_properties[77].values == 'DSGN0560', :]
embedding_DSGN0561 = embedding[lautropia_cells_physical_properties[77].values == 'DSGN0561', :]

fig = plt.figure()
color_list = ['maroon', 'darkorange', 'dodgerblue']
fig.set_size_inches(cm_to_inches(5.5), cm_to_inches(5.5))
plt.plot(embedding_DSGN0549[:,0], embedding_DSGN0549[:,1], linestyle = 'None', marker = 'o', markersize = 1, alpha = 0.8, color = color_list[0], label = 'Probe set 1')
plt.plot(embedding_DSGN0560[:,0], embedding_DSGN0560[:,1], linestyle = 'None', marker = 'o', markersize = 1, alpha = 0.8, color = color_list[1], label = 'Probe set 2')
plt.plot(embedding_DSGN0561[:,0], embedding_DSGN0561[:,1], linestyle = 'None', marker = 'o', markersize = 1, alpha = 0.8, color = color_list[2], label = 'Probe set 3')
plt.xlabel('UMAP 1', fontsize = 8, labelpad = 0)
plt.ylabel('UMAP 2', fontsize = 8, labelpad = 0)
plt.xticks([5,10,15,20])
plt.yticks([5,10,15,20])
l = plt.legend(fontsize = 8, loc = 9, frameon = False)
for p in l.legendHandles:
    p._legmarker.set_alpha(0)

for i in range(3):
    l.get_texts()[i].set_color(color_list[i])

plt.tick_params(labelsize = 8, direction = 'in')
plt.subplots_adjust(left = 0.15, right = 0.98, bottom = 0.12, top = 0.98)
plt.savefig('{}_lautropia_physical_properties_umap.pdf'.format(sample), dpi = 300)
plt.close()

lc_0549_size_mean = lautropia_cells_DSGN0549[76].mean()
lc_0549_size_std = lautropia_cells_DSGN0549[76].std()
# lc_0549 = lautropia_cells_DSGN0549.loc[lautropia_cells_DSGN0549[76].values <= lc_0549_mean + 3*lc_0549_std, :]
lc_0560_size_mean = lautropia_cells_DSGN0560[76].mean()
lc_0560_size_std = lautropia_cells_DSGN0560[76].std()
# lc_0560 = lautropia_cells_DSGN0560.loc[lautropia_cells_DSGN0560[76].values <= lc_0560_mean + 3*lc_0560_std, :]
lc_0561_size_mean = lautropia_cells_DSGN0561[76].mean()
lc_0561_size_std = lautropia_cells_DSGN0561[76].std()
# lc_0561 = lautropia_cells_DSGN0561.loc[lautropia_cells_DSGN0561[76].values <= lc_0561_mean + 3*lc_0561_std, :]

# lc_0549_bs = astropy.stats.bootstrap(lautropia_cells_DSGN0549[76].values, bootnum = 10000, samples = 1)
# lc_0560_bs = astropy.stats.bootstrap(lautropia_cells_DSGN0560[76].values, bootnum = 10000, samples = 1)
# lc_0561_bs = astropy.stats.bootstrap(lautropia_cells_DSGN0561[76].values, bootnum = 10000, samples = 1)

fts = 8
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5.5), cm_to_inches(5.5))
gs = GridSpec(2, 2)
ax = plt.subplot(gs[0,0])
plt.hist(lautropia_cells_DSGN0549[76].values*0.07*0.07, bins = np.arange(0,500*0.07*0.07,25*0.07*0.07), color = 'red', density = True, alpha = 0.8)
plt.hist(lautropia_cells_DSGN0560[76].values*0.07*0.07, bins = np.arange(0,500*0.07*0.07,25*0.07*0.07), color = 'darkorange', density = True, alpha = 0.8)
plt.hist(lautropia_cells_DSGN0561[76].values*0.07*0.07, bins = np.arange(0,500*0.07*0.07,25*0.07*0.07), color = 'dodgerblue', density = True, alpha = 0.8)
plt.xlabel(r'Size [$\mu$m$^2$]', fontsize = fts, labelpad = 0)
plt.ylabel('Density', fontsize = fts, labelpad = 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(labelsize = fts, direction = 'in', length = 2)
ax = plt.subplot(gs[1,1])
plt.hist(lautropia_cells_DSGN0549[74].values, bins = np.arange(0,1,0.05), color = 'red', density = True, alpha = 0.8)
plt.hist(lautropia_cells_DSGN0560[74].values, bins = np.arange(0,1,0.05), color = 'darkorange', density = True, alpha = 0.8)
plt.hist(lautropia_cells_DSGN0561[74].values, bins = np.arange(0,1,0.05), color = 'dodgerblue', density = True, alpha = 0.8)
plt.xlabel(r'Eccentricity', fontsize = fts, labelpad = 0)
plt.ylabel('Density', fontsize = fts, labelpad = 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(labelsize = fts, direction = 'in', length = 2)
# ax = plt.subplot(gs[2,2])
# plt.hist(lc_0561[76].values*0.07*0.07, bins = np.arange(0,500*0.07*0.07,25*0.07*0.07), color = 'red', density = True)
# plt.xlabel(r'Size [$\mu$m$^2$]', fontsize = fts, labelpad = 0)
# plt.ylabel('Density', fontsize = fts, labelpad = 0)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tick_params(labelsize = fts, direction = 'in', length = 2)
ax = plt.subplot(gs[0,1])
plt.plot(lautropia_cells_DSGN0549[76].values*0.07*0.07,lautropia_cells_DSGN0549[74].values, color = 'red', markersize = 3, alpha = 0.5, markeredgewidth = 0, linestyle = 'None')
plt.plot(lautropia_cells_DSGN0560[76].values*0.07*0.07,lautropia_cells_DSGN0560[74].values, color = 'darkorange', markersize = 3, alpha = 0.5, markeredgewidth = 0, linestyle = 'None')
plt.plot(lautropia_cells_DSGN0561[76].values*0.07*0.07,lautropia_cells_DSGN0561[74].values, color = 'dodgerblue', markersize = 3, alpha = 0.5, markeredgewidth = 0, linestyle = 'None')
plt.xlabel('Size')
plt.ylabel('Eccentricity')




hist_cmap = 'inferno'
ax = plt.subplot(gs[0,1])
plt.hist2d(lc_0549_bs[:,0]*0.07*0.07, lc_0560_bs[:,0]*0.07*0.07, bins = (25,25), range = [[0, 500*0.07*0.07], [0,500*0.07*0.07]], cmap = hist_cmap)
plt.xlim(0, 500*0.07*0.07)
plt.ylim(0, 500*0.07*0.07)
plt.xlabel(r'DSGN0549', fontsize = fts, labelpad = 0)
plt.ylabel(r'DSGN0560', fontsize = fts, labelpad = 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(labelsize = fts, direction = 'in', length = 2)
ax = plt.subplot(gs[0,2])
plt.hist2d(lc_0549_bs[:,0]*0.07*0.07, lc_0561_bs[:,0]*0.07*0.07, bins = (25,25), range = [[0, 500*0.07*0.07], [0,500*0.07*0.07]], cmap = hist_cmap)
plt.xlim(0, 500*0.07*0.07)
plt.ylim(0, 500*0.07*0.07)
plt.xlabel(r'DSGN0549', fontsize = fts, labelpad = 0)
plt.ylabel(r'DSGN0561', fontsize = fts, labelpad = 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(labelsize = fts, direction = 'in', length = 2)
ax = plt.subplot(gs[1,2])
plt.hist2d(lc_0560_bs[:,0]*0.07*0.07, lc_0561_bs[:,0]*0.07*0.07, bins = (25,25), range = [[0, 500*0.07*0.07], [0,500*0.07*0.07]], cmap = hist_cmap)
plt.xlim(0, 500*0.07*0.07)
plt.ylim(0, 500*0.07*0.07)
plt.xlabel(r'DSGN0560', fontsize = fts, labelpad = 0)
plt.ylabel(r'DSGN0561', fontsize = fts, labelpad = 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(labelsize = fts, direction = 'in', length = 2)
plt.subplots_adjust(left = 0.12, right = 0.98, bottom = 0.1, top = 0.98, wspace = 0.65, hspace = 0.65)
plt.savefig('{}_lautropia_size_comparison.pdf'.format(sample), dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0561/04_02_2019_DSGN0561_fov_12'
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {67:str}, header = None)
image_seg = np.load('{}_seg.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_orientation = np.zeros(image_seg.shape)
for i in range(cell_info.shape[0]):
    image_orientation[image_seg == cell_info.iloc[i, 69]] = cell_info.iloc[i, 75]

image_orientation[image_seg == 0] = np.nan

cbarboxcolor = 'dimgrey'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(5.5), cm_to_inches(5.5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
cmap = matplotlib.cm.bwr
cmap.set_bad(color = 'black')
mappable = ax.imshow(image_orientation, cmap = cmap)
cbaxes = inset_axes(ax, width="30%", height="4%", loc = 2, bbox_to_anchor = (0.05,0.05,0.95,0.95), bbox_transform = ax.transAxes)
cbar = plt.colorbar(mappable, cax=cbaxes, ticks = [-np.pi/4, 0, np.pi/4], orientation='horizontal')
cbar.ax.set_xticklabels([r'-$\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$'], color = 'white')
cbar.set_label(r'Orientation [$\it{rad}$]', color = 'white', fontsize = 8, labelpad = 0)
cbar.ax.tick_params(labelsize = 8, direction = 'in', color = cbarboxcolor, length = 3)
ax.axis('off')
plt.savefig('{}_orientation.pdf'.format(sample), dpi = 300)
plt.close()

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

fig = plt.figure()
fig.set_size_inches(cm_to_inches(4), cm_to_inches(4))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification_filtered)
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8})
# plt.gca().add_artist(scalebar)
plt.savefig('{}_identification_filtered.pdf'.format(sample), dpi = 300)
plt.close()



fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification_filtered[700:900, 1700:1900])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.savefig('{}_macellibacteroides_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification_filtered[1300:1500, 600:800])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.savefig('{}_longibaculum_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

fig = plt.figure()
fig.set_size_inches(cm_to_inches(1.75), cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification_filtered[1250:1450, 1200:1400])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', font_properties = {'size': 8}, height_fraction = 0.02)
# plt.gca().add_artist(scalebar)
plt.savefig('{}_bacteroides_neighborhood.pdf'.format(sample), dpi = 300)
plt.close()

cell_id = image_seg[1400, 700]
cell_barcode = cell_info.iloc[cell_info[69].values == cell_id, 67].values
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlabel('Channel', fontsize = 8, labelpad = 0)
plt.ylabel('Intensity', fontsize = 8, labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.35)
plt.savefig('{}_longibaculum_spectrum.pdf'.format(sample), dpi = 300)

cell_id = image_seg[806, 1821]
cell_barcode = cell_info.iloc[cell_info[69].values == cell_id, 67].values
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
# plt.xlabel('Channel', fontsize = 8, labelpad = 0)
# plt.ylabel('Intensity', fontsize = 8, labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.35)
plt.savefig('{}_macellibacteroides_spectrum.pdf'.format(sample), dpi = 300)

cell_id = image_seg[1373, 1276]
cell_barcode = cell_info.iloc[cell_info[69].values == cell_id, 67].values
spec = cell_info.iloc[cell_info[69].values == cell_id, 0:63].values
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.75), cm_to_inches(1.75))
plt.plot(np.arange(0,23), spec[0,0:23], '-o', color = 'limegreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(23,43), spec[0,23:43], '-o', color = 'yellowgreen', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(43,57), spec[0,43:57], '-o', color = 'darkorange', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.plot(np.arange(57,63), spec[0,57:63], '-o', color = 'red', linewidth = 1, markersize = 3, markeredgewidth = 0)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
# plt.xlabel('Channel', fontsize = 8, labelpad = 0)
# plt.ylabel('Intensity', fontsize = 8, labelpad = 0)
plt.ylim(0,1.05)
plt.tick_params(labelsize = 8, direction = 'in', length = 2)
plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.35)
plt.savefig('{}_bacteroides_spectrum.pdf'.format(sample), dpi = 300)

image_orientation = np.zeros(image_seg.shape)
for i in range(cell_info.shape[0]):
    image_orientation[image_seg == cell_info.iloc[i, 69]] = cell_info.iloc[i, 75]

fig = plt.figure()
fig.set_size_inches(cm_to_inches(5.5), cm_to_inches(5.5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
mappable = ax.imshow(image_orientation, cmap = 'bwr')
cbaxes = inset_axes(ax, width="30%", height="3%", loc=2)
cbar = plt.colorbar(mappable, cax=cbaxes, ticks = [-np.pi/2, 0, np.pi/2], orientation='horizontal')
cbar.ax.set_xticklabels([r'-$\pi$/2', '0', r'$\pi$/2'])
cbar.ax.tick_params(labelsize = 4)
cbar.set_label(r'Orientation [$\it{rad}$]', fontsize = 4)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black')
plt.gca().add_artist(scalebar)
plt.savefig('{}_orientation.pdf'.format(sample), dpi = 300)
plt.close()

image_identification_taxon = np.zeros(image_identification.shape)
for i in range(cell_info.shape[0]):
    if cell_info.iloc[i, 67] == '0000110':
        image_identification_taxon[image_seg == cell_info.iloc[i, 69]] = 1

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

macellibacteroides_cells = cell_info.iloc[cell_info[67].values == '0100000', :]
macellibacteroides_epithelial_distance = np.zeros(macellibacteroides_cells.shape[0])
for i in range(macellibacteroides_cells.shape[0]):
    x = macellibacteroides_cells.iloc[i, 70]
    y = macellibacteroides_cells.iloc[i, 71]
    macellibacteroides_epithelial_distance[i] = np.abs(760*y + 2000*x - 1980*2000)/np.sqrt(760**2+2000**2)

probes = pd.read_csv('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0557/DSGN0557_primerset_B_barcode_selection_MostSimple_full_length_probes.csv', dtype = {'code': str})
barcode_sets = probes.code.drop_duplicates()
distance_list = []
for c in barcode_sets:
    cells = cell_info.iloc[cell_info[67].values == c, :]
    cell_epithelial_distance = np.zeros(cells.shape[0])
    for i in range(cells.shape[0]):
        x = cells.iloc[i, 70]
        y = cells.iloc[i, 71]
        cell_epithelial_distance[i] = np.abs(760*y + 2000*x - 1980*2000)/np.sqrt(760**2+2000**2)
    distance_list.append(cell_epithelial_distance)

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

macellibacteroides_cells = cell_info.iloc[cell_info[67].values == '0100000', :]
macellibacteroides_distance = np.zeros((macellibacteroides_cells.shape[0], macellibacteroides_cells.shape[0]))
for i in range(macellibacteroides_cells.shape[0]):
    for j in range(macellibacteroides_cells.shape[0]):
        x1 = macellibacteroides_cells.iloc[i, 70]
        y1 = macellibacteroides_cells.iloc[i, 71]
        x2 = macellibacteroides_cells.iloc[j, 70]
        y2 = macellibacteroides_cells.iloc[j, 71]
        macellibacteroides_distance[i,j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

dr = 2
nr = 200
g_bacteroides = pair_correlation(bacteroides_distance, dr, nr, image_seg.shape[0]*image_seg.shape[0])
g_hespellia = pair_correlation(hespellia_distance, dr, nr, image_seg.shape[0]*image_seg.shape[0])
g_macellibacteroides = pair_correlation(macellibacteroides_distance, dr, nr, image_seg.shape[0]*image_seg.shape[0])

fig = plt.figure()
color_list = ['darkorange', 'dodgerblue']
# color_list = ['maroon', 'darkorange', 'dodgerblue']
fig.set_size_inches(cm_to_inches(4), cm_to_inches(2.75))
# plt.plot((np.arange(0, dr*nr, dr)/macellibacteroides_cells[73].mean()), g_macellibacteroides, color = color_list[0], alpha = 0.8)
plt.plot((np.arange(0, dr*nr, dr)/hespellia_cells[73].mean()), g_hespellia, color = color_list[0], alpha = 0.8)
plt.plot((np.arange(0, dr*nr, dr)/bacteroides_cells[73].mean()), g_bacteroides, color = color_list[1], alpha = 0.8)
plt.xlabel(r'$\it{r}$/$a_{minor}$ [-]', fontsize = 8, color = 'black', labelpad = 0)
plt.ylabel('Pair correlation', fontsize = 8, color = 'black', labelpad = 0)
plt.tick_params(direction = 'in', labelsize = 8, colors = 'black')
plt.subplots_adjust(left = 0.16, right = 0.95, bottom = 0.25, top = 0.95)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.axes().spines['bottom'].set_color('black')
plt.axes().spines['left'].set_color('black')
l = plt.legend([r'$\it{Hespellia}$', r'$\it{Bacteroides}$'], fontsize = 8, loc = 2, frameon = False, bbox_to_anchor=(0.7, 1.05))
# l = plt.legend(['Macellibacteroides', 'Hespellia', 'Bacteroides'], fontsize = 8, loc = 2, frameon = False, bbox_to_anchor=(0.8, 1.05))
for i in range(2):
    l.get_texts()[i].set_color(color_list[i])
    l.get_texts()[i].set_ha('right')
    l.get_lines()[i].set_alpha(0)

plt.xlim(0,22)
plt.savefig('{}_gr.pdf'.format(sample), dpi = 300, transparent = True)
plt.close()



image_seg_bacteroides = np.zeros(image_seg.shape)
for i in range(cell_info.shape[0]):
    if cell_info.iloc[i, 67] == '1000001':
        image_seg_bacteroides[image_seg == cell_info.iloc[i, 69]] = 1

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_seg_bacteroides)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black')
plt.gca().add_artist(scalebar)
plt.savefig('{}_bacteroides_identification.pdf'.format(sample), dpi = 300)
plt.close()


fig = plt.figure()
fig.set_size_inches(cm_to_inches(4), cm_to_inches(2.75))
# plt.hist(macellibacteroides_epithelial_distance*0.07, bins = 30, density = True, histtype = 'step', color = color_list[0], alpha = 0.8)
plt.hist(hespellia_epithelial_distance*0.07, bins = 30, density = True, histtype = 'step', color = color_list[0], alpha = 0.8)
plt.hist(bacteroides_epithelial_distance*0.07, bins = 30, density = True, histtype = 'step', color = color_list[1], alpha = 0.8)
plt.xlabel(r'Distance [$\mu$m]', fontsize = 8, color = 'black', labelpad = 0)
plt.ylabel('Density [$\mu$m$^{-1}$]', fontsize = 8, color = 'black', labelpad = 0)
plt.tick_params(labelsize = 8, direction = 'in', colors = 'black')
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.ylim(0,0.028)
plt.yticks([0,0.01,0.02], [0,1,2])
l = plt.legend([r'$\it{Hespellia}$', r'$\it{Bacteroides}$'], fontsize = 8, loc = 2, frameon = False, bbox_to_anchor=(0.7, 1.05))
for i in range(2):
    l.get_texts()[i].set_color(color_list[i])
    l.get_texts()[i].set_ha('right')
    l.get_patches()[i].set_alpha(0)

plt.axes().spines['bottom'].set_color('black')
plt.axes().spines['left'].set_color('black')
plt.text(-2,0.025,r'$\times 10^{-2}$')
plt.ylim(0,0.03)
plt.subplots_adjust(left = 0.16, right = 0.95, bottom = 0.25, top = 0.95)
plt.savefig('{}_epithelial_distance.pdf'.format(sample), dpi = 300, transparent = True)
plt.close()



fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.plot(taxon_abundance_counts.counts.values, taxon_abundance_counts.abundance.values, 'o', color = 'orange', markersize = 2, alpha = 0.5)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlabel('PacBio Abundance [-]', fontsize = 8)
plt.ylabel('HiPRFISH Abundance [-]', fontsize = 8)
plt.axes().set_aspect('equal')
plt.tight_layout()
plt.savefig('{}_abundance_correlation.pdf'.format(sample), dpi = 300)

taxon_lookup = pd.read_csv('/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_21_2019_DSGN0557_wb_IDT/taxon_color_lookup.csv', dtype = {'code': str})
adjacency_seg = np.load('{}_adjacency_seg.npy'.format(sample))
edge_map = skimage.filters.sobel(adjacency_seg > 0)
rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
adjacency_matrix = pd.DataFrame(np.zeros((taxon_lookup.shape[0], taxon_lookup.shape[0])), index = taxon_lookup.code.values, columns = taxon_lookup.code.values)
for i in range(cell_info.shape[0]):
    edges = list(rag.edges(i+1))
    for e in edges:
        node_1 = e[0]
        node_2 = e[1]
        if (node_1 != 0) and (node_2 !=0):
            barcode_1 = cell_info.iloc[node_1-1,67]
            barcode_2 = cell_info.iloc[node_2-1, 67]
            adjacency_matrix.loc[barcode_1, barcode_2] += 1

detected_taxon = list(set(cell_info.iloc[:,67]))
taxon = list(taxon_lookup.sci_name.values)
# detected_taxon_nodes = [taxon_lookup.loc[taxon_lookup.code.values == d, 'sci_name'].values[0] for d in detected_taxon_converted]
G = nx.Graph()
# G.add_nodes_from(detected_taxon_nodes)
G.add_nodes_from(taxon)
color_list = []
for node in G.nodes:
    code = taxon_lookup.loc[taxon_lookup.sci_name.values == node, 'code'].values[0]
    if code in detected_taxon:
        # rgb = hsv_to_rgb(taxon_lookup.loc[taxon_lookup.sci_name == node, ['H', 'S', 'V']].values)
        # hex = '#{:02x}{:02x}{:02x}'.format(int(rgb[0,0]*255), int(rgb[0,1]*255), int(rgb[0,2]*255))
        color_list.append('blue')
    else:
        color_list.append('red')

abundance = cell_info.groupby(67).size().reset_index()
abundance.columns = ['barcode', 'abundance']
size_list = []
for node in G.nodes:
    code = taxon_lookup.loc[taxon_lookup.sci_name.values == node, 'code'].values[0]
    if code in detected_taxon:
        size_list.append((np.log(abundance.loc[abundance.barcode.values == code, 'abundance'].values[0])+1)*10)
    else:
        size_list.append((np.log(abundance.abundance.mean()))*10)

for i in detected_taxon:
    for j in detected_taxon:
        edge_color = 0.5*(hsv_to_rgb(taxon_lookup.loc[taxon_lookup.code == i, ['H', 'S', 'V']].values) + hsv_to_rgb(taxon_lookup.loc[taxon_lookup.code == j, ['H', 'S', 'V']].values))
        edge_color_hex ='#{:02x}{:02x}{:02x}'.format(int(edge_color[0,0]*255), int(edge_color[0,1]*255), int(edge_color[0,2]*255))
        edge_weight = adjacency_matrix.loc[i,j]
        if edge_weight > 0:
            node_i = taxon_lookup.loc[taxon_lookup.code.values == i, 'sci_name'].values[0]
            node_j = taxon_lookup.loc[taxon_lookup.code.values == j, 'sci_name'].values[0]
            G.add_edge(node_i, node_j, color = edge_color_hex, weight = edge_weight)

edges = G.edges()
# joblib.dump(G, '{}/spatial_association_network.pkl'.format(input_folder))
weights = [G[u][v]['weight'] for u,v in edges]
min_weight = np.min(np.log(weights))
max_weight = np.max(np.log(weights))
norm = matplotlib.colors.Normalize(vmin = min_weight, vmax = max_weight, clip = True)
mapper = cm.ScalarMappable(norm = norm, cmap = cm.Reds)
colors = [mapper.to_rgba(np.log(x)) for x in weights]
min_size = 0
max_size = np.max(size_list)
norm = matplotlib.colors.Normalize(vmin = min_size, vmax = max_size, clip = True)
mapper = cm.ScalarMappable(norm = norm, cmap = cm.Blues)
# node_colors = [mapper.to_rgba((min_size + max_size)/2) for x in size_list]
fig = plt.figure()
fig.set_size_inches(cm_to_inches(6),cm_to_inches(6))
# node_position = joblib.load('{}/node_position.pkl'.format(os.path.split(input_folder)[0]))
dist_df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
for row, data in nx.shortest_path_length(G):
    for col, dist in data.items():
        dist_df.loc[row,col] = np.log(dist+1)

dist_df = dist_df.fillna(dist_df.max().max())
node_position = nx.kamada_kawai_layout(G, dist = dist_df.to_dict())
nx.draw(G, pos = node_position, edges = edges, node_color = color_list, node_size = size_list, edge_color = colors, with_labels = True, font_size = 4)
# nx.draw_kamada_kawai(G, edges = edges, node_color = color_list, node_size = size_list, edge_color = colors, with_labels = True, font_size = 6)
plt.xlim(-1.1, 1.4)
plt.ylim(-1.1, 1.1)
plt.savefig('{}_spatial_association_network.pdf'.format(sample), format = 'png', dpi = 300)
plt.close()

weight_array = adjacency_matrix.sum(axis = 1) - np.diagonal(adjacency_matrix.values)
taxon_abundance_counts_weight_sorted = taxon_abundance_counts_weight.sort_values(by = 'weight', ascending = False)
taxa_sci_name = []
for i in taxon_abundance_counts_weight_sorted.code.values:
    taxa_sci_name.append(taxon_lookup.loc[taxon_lookup.code.values == i, 'sci_name'].values[0])

fig = plt.figure()
fig.set_size_inches(cm_to_inches(12), cm_to_inches(5))
plt.plot(taxon_abundance_counts_weight_sorted.weight.values, 'o', color = 'orange', markersize = 2, alpha = 0.5)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xticks(np.arange(23), taxa_sci_name, rotation = 30, ha = 'right', fontsize = 6)
plt.ylabel('Total edge weight [-]', fontsize = 8)
plt.yscale('log')
plt.tight_layout()
plt.savefig('{}_edge_weight.pdf'.format(sample), dpi = 300)
plt.close()

tick_labels = []
for i in adjacency_matrix.index.values:
    tick_labels.append(taxon_lookup.loc[taxon_lookup.code.values == i, 'sci_name'].values[0])
fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.imshow(np.log(adjacency_matrix.values), cmap = 'bwr')
plt.axes().xaxis.tick_top()
plt.xticks(np.arange(adjacency_matrix.shape[0]), tick_labels, rotation = 90, fontsize = 4)
plt.yticks(np.arange(adjacency_matrix.shape[0]), tick_labels, fontsize = 4)
plt.tick_params(length = 0)
plt.tight_layout()
plt.savefig('{}_adjacency_matrix.pdf'.format(sample), dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0549/DSGN0549_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0549_probes = pd.read_csv(sample, dtype = {'code':str})
DSGN0549_total_taxa = DSGN0549_probes.target_taxon.drop_duplicates().shape[0]
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0560/DSGN0560_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0560_probes = pd.read_csv(sample, dtype = {'code':str})
DSGN0560_total_taxa = DSGN0560_probes.target_taxon.drop_duplicates().shape[0]
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0561/DSGN0561_primerset_A_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0561_probes = pd.read_csv(sample, dtype = {'code':str})
DSGN0561_total_taxa = DSGN0561_probes.target_taxon.drop_duplicates().shape[0]

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0557/DSGN0557_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0557_probes = pd.read_csv(sample, dtype = {'code':str})
DSGN0557_total_taxa = DSGN0557_probes.target_taxon.drop_duplicates().shape[0]
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0563/DSGN0563_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0563_probes = pd.read_csv(sample, dtype = {'code':str})
DSGN0563_total_taxa = DSGN0563_probes.target_taxon.drop_duplicates().shape[0]
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0564/DSGN0564_primerset_A_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0564_probes = pd.read_csv(sample, dtype = {'code':str})
DSGN0564_total_taxa = DSGN0564_probes.target_taxon.drop_duplicates().shape[0]


fig = plt.figure()
fig.set_size_inches(cm_to_inches(8), cm_to_inches(5))
plt.plot([12,13,14], [DSGN0549_total_taxa, DSGN0560_total_taxa, DSGN0561_total_taxa], 'o', color = 'orange', label = 'Oral plaque')
plt.plot([12,13,14], [DSGN0557_total_taxa, DSGN0563_total_taxa, DSGN0564_total_taxa], 'o', color = 'blue', label = 'Mouse gut')
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(direction = 'in', labelsize = 8)
plt.xlabel('Maximum continuous homology [bp]', fontsize = 8)
plt.ylabel('Targeted taxa [-]', fontsize = 8)
plt.legend(fontsize = 8, frameon = False, bbox_to_anchor = (1.1, 1))
plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/targeted_taxon_number.pdf'.format(sample), dpi = 300)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.hist([DSGN0549_probes.Tm.values, DSGN0560_probes.Tm.values, DSGN0561_probes.Tm.values], bins = 50, color = ['chartreuse', 'blue', 'orange'], density = True, histtype = 'step', label = ['DSGN0549', 'DSGN0560', 'DSGN0561'])
plt.xlabel(r'Melting temperature [$^\circ$C]', fontsize = 8)
plt.ylabel('Probability density [-]', fontsize = 8)
plt.tick_params(direction = 'in', labelsize = 8)
plt.legend(fontsize = 8, frameon = False)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlim(53, 68)
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/probe_melting_temp_oral.pdf'.format(sample), dpi = 300)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.hist([DSGN0557_probes.Tm.values, DSGN0563_probes.Tm.values, DSGN0564_probes.Tm.values], bins = 50, color = ['chartreuse', 'blue', 'orange'], density = True, histtype = 'step', label = ['DSGN0557', 'DSGN0563', 'DSGN0564'])
plt.xlabel(r'Melting temperature [$^\circ$C]', fontsize = 8)
plt.ylabel('Probability density [-]', fontsize = 8)
plt.tick_params(direction = 'in', labelsize = 8)
plt.legend(fontsize = 8, frameon = False)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlim(53, 68)
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/probe_melting_temp_mouse.pdf'.format(sample), dpi = 300)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.boxplot([DSGN0549_probes.Tm.values, DSGN0560_probes.Tm.values, DSGN0561_probes.Tm.values], bins = 50, color = ['chartreuse', 'blue', 'orange'], density = True, histtype = 'step', label = ['DSGN0549', 'DSGN0560', 'DSGN0561'])
plt.xlabel(r'Melting temperature [$^\circ$C]', fontsize = 8)
plt.ylabel('Probability density [-]', fontsize = 8)
plt.tick_params(direction = 'in', labelsize = 8)
plt.legend(fontsize = 8, frameon = False)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlim(53, 68)
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/probe_per_taxa_oral.pdf'.format(sample), dpi = 300)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(8), cm_to_inches(5))
abundance_probe_multiplexity = DSGN0549_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
abundance_probe_multiplexity = abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
abundance_probe_multiplexity['relative_abundance'] = abundance_probe_multiplexity.abundance.values/abundance_probe_multiplexity.abundance.sum()
plt.plot(abundance_probe_multiplexity.relative_abundance, abundance_probe_multiplexity.probe_multiplexity, 'o', color = 'chartreuse', alpha = 0.5)
abundance_probe_multiplexity = DSGN0560_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
abundance_probe_multiplexity = abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
abundance_probe_multiplexity['relative_abundance'] = abundance_probe_multiplexity.abundance.values/abundance_probe_multiplexity.abundance.sum()
plt.plot(abundance_probe_multiplexity.relative_abundance, abundance_probe_multiplexity.probe_multiplexity, 'o', color = 'blue', alpha = 0.5)
abundance_probe_multiplexity = DSGN0561_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
abundance_probe_multiplexity = abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
abundance_probe_multiplexity['relative_abundance'] = abundance_probe_multiplexity.abundance.values/abundance_probe_multiplexity.abundance.sum()
plt.plot(abundance_probe_multiplexity.relative_abundance, abundance_probe_multiplexity.probe_multiplexity, 'o', color = 'orange', alpha = 0.5)
plt.xscale('log')
plt.yscale('log')
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlabel('Relative Abundance', fontsize = 8)
plt.ylabel('Probe Plurality', fontsize = 8)
plt.tick_params(direction = 'in', labelsize = 8)
plt.legend(['DSGN0549', 'DSGN0560', 'DSGN0561'], frameon = False, fontsize = 8, bbox_to_anchor = (1.1, 1))
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/probe_plurality_oral.pdf'.format(sample), dpi = 300)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(8), cm_to_inches(5))
abundance_probe_multiplexity = DSGN0557_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
abundance_probe_multiplexity = abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
abundance_probe_multiplexity['relative_abundance'] = abundance_probe_multiplexity.abundance.values/abundance_probe_multiplexity.abundance.sum()
plt.plot(abundance_probe_multiplexity.relative_abundance, abundance_probe_multiplexity.probe_multiplexity, 'o', color = 'chartreuse', alpha = 0.5)
abundance_probe_multiplexity = DSGN0563_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
abundance_probe_multiplexity = abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
abundance_probe_multiplexity['relative_abundance'] = abundance_probe_multiplexity.abundance.values/abundance_probe_multiplexity.abundance.sum()
plt.plot(abundance_probe_multiplexity.relative_abundance, abundance_probe_multiplexity.probe_multiplexity, 'o', color = 'blue', alpha = 0.5)
abundance_probe_multiplexity = DSGN0564_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
abundance_probe_multiplexity = abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
abundance_probe_multiplexity['relative_abundance'] = abundance_probe_multiplexity.abundance.values/abundance_probe_multiplexity.abundance.sum()
plt.plot(abundance_probe_multiplexity.relative_abundance, abundance_probe_multiplexity.probe_multiplexity, 'o', color = 'orange', alpha = 0.5)
plt.xscale('log')
plt.yscale('log')
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.xlabel('Relative Abundance', fontsize = 8)
plt.ylabel('Probe Plurality', fontsize = 8)
plt.tick_params(direction = 'in', labelsize = 8)
plt.legend(['DSGN0557', 'DSGN0563', 'DSGN0564'], frameon = False, fontsize = 8, bbox_to_anchor = (1.1, 1))
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/probe_plurality_mouse.pdf'.format(sample), dpi = 300)

def clasi_fish_cost(n):
    cost = n*300
    return(cost)

def hipr_fish_cost(n):
    cost = np.ceil(np.log(n))*300 + 1800
    return(cost)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.plot(np.arange(1, 1023), hipr_fish_cost(np.arange(1, 1023)), 's', label = 'HiPR-FISH', alpha = 0.8, markersize = 4, markeredgewidth = 0, color = (1, 0, 0.5))
plt.plot(np.arange(1, 120), clasi_fish_cost(np.arange(1, 120)), 'v', label = 'Improved CLASI-FISH', alpha = 0.8, markersize = 4, markeredgewidth = 0, color = (1, 0.5, 0))
plt.plot(np.arange(1, 15), clasi_fish_cost(np.arange(1, 15)), 'o', label = 'CLASI-FISH', alpha = 0.8, markersize = 4, markeredgewidth = 0, color = (0,0.5,1))

# plt.axes().spines['right'].set_visible(False)
# plt.axes().spines['top'].set_visible(False)
plt.xlabel('Number of taxa [-]', fontsize = 8, color = 'black')
plt.ylabel(r'Cost [$\$$]', fontsize = 8, color = 'black')
plt.ylim([99,10**5])
plt.yscale('log')
plt.axes().tick_params(direction = 'in', labelsize = 8, colors = 'black')
plt.axes().spines['left'].set_color('black')
plt.axes().spines['right'].set_color('black')
plt.axes().spines['bottom'].set_color('black')
plt.axes().spines['top'].set_color('black')
l = plt.legend(frameon = False, fontsize = 8)
for t in l.get_texts():
    t.set_color('black')

# plt.axes().set_major_formatter(ScalarFormatter())
plt.minorticks_off()
plt.subplots_adjust(left = 0.2, bottom = 0.15, right = 0.95, top = 0.95)
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/cost_comparison.pdf', dpi = 300, transparent = True)

data_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_reference_02_20_2019'
fluorophore_list = ['Alexa 488', 'Alexa 546', 'Rhodamine Red X', 'Pacific Green', 'Pacific Blue', 'Alexa 610', 'Alexa 647', 'DyLight 510 LS', 'Alexa 405', 'Alexa 532']
fig = plt.figure()
fig.set_size_inches(cm_to_inches(13.5), cm_to_inches(11.5))
gs = GridSpec(5, 2)
for i in range(10):
    pc, pr = np.divmod(i, 5)
    enc = 2**i
    spec = pd.read_csv('{}/08_18_2018_enc_{}_avgint.csv'.format(data_folder, enc), header = None)
    image_adjacent = sam_tab
    bkg = pd.read_csv('{}/08_18_2018_enc_{}_bkg.csv'.format(data_folder, enc), header = None)
    spec_average = np.average(spec.values, axis = 0)
    spec_average[0:32] = spec_average[0:32] - bkg.iloc[:,0].values
    spec_std = np.std(spec.values, axis = 0)
    ax = plt.subplot(gs[pr, pc])
    ax.errorbar(np.arange(0,32), spec_average[0:32], yerr = spec_std[0:32], color = 'purple', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '405 nm')
    ax.errorbar(np.arange(32,55), spec_average[32:55], yerr = spec_std[32:55], color = 'chartreuse', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '488 nm')
    ax.errorbar(np.arange(55,75), spec_average[55:75], yerr = spec_std[55:75], color = 'yellowgreen', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '514 nm')
    ax.errorbar(np.arange(75,89), spec_average[75:89], yerr = spec_std[75:89], color = 'orange', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '561 nm')
    ax.errorbar(np.arange(89,95), spec_average[89:95], yerr = spec_std[89:95], color = 'red', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '633 nm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.ylim(0, 0.9)
    plt.tick_params(direction = 'in', labelsize = 8, colors = 'black')
    plt.text(0.12, 0.6, "Readout Probe {}\n{}".format(i+1, fluorophore_list[i]), transform = ax.transAxes, fontsize = 8, color = 'black')

ax = plt.subplot(gs[4, 0])
ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
plt.tick_params(direction = 'in', labelsize = 8, colors = 'black')
l = ax.legend(frameon = False, fontsize = 6)
for t in l.get_texts():
    t.set_color('black')

plt.xlabel('Channels [-]', fontsize = 8, color = 'black')
plt.ylabel('Intensity [-]', fontsize = 8, color = 'black')
plt.subplots_adjust(left = 0.1, right = 0.98, bottom = 0.1, top = 0.98, wspace = 0.2, hspace = 0.2)
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/readout_probe_spectra.pdf'.format(sample), dpi = 300, transparent = True)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_1023_reference_08_18_2018'
# fluorophore_list = ['Alexa 488', 'Alexa 546', 'Rhodamine Red X', 'Pacific Green', 'Pacific Blue', 'Alexa 610', 'Alexa 647', 'DyLight 510 LS', 'Alexa 405', 'Alexa 532']
fig = plt.figure()
fig.set_size_inches(cm_to_inches(16), cm_to_inches(12))
encodings = [34, 139, 259, 306, 371, 579, 682, 783, 894, 1023]
gs = GridSpec(5, 2)
for i in range(10):
    pc, pr = np.divmod(i, 5)
    enc = encodings[i]
    barcode = '{0:010b}'.format(enc)
    spec = pd.read_csv('{}/08_18_2018_enc_{}_avgint.csv'.format(sample, enc), header = None)
    spec_average = np.average(spec.values, axis = 0)
    spec_std = np.std(spec.values, axis = 0)
    ax = plt.subplot(gs[pr, pc])
    ax.errorbar(np.arange(0,32), spec_average[0:32], yerr = spec_std[0:32], color = 'purple', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '405 nm')
    ax.errorbar(np.arange(32,55), spec_average[32:55], yerr = spec_std[32:55], color = 'chartreuse', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '488 nm')
    ax.errorbar(np.arange(55,75), spec_average[55:75], yerr = spec_std[55:75], color = 'yellowgreen', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '514 nm')
    ax.errorbar(np.arange(75,89), spec_average[75:89], yerr = spec_std[75:89], color = 'orange', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '561 nm')
    ax.errorbar(np.arange(89,95), spec_average[89:95], yerr = spec_std[89:95], color = 'red', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '633 nm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.ylim(0, 0.9)
    plt.tick_params(direction = 'in', labelsize = 8, colors = 'black')
    plt.text(0.12, 0.6, barcode, transform = ax.transAxes, fontsize = 8, color = 'black')

ax = plt.subplot(gs[4, 0])
ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
l = ax.legend(frameon = False, fontsize = 8, markerscale = 0)
for t in l.get_texts():
    t.set_color('black')

# color_list = ['purple', 'chartreuse', 'yellowgreen', 'orange', 'red']
# for i in range(5):
#     text = leg.get_texts()[i]
#     plt.setp(text, color = color_list[i])
plt.xlabel('Channels', fontsize = 8, color = 'black')
plt.ylabel('Intensity', fontsize = 8, color = 'black')
plt.subplots_adjust(left = 0.1, right = 0.95, bottom = 0.1, top = 0.95, wspace = 0.2, hspace = 0.2)
# plt.subplots_adjust(left = 0.35, right = 0.95, bottom = 0.15, top = 0.95, wspace = 0.2, hspace = 0.2)
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/select_barcode_spectra.pdf'.format(sample), dpi = 300, transparent = True)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_1023_reference_08_18_2018'
# fluorophore_list = ['Alexa 488', 'Alexa 546', 'Rhodamine Red X', 'Pacific Green', 'Pacific Blue', 'Alexa 610', 'Alexa 647', 'DyLight 510 LS', 'Alexa 405', 'Alexa 532']
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5.5), cm_to_inches(5.5))
encodings = [259, 579, 894]
gs = GridSpec(3,1)
for i in range(3):
    enc = encodings[i]
    barcode = '{0:010b}'.format(enc)
    spec = pd.read_csv('{}/08_18_2018_enc_{}_avgint.csv'.format(sample, enc), header = None)
    spec_average = np.average(spec.values, axis = 0)
    spec_std = np.std(spec.values, axis = 0)
    ax = plt.subplot(gs[i,0])
    ax.errorbar(np.arange(0,32), spec_average[0:32], yerr = spec_std[0:32], color = 'purple', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '405 nm')
    ax.errorbar(np.arange(32,55), spec_average[32:55], yerr = spec_std[32:55], color = 'chartreuse', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '488 nm')
    ax.errorbar(np.arange(55,75), spec_average[55:75], yerr = spec_std[55:75], color = 'yellowgreen', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '514 nm')
    ax.errorbar(np.arange(75,89), spec_average[75:89], yerr = spec_std[75:89], color = 'orange', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '561 nm')
    ax.errorbar(np.arange(89,95), spec_average[89:95], yerr = spec_std[89:95], color = 'red', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '633 nm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    plt.ylim(-0.05, 0.9)
    plt.tick_params(direction = 'in', labelsize = 8, colors = 'black')
    plt.text(0.12, 0.6, barcode, transform = ax.transAxes, fontsize = 8, color = 'black')

ax = plt.subplot(gs[2,0])
ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
# l = ax.legend(frameon = False, fontsize = 6, markerscale = 0)
# for t in l.get_texts():
#     t.set_color('black')

# color_list = ['purple', 'chartreuse', 'yellowgreen', 'orange', 'red']
# for i in range(5):
#     text = leg.get_texts()[i]
#     plt.setp(text, color = color_list[i])
plt.xlabel('Channels', fontsize = 8, color = 'black')
plt.ylabel('Intensity', fontsize = 8, color = 'black')
plt.subplots_adjust(left = 0.2, right = 0.95, bottom = 0.2, top = 0.95, wspace = 0.2, hspace = 0.2)\
# plt.subplots_adjust(left = 0.35, right = 0.95, bottom = 0.15, top = 0.95, wspace = 0.2, hspace = 0.2)
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/select_barcode_spectra.pdf'.format(sample), dpi = 300, transparent = True)


sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_1023_reference_08_18_2018'
# fluorophore_list = ['Alexa 488', 'Alexa 546', 'Rhodamine Red X', 'Pacific Green', 'Pacific Blue', 'Alexa 610', 'Alexa 647', 'DyLight 510 LS', 'Alexa 405', 'Alexa 532']
fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
encodings = [134, 355, 1023]
label_locations = [[0.12, 0.6], [0.4, 0.6], [0.12, 0.6]]
gs = GridSpec(3, 1)
for i in range(3):
    enc = encodings[i]
    barcode = '{0:010b}'.format(enc)
    spec = pd.read_csv('{}/08_18_2018_enc_{}_avgint.csv'.format(sample, enc), header = None)
    spec_average = np.average(spec.values, axis = 0)
    spec_std = np.std(spec.values, axis = 0)
    ax = plt.subplot(gs[i, 0])
    ax.errorbar(np.arange(0,32), spec_average[0:32], yerr = spec_std[0:32], alpha = 0.8, color = 'darkviolet', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '405 nm')
    ax.errorbar(np.arange(32,55), spec_average[32:55], yerr = spec_std[32:55], alpha = 0.8, color = 'limegreen', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '488 nm')
    ax.errorbar(np.arange(55,75), spec_average[55:75], yerr = spec_std[55:75], alpha = 0.8,color = 'yellowgreen', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '514 nm')
    ax.errorbar(np.arange(75,89), spec_average[75:89], yerr = spec_std[75:89], alpha = 0.8,color = 'darkorange', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '561 nm')
    ax.errorbar(np.arange(89,95), spec_average[89:95], yerr = spec_std[89:95], alpha = 0.8,color = 'red', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '633 nm')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    plt.ylim(-0.05, 1.05)
    plt.tick_params(direction = 'in', labelsize = 8, colors = 'black')
    plt.text(label_locations[i][0], label_locations[i][1], barcode, transform = ax.transAxes, fontsize = 8, color = 'black')

ax = plt.subplot(gs[2, 0])
ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
# l = ax.legend(frameon = False, fontsize = 6, markerscale = 0)
# for t in l.get_texts():
#     t.set_color('black')

# color_list = ['purple', 'chartreuse', 'yellowgreen', 'orange', 'red']
# for i in range(5):
#     text = leg.get_texts()[i]
#     plt.setp(text, color = color_list[i])
plt.xlabel('Channels', fontsize = 8, color = 'black')
plt.ylabel('Intensity', fontsize = 8, color = 'black')
plt.subplots_adjust(left = 0.15, right = 0.95, bottom = 0.15, top = 0.95, wspace = 0.2, hspace = 0.2)
# plt.subplots_adjust(left = 0.35, right = 0.95, bottom = 0.15, top = 0.95, wspace = 0.2, hspace = 0.2)
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/select_barcode_spectra_small.pdf'.format(sample), dpi = 300, transparent = True)


fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.plot([6, 10, 16], [15, 1023, 120], 'o', color = 'orange')
plt.text(6.2, 17, r'Valm $\it{et. al.} (2012)$', fontsize = 8)
plt.text(11, 160, r'Valm $\it{et. al.} (2016)$', fontsize = 8)
plt.text(8, 500, 'This work', fontsize = 8)
plt.yscale('log')
plt.xlim(5, 20)
plt.ylim(1, 1200)
plt.xlabel('Number of fluorophores [-]', fontsize = 8)
plt.ylabel('Multiplexity limit [-]', fontsize = 8)
plt.tick_params(direction = 'in', labelsize = 8)
plt.minorticks_off()
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/multiplexity_comparison.pdf'.format(sample), dpi = 300)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0565/DSGN0565_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0565_probes = pd.read_csv(sample, dtype = {'code':str})
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0566/DSGN0566_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0566_probes = pd.read_csv(sample, dtype = {'code':str})
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0567/DSGN0567_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0567_probes = pd.read_csv(sample, dtype = {'code':str})

DSGN0565_total_taxa = DSGN0565_probes.target_taxon.drop_duplicates().shape[0]
DSGN0566_total_taxa = DSGN0566_probes.target_taxon.drop_duplicates().shape[0]
DSGN0567_total_taxa = DSGN0567_probes.target_taxon.drop_duplicates().shape[0]

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.plot([12,13,14], [DSGN0565_total_taxa, DSGN0566_total_taxa, DSGN0567_total_taxa], 'o', color = 'blue', label = 'Mouse Gut Microbiome')
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(direction = 'in', labelsize = 8)
plt.xlabel('Maximum homology [bp]', fontsize = 8)
plt.ylabel('Targeted taxa [-]', fontsize = 8)
plt.legend(fontsize = 8, frameon = False)
plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0565_0566_0567_targeted_taxon_number.pdf'.format(sample), dpi = 300)
plt.close()

DSGN0565_abundance_probe_multiplexity = DSGN0565_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0565_abundance_probe_multiplexity = DSGN0565_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0565_abundance_probe_multiplexity['relative_abundance'] = DSGN0565_abundance_probe_multiplexity.abundance.values/DSGN0565_abundance_probe_multiplexity.abundance.sum()
DSGN0565_abundance_probe_multiplexity['mch'] = 12
DSGN0566_abundance_probe_multiplexity = DSGN0566_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0566_abundance_probe_multiplexity = DSGN0566_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0566_abundance_probe_multiplexity['relative_abundance'] = DSGN0566_abundance_probe_multiplexity.abundance.values/DSGN0566_abundance_probe_multiplexity.abundance.sum()
DSGN0566_abundance_probe_multiplexity['mch'] = 13
DSGN0567_abundance_probe_multiplexity = DSGN0567_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0567_abundance_probe_multiplexity = DSGN0567_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0567_abundance_probe_multiplexity['relative_abundance'] = DSGN0567_abundance_probe_multiplexity.abundance.values/DSGN0567_abundance_probe_multiplexity.abundance.sum()
DSGN0567_abundance_probe_multiplexity['mch'] = 14

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
bp = plt.boxplot([DSGN0565_abundance_probe_multiplexity.probe_multiplexity.values,
             DSGN0566_abundance_probe_multiplexity.probe_multiplexity.values,
             DSGN0567_abundance_probe_multiplexity.probe_multiplexity.values],
             patch_artist = True,
             notch = True,
             labels = [12, 13, 14])
plt.xlabel('Maximum homology [bp]', fontsize = 8)
plt.ylabel('Probe multiplicity', fontsize = 8)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(direction = 'in', labelsize = 8)
plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
for patch in bp['boxes']:
    patch.set_facecolor('blue')

plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0565_0566_0567_probe_multiplicity_boxplot.pdf'.format(sample), dpi = 300)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0568/DSGN0568_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0568_probes = pd.read_csv(sample, dtype = {'code':str})
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0569/DSGN0569_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0569_probes = pd.read_csv(sample, dtype = {'code':str})
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0570/DSGN0570_primerset_B_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0570_probes = pd.read_csv(sample, dtype = {'code':str})

DSGN0568_total_taxa = DSGN0568_probes.target_taxon.drop_duplicates().shape[0]
DSGN0569_total_taxa = DSGN0569_probes.target_taxon.drop_duplicates().shape[0]
DSGN0570_total_taxa = DSGN0570_probes.target_taxon.drop_duplicates().shape[0]

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.plot([12,13,14], [DSGN0568_total_taxa, DSGN0569_total_taxa, DSGN0570_total_taxa], 'o', color = 'orange', label = 'Muribacculum')
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(direction = 'in', labelsize = 8)
plt.xlabel('Maximum homology [bp]', fontsize = 8)
plt.ylabel('Targeted taxa [-]', fontsize = 8)
# plt.legend(fontsize = 8, frameon = False)
plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0568_0569_0570_targeted_taxon_number.pdf'.format(sample), dpi = 300)
plt.close()

DSGN0568_abundance_probe_multiplexity = DSGN0568_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0568_abundance_probe_multiplexity = DSGN0568_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0568_abundance_probe_multiplexity['relative_abundance'] = DSGN0568_abundance_probe_multiplexity.abundance.values/DSGN0568_abundance_probe_multiplexity.abundance.sum()
DSGN0568_abundance_probe_multiplexity['mch'] = 12
DSGN0569_abundance_probe_multiplexity = DSGN0569_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0569_abundance_probe_multiplexity = DSGN0569_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0569_abundance_probe_multiplexity['relative_abundance'] = DSGN0569_abundance_probe_multiplexity.abundance.values/DSGN0569_abundance_probe_multiplexity.abundance.sum()
DSGN0566_abundance_probe_multiplexity['mch'] = 13
DSGN0570_abundance_probe_multiplexity = DSGN0570_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0570_abundance_probe_multiplexity = DSGN0570_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0570_abundance_probe_multiplexity['relative_abundance'] = DSGN0570_abundance_probe_multiplexity.abundance.values/DSGN0570_abundance_probe_multiplexity.abundance.sum()
DSGN0570_abundance_probe_multiplexity['mch'] = 14

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
bp = plt.boxplot([DSGN0568_abundance_probe_multiplexity.probe_multiplexity.values,
             DSGN0569_abundance_probe_multiplexity.probe_multiplexity.values,
             DSGN0570_abundance_probe_multiplexity.probe_multiplexity.values],
             patch_artist = True,
             notch = True,
             labels = [12, 13, 14])
plt.xlabel('Maximum homology [bp]', fontsize = 8)
plt.ylabel('Probe multiplicity', fontsize = 8)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(direction = 'in', labelsize = 8)
plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
for patch in bp['boxes']:
    patch.set_facecolor('blue')

plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0568_0569_0570_probe_multiplicity_boxplot.pdf'.format(sample), dpi = 300)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0571/DSGN0571_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0571_probes = pd.read_csv(sample, dtype = {'code':str})
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0572/DSGN0572_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0572_probes = pd.read_csv(sample, dtype = {'code':str})
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0573/DSGN0573_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0573_probes = pd.read_csv(sample, dtype = {'code':str})

DSGN0571_total_taxa = DSGN0571_probes.target_taxon.drop_duplicates().shape[0]
DSGN0572_total_taxa = DSGN0572_probes.target_taxon.drop_duplicates().shape[0]
DSGN0573_total_taxa = DSGN0573_probes.target_taxon.drop_duplicates().shape[0]

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.plot([12,13,14], [DSGN0571_total_taxa, DSGN0572_total_taxa, DSGN0573_total_taxa], 'o', color = 'chartreuse', label = 'Oral plaque microbiome')
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(direction = 'in', labelsize = 8)
plt.xlabel('Maximum homology [bp]', fontsize = 8)
plt.ylabel('Targeted taxa [-]', fontsize = 8)
plt.legend(fontsize = 8, frameon = False)
plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0571_0572_0573_targeted_taxon_number.pdf'.format(sample), dpi = 300)

DSGN0571_abundance_probe_multiplexity = DSGN0571_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0571_abundance_probe_multiplexity = DSGN0571_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0571_abundance_probe_multiplexity['relative_abundance'] = DSGN0571_abundance_probe_multiplexity.abundance.values/DSGN0571_abundance_probe_multiplexity.abundance.sum()
DSGN0571_abundance_probe_multiplexity['mch'] = 12
DSGN0572_abundance_probe_multiplexity = DSGN0572_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0572_abundance_probe_multiplexity = DSGN0572_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0572_abundance_probe_multiplexity['relative_abundance'] = DSGN0572_abundance_probe_multiplexity.abundance.values/DSGN0572_abundance_probe_multiplexity.abundance.sum()
DSGN0572_abundance_probe_multiplexity['mch'] = 13
DSGN0573_abundance_probe_multiplexity = DSGN0573_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0573_abundance_probe_multiplexity = DSGN0573_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0573_abundance_probe_multiplexity['relative_abundance'] = DSGN0573_abundance_probe_multiplexity.abundance.values/DSGN0573_abundance_probe_multiplexity.abundance.sum()
DSGN0573_abundance_probe_multiplexity['mch'] = 14

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
bp = plt.boxplot([DSGN0571_abundance_probe_multiplexity.probe_multiplexity.values,
             DSGN0572_abundance_probe_multiplexity.probe_multiplexity.values,
             DSGN0573_abundance_probe_multiplexity.probe_multiplexity.values],
             patch_artist = True,
             notch = True,
             labels = [12, 13, 14])
plt.xlabel('Maximum homology [bp]', fontsize = 8)
plt.ylabel('Probe multiplicity', fontsize = 8)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(direction = 'in', labelsize = 8)
plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
for patch in bp['boxes']:
        patch.set_facecolor('chartreuse')

plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0571_0572_0573_probe_multiplicity_boxplot.pdf'.format(sample), dpi = 300)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0574/DSGN0574_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0574_probes = pd.read_csv(sample, dtype = {'code':str})
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0575/DSGN0575_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0575_probes = pd.read_csv(sample, dtype = {'code':str})
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0576/DSGN0576_primerset_C_barcode_selection_MostSimple_full_length_probes.csv'
DSGN0576_probes = pd.read_csv(sample, dtype = {'code':str})

DSGN0574_total_taxa = DSGN0574_probes.target_taxon.drop_duplicates().shape[0]
DSGN0575_total_taxa = DSGN0575_probes.target_taxon.drop_duplicates().shape[0]
DSGN0576_total_taxa = DSGN0576_probes.target_taxon.drop_duplicates().shape[0]

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
plt.plot([12,13,14], [DSGN0574_total_taxa, DSGN0575_total_taxa, DSGN0576_total_taxa], 'o', color = 'purple', label = 'Oral top 10')
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(direction = 'in', labelsize = 8)
plt.xlabel('Maximum homology [bp]', fontsize = 8)
plt.ylabel('Targeted taxa [-]', fontsize = 8)
plt.legend(fontsize = 8, frameon = False)
plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0574_0575_0576_targeted_taxon_number.pdf'.format(sample), dpi = 300)

DSGN0574_abundance_probe_multiplexity = DSGN0574_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0574_abundance_probe_multiplexity = DSGN0574_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0574_abundance_probe_multiplexity['relative_abundance'] = DSGN0574_abundance_probe_multiplexity.abundance.values/DSGN0574_abundance_probe_multiplexity.abundance.sum()
DSGN0574_abundance_probe_multiplexity['mch'] = 12
DSGN0575_abundance_probe_multiplexity = DSGN0575_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0575_abundance_probe_multiplexity = DSGN0575_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0575_abundance_probe_multiplexity['relative_abundance'] = DSGN0575_abundance_probe_multiplexity.abundance.values/DSGN0575_abundance_probe_multiplexity.abundance.sum()
DSGN0575_abundance_probe_multiplexity['mch'] = 13
DSGN0576_abundance_probe_multiplexity = DSGN0576_probes.groupby('target_taxon').agg({'abundance': np.unique, 'probe_numeric_id': lambda x: np.count_nonzero(np.unique(x))}).reset_index()
DSGN0576_abundance_probe_multiplexity = DSGN0576_abundance_probe_multiplexity.rename(columns = {'probe_numeric_id': 'probe_multiplexity'})
DSGN0576_abundance_probe_multiplexity['relative_abundance'] = DSGN0576_abundance_probe_multiplexity.abundance.values/DSGN0576_abundance_probe_multiplexity.abundance.sum()
DSGN0576_abundance_probe_multiplexity['mch'] = 14

fig = plt.figure()
fig.set_size_inches(cm_to_inches(6), cm_to_inches(6))
bp = plt.boxplot([DSGN0574_abundance_probe_multiplexity.probe_multiplexity.values,
             DSGN0575_abundance_probe_multiplexity.probe_multiplexity.values,
             DSGN0576_abundance_probe_multiplexity.probe_multiplexity.values],
             patch_artist = True,
             notch = True,
             labels = [12, 13, 14])
plt.xlabel('Maximum homology [bp]', fontsize = 8)
plt.ylabel('Probe multiplicity', fontsize = 8)
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.tick_params(direction = 'in', labelsize = 8)
plt.axes().yaxis.set_major_locator(MaxNLocator(integer=True))
for patch in bp['boxes']:
        patch.set_facecolor('purple')

plt.tight_layout()
plt.savefig('/workdir/hs673/Runs/V1/Samples/HIPRFISH_temp/simulation/DSGN0574_0575_0576_probe_multiplicity_boxplot.pdf'.format(sample), dpi = 300)

image = np.zeros((500, 500, 100, 23))
particles = skimage.morphology.ball(5)
for i in range(30):
    for j in range(30):
        for k in range(5):
            image[(i+1)*15 - 5: (i+1)*15 + 6, (j+1)*15 - 5: (j+1)*15 + 6, (k+1)*15 - 5: (k+1)*15 + 6, np.random.randint(23)] = particles

image_label = skimage.measure.label(image)
image_label_color = color.label2rgb(image_label, bg_color = (0,0,0), bg_label = 0)
image_label_color[:,0:2,0:2,:] = 1
image_label_color[0:2,:,0:2,:] = 1
image_label_color[0:2,0:2,:,:] = 1
image_label_color[:,497:499,97:99,:] = 1
image_label_color[497:499,:,97:99,:] = 1
image_label_color[497:499,497:499,:,:] = 1
image_label_color[:,0:2,97:99,:] = 1
image_label_color[0:2,:,97:99,:] = 1
image_label_color[0:2,497:499,:,:] = 1
image_label_color[:,497:499,0:2,:] = 1
image_label_color[497:499,:,0:2,:] = 1
image_label_color[497:499,0:2,:,:] = 1

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/08_18_2018_1023_reference/08_18_2018_enc_1'
excitations = ['405', '488', '514', '561', '633']
image_name = ['{}_{}.czi'.format(sample, x) for x in excitations]
image_stack = [bioformats.load_image(filename) for filename in image_name]
segmentation, image_registered = segment_images(image_stack)

fig = plt.figure(frameon = False, figsize = (cm_to_inches(5), cm_to_inches(5)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_t1, cmap = 'inferno')
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black', box_color = 'black')
plt.gca().add_artist(scalebar)
segfilename = sample + '_t_1_patch.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/08_18_2018_1023_reference'
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(3))
spec = pd.read_csv('{}/08_18_2018_enc_1_avgint.csv'.format(sample, enc), header = None)
spec_average = np.average(spec.values, axis = 0)
spec_std = np.std(spec.values, axis = 0)
plt.errorbar(np.arange(0,32), spec_average[0:32], yerr = spec_std[0:32], color = 'purple', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '405 nm')
plt.errorbar(np.arange(32,55), spec_average[32:55], yerr = spec_std[32:55], color = 'chartreuse', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '488 nm')
plt.errorbar(np.arange(55,75), spec_average[55:75], yerr = spec_std[55:75], color = 'yellowgreen', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '514 nm')
plt.errorbar(np.arange(75,89), spec_average[75:89], yerr = spec_std[75:89], color = 'orange', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '561 nm')
plt.errorbar(np.arange(89,95), spec_average[89:95], yerr = spec_std[89:95], color = 'red', fmt = '-o', markersize = 2, ecolor = 'w', capsize = 1, linewidth = 1, elinewidth = 1, capthick = 1, label = '633 nm')
plt.axes().spines['right'].set_visible(False)
plt.axes().spines['top'].set_visible(False)
plt.axes().spines['left'].set_color('black')
plt.axes().spines['right'].set_color('black')
plt.axes().spines['top'].set_color('black')
plt.axes().spines['bottom'].set_color('black')
plt.tick_params(direction = 'in', labelsize = 8, colors = 'black')
l = plt.legend(frameon = False, fontsize = 6)
for t in l.get_texts():
    t.set_color('black')

plt.xlabel('Channels', color = 'black', fontsize = 8)
plt.ylabel('Intensity [A.U.]', color = 'black', fontsize = 8)
plt.tight_layout()
spec_filename = '{}/08_18_2018_enc_1_avgint.pdf'.format(sample)
plt.savefig(spec_filename, dpi = 300, transparent = True)


sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_207_colon_1_fov_1'
restored = np.load('{}_noise2noise_restored.npy'.format(sample))
image_registered = np.load('{}_registered_channel_average.npy'.format(sample))
image_registered = np.moveaxis(image_registered, [0,1,2,3],[3,2,1,0])
image_seg = np.load('{}_seg.npy'.format(sample))
image_seg_color = color.label2rgb(image_seg, bg_color = (0,0,0), bg_label = 0)

restored_sum = np.sum(restored, axis = 0)
restored_norm = restored_sum/np.max(restored_sum)

fig = plt.figure(frameon = False, figsize = (cm_to_inches(5), cm_to_inches(5)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered[0, 40, 300:500, 750:950, 0], cmap = 'inferno')
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black', box_color = 'black')
plt.gca().add_artist(scalebar)
segfilename = sample + '_t_0_patch.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False, figsize = (cm_to_inches(5), cm_to_inches(5)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered[0, 40, 300:500, 750:950, 1], cmap = 'inferno')
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black', box_color = 'black')
plt.gca().add_artist(scalebar)
segfilename = sample + '_t_1_patch.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False, figsize = (cm_to_inches(5), cm_to_inches(5)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_sum[300:500, 750:950], cmap = 'inferno')
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black', box_color = 'black')
plt.gca().add_artist(scalebar)
segfilename = sample + '_patch_time_average.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False, figsize = (cm_to_inches(5), cm_to_inches(5)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(restored[0, 40, 300:500, 750:950], cmap = 'inferno')
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black', box_color = 'black')
plt.gca().add_artist(scalebar)
segfilename = sample + '_restored_patch.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False, figsize = (cm_to_inches(5), cm_to_inches(5)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_final[300:500, 750:950], cmap = 'inferno')
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black', box_color = 'black')
plt.gca().add_artist(scalebar)
segfilename = sample + '_local_neighborhood_enhancement_patch.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

image_registered_sum = np.sum(image_registered, axis = (0,4))
image_registered_sum_log = np.log10(image_registered_sum+1e-8)
image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_log.reshape(np.prod(image_registered_sum_log.shape), 1)).reshape(image_registered_sum_log.shape)
image0 = restored_sum*(image_bkg_filter == 0)
image1 = restored_sum*(image_bkg_filter == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
if (i0 < i1):
    image_bkg_filter_mask = (image_bkg_filter == 1)
else:
    image_bkg_filter_mask = (image_bkg_filter == 0)

fig = plt.figure(frameon = False, figsize = (cm_to_inches(5), cm_to_inches(5)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_bkg_filter_mask[40,300:500, 750:950])
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black', box_color = 'black')
plt.gca().add_artist(scalebar)
segfilename = sample + '_patch_bkg_filter_mask.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False, figsize = (cm_to_inches(5), cm_to_inches(5)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.moveaxis(image_seg_color, [0,1],[1,0])[300:500,750:950])
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black', box_color = 'black')
plt.gca().add_artist(scalebar)
segfilename = sample + '_patch_segmentation.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/microbiome_experiments/imaging/05_01_2019_DSGN0573C/05_01_2019_DSGN0573C_fov_1_corncob'
image_raw = np.load('{}_registered.npy'.format(sample))
image_raw_average = np.average(image_raw, axis = 3)
image_restored = np.load('{}_restored.npy'.format(sample))
image_segmentation = np.load('{}_seg.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))

fig = plt.figure(frameon = False, figsize = (cm_to_inches(2.625), cm_to_inches(2.625)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_raw_average[:,:,20], cmap = 'inferno')
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 2, height_fraction = 0.02, box_color = 'white', font_properties = {'size': 8})
# plt.gca().add_artist(scalebar)
segfilename = sample + '_corncob_registered_channel_average.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False, figsize = (cm_to_inches(2.625), cm_to_inches(2.625)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_restored[:,:,20], cmap = 'inferno')
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 2, height_fraction = 0.02, box_color = 'white', font_properties = {'size': 8})
# plt.gca().add_artist(scalebar)
segfilename = sample + '_corncob_noise2noise_restored.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False, figsize = (cm_to_inches(2.625), cm_to_inches(2.625)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
image_segmentation_color = color.label2rgb(image_segmentation, bg_color = (0,0,0), bg_label = 0)
ax.imshow(image_segmentation_color[:,:,20])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 2, height_fraction = 0.02, box_color = 'white', font_properties = {'size': 8})
# plt.gca().add_artist(scalebar)
segfilename = sample + '_corncob_segmentation.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False, figsize = (cm_to_inches(2.625), cm_to_inches(2.625)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[:,:,20])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 2, height_fraction = 0.02, box_color = 'white', font_properties = {'size': 8})
# plt.gca().add_artist(scalebar)
segfilename = sample + '_corncob_identification.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

dim = 0.2
image_identification_highlight = dim*image_identification.copy()
for i in [49,89,102,111,112,143]:
    image_identification_highlight[image_segmentation == i,:] = (1/dim)*image_identification_highlight[image_segmentation == i,:]

fig = plt.figure(frameon = False, figsize = (cm_to_inches(2.625), cm_to_inches(2.625)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification_highlight[:,:,20])
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 2, height_fraction = 0.02, box_color = 'white', font_properties = {'size': 8})
# plt.gca().add_artist(scalebar)
segfilename = sample + '_corncob_identification.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()


fig = plt.figure(frameon = False, figsize = (cm_to_inches(5), cm_to_inches(5)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[:,:,20], alpha = 0.5)
ax.imshow(image_restored[:,:,20], cmap = 'inferno', alpha = 0.5)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black', box_color = 'black')
plt.gca().add_artist(scalebar)
segfilename = sample + '_corncob_restored_identification_overlay.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures'
patch_size = 11
phi_range = 9
increment = 5
intervals = np.zeros(2)
line_matrices = np.zeros((patch_size, 2, phi_range), dtype = int)
for phi in range(phi_range):
  angle_index = phi
  intervals[0] = int(np.round(increment*np.cos(phi*np.pi/phi_range)))
  intervals[1] = int(np.round(increment*np.sin(phi*np.pi/phi_range)))
  max_interval = intervals[np.argmax(np.abs(intervals))]
  interval_signs = np.sign(intervals)
  line_n = int(2*np.abs(max_interval)+1)
  if line_n < patch_size:
    line_diff = int((patch_size - line_n)/2)
    for li in range(line_n):
      h1 = interval_signs[0]*li*(2*np.abs(intervals[0])+1)/line_n
      line_matrices[li+line_diff, 0, angle_index] = int(np.sign(h1)*np.floor(np.abs(h1)) + increment -  intervals[0])
      h2 = interval_signs[1]*li*(2*np.abs(intervals[1])+1)/line_n
      line_matrices[li+line_diff, 1, angle_index] = int(np.sign(h2)*np.floor(np.abs(h2)) + increment -  intervals[1])
    for li in range(line_diff):
      line_matrices[li, :, angle_index] = line_matrices[line_diff, :, angle_index]
    for li in range(line_diff):
      line_matrices[li+line_n+line_diff, :, angle_index] = line_matrices[line_n + line_diff - 1, :, angle_index]
  else:
    for li in range(line_n):
      h1 = interval_signs[0]*li*(2*np.abs(intervals[0])+1)/line_n
      line_matrices[li, 0, angle_index] = int(np.sign(h1)*np.floor(np.abs(h1)) + increment -  intervals[0])
      h2 = interval_signs[1]*li*(2*np.abs(intervals[1])+1)/line_n
      line_matrices[li, 1, angle_index] = int(np.sign(h2)*np.floor(np.abs(h2)) + increment -  intervals[1])

lp_patches = np.zeros((11,11,9))
for i in range(9):
    lp_patches[line_matrices[:,0,i],line_matrices[:,1,i], i] = 1

for i in range(9):
    fig = plt.figure(frameon = False, figsize = (cm_to_inches(1), cm_to_inches(1)))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.axis('off')
    ax.imshow(lp_patches[:,:,i], cmap = 'gray')
    filename = '{}/line_matrices_patch_{}.pdf'.format(sample, i)
    fig.savefig(filename, dpi = 300)
    plt.close()

patch_size = 11
phi_range = 9
theta_range = 9
increment = 5
intervals = np.zeros(3)
line_matrices = np.zeros((patch_size, 3, (theta_range - 1)*phi_range), dtype = int)
for theta in range(1, theta_range):
    for phi in range(phi_range):
      angle_index = (theta - 1)*phi_range + phi
      intervals[0] = int(np.round(increment*np.cos(phi*np.pi/phi_range)*np.sin(theta*np.pi/theta_range)))
      intervals[1] = int(np.round(increment*np.sin(phi*np.pi/phi_range)*np.sin(theta*np.pi/theta_range)))
      intervals[2] = int(np.round(increment*np.cos(theta*np.pi/theta_range)))
      max_interval = intervals[np.argmax(np.abs(intervals))]
      interval_signs = np.sign(intervals)
      line_n = int(2*np.abs(max_interval)+1)
      if line_n < patch_size:
        line_diff = int((patch_size - line_n)/2)
        for li in range(line_n):
          h1 = interval_signs[0]*li*(2*np.abs(intervals[0])+1)/line_n
          line_matrices[li+line_diff, 0, angle_index] = int(np.sign(h1)*np.floor(np.abs(h1)) + increment -  intervals[0])
          h2 = interval_signs[1]*li*(2*np.abs(intervals[1])+1)/line_n
          line_matrices[li+line_diff, 1, angle_index] = int(np.sign(h2)*np.floor(np.abs(h2)) + increment -  intervals[1])
          h3 = interval_signs[2]*li*(2*np.abs(intervals[2])+1)/line_n
          line_matrices[li+line_diff, 2, angle_index] = int(np.sign(h3)*np.floor(np.abs(h3)) + increment -  intervals[2])
        for li in range(line_diff):
          line_matrices[li, :, angle_index] = line_matrices[line_diff, :, angle_index]
        for li in range(line_diff):
          line_matrices[li+line_n+line_diff, :, angle_index] = line_matrices[line_n + line_diff - 1, :, angle_index]
      else:
        for li in range(line_n):
          h1 = interval_signs[0]*li*(2*np.abs(intervals[0])+1)/line_n
          line_matrices[li, 0, angle_index] = int(np.sign(h1)*np.floor(np.abs(h1)) + increment -  intervals[0])
          h2 = interval_signs[1]*li*(2*np.abs(intervals[1])+1)/line_n
          line_matrices[li, 1, angle_index] = int(np.sign(h2)*np.floor(np.abs(h2)) + increment -  intervals[1])
          h3 = interval_signs[2]*li*(2*np.abs(intervals[2])+1)/line_n
          line_matrices[li, 2, angle_index] = int(np.sign(h3)*np.floor(np.abs(h3)) + increment -  intervals[2])


sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_207_colon_1_fov_1'
image_restored = np.load('/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_207_colon_1_fov_1_noise2noise_restored.npy')
image_restored = image_restored[0,20,:,:]
image_restored_norm = image_restored/np.max(image_restored)
lp = np.zeros((11,9))
for i in range(9):
    lp[:,i] = image_restored_norm[374:385,278:289][line_matrices[:,0,i], line_matrices[:,1,i]]

fig = plt.figure(frameon = False, figsize = (cm_to_inches(3), cm_to_inches(3*11/9)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.axis('off')
ax.imshow(lp, cmap = 'inferno', vmin = 0, vmax = 1)
filename = '{}_lp.pdf'.format(sample, i)
fig.savefig(filename, dpi = 300)
plt.close()

lp_min = np.min(lp, axis = 0)
lp_max = np.max(lp, axis = 0)
lp_norm = (lp - lp_min[None,:])/(lp_max[None,:] - lp_min[None,:])
fig = plt.figure(frameon = False, figsize = (cm_to_inches(3), cm_to_inches(3*11/9)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.axis('off')
ax.imshow(lp_norm, cmap = 'inferno', vmin = 0, vmax = 1)
filename = '{}_lp_norm.pdf'.format(sample, i)
fig.savefig(filename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False, figsize = (cm_to_inches(3), cm_to_inches(3/9)))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.axis('off')
ax.imshow(lp_norm[5,:].reshape(1,9), cmap = 'inferno', vmin = 0, vmax = 1)
filename = '{}_lp_norm_central_line.pdf'.format(sample, i)
fig.savefig(filename, dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_205_colon_1_fov_1'
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), header = None, dtype = {67:str})
cell_ids_list = cell_info[67].drop_duplicates()
nr = 500
g_matrix = np.zeros((cell_ids_list.shape[0], nr))
for k in range(cell_ids_list.shape[0]):
    cid = cell_ids_list.values[k]
    cell_info_sub = cell_info.loc[cell_info[67].values == cid, :]
    distance = np.zeros((cell_info_sub.shape[0], cell_info_sub.shape[0]))
    for i in range(cell_info_sub.shape[0]):
        for j in range(cell_info_sub.shape[0]):
            x1 = cell_info_sub.iloc[i, 70]
            x2 = cell_info_sub.iloc[j, 70]
            y1 = cell_info_sub.iloc[i, 71]
            y2 = cell_info_sub.iloc[j, 71]
            distance[i,j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    g_matrix[k,:] = pair_correlation(distance, 4, 500, image_seg.shape[0]*image_seg.shape[0])



fig = plt.figure()
fig.set_size_inches(cm_to_inches(3), cm_to_inches(3))
plt.plot(np.arange(0, 2000, 4)*0.07, g, color = 'orange')
plt.xlabel(r'$\it{r}$ [$\mu$m]', fontsize = 6)
plt.ylabel(r'$g$', fontsize = 6)
plt.tick_params(direction = 'in', labelsize = 6)
plt.subplots_adjust(left = 0.3, right = 1, bottom = 0.3, top = 1)
plt.xlim(0,3)
plt.savefig('{}_gr.pdf'.format(sample), dpi = 300, transparent = True)
plt.close()



sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_205_colon_1_fov_1_z_20'
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), header = None)
save_segmentation(image_seg, sample)
save_identification(image_identification, sample)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_205_colon_1_fov_1_z_40'
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), header = None)
save_segmentation(image_seg, sample)
save_identification(image_identification, sample)


sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_207_colon_1_fov_1_z_20'
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), header = None)
save_segmentation(image_seg, sample)
save_identification(image_identification, sample)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_207_colon_1_fov_1_z_40'
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), header = None)
save_segmentation(image_seg, sample)
save_identification(image_identification, sample)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_209_colon_1_fov_1_z_20'
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), header = None)
save_segmentation(image_seg, sample)
save_identification(image_identification, sample)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_209_colon_1_fov_1_z_40'
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), header = None)
save_segmentation(image_seg, sample)
save_identification(image_identification, sample)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_214_colon_1_fov_1_z_20'
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), header = None)
save_segmentation(image_seg, sample)
save_identification(image_identification, sample)

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_214_colon_1_fov_1_z_40'
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), header = None)
save_segmentation(image_seg, sample)
save_identification(image_identification, sample)

fig = plt.figure(frameon = False)
fig.set_size_inches(5,5)
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(color.label2rgb(image_seg,bg_color = (0,0,0), bg_label = 0))
# scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'black')
# plt.gca().add_artist(scalebar)
plt.axis('off')
segfilename = sample + '_seg.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_205_colon_1_fov_1'
adjacency_matrix_205 = []
for z in [20,30,40,50,60]:
    adjacency_matrix_205.append(pd.read_csv('{}_z_{}_adjacency_matrix.csv'.format(sample, z), dtype = {0:str}).set_index('Unnamed: 0'))

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_207_colon_1_fov_1'
adjacency_matrix_207 = []
for z in [20,30,40,50,60]:
    adjacency_matrix_207.append(pd.read_csv('{}_z_{}_adjacency_matrix.csv'.format(sample, z), dtype = {0:str}).set_index('Unnamed: 0'))

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_209_colon_1_fov_1'
adjacency_matrix_209 = []
for z in [20,30,40,50,60]:
    adjacency_matrix_209.append(pd.read_csv('{}_z_{}_adjacency_matrix.csv'.format(sample, z), dtype = {0:str}).set_index('Unnamed: 0'))

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/05_09_2019_DSGN0567B_214_colon_1_fov_1'
adjacency_matrix_214 = []
for z in [20,30,40,50,60]:
    adjacency_matrix_214.append(pd.read_csv('{}_z_{}_adjacency_matrix.csv'.format(sample, z), dtype = {0:str}).set_index('Unnamed: 0'))

adjacency_matrix_205_np = np.stack([x.values for x in adjacency_matrix_205], axis = 2)
adjacency_matrix_207_np = np.stack([x.values for x in adjacency_matrix_207], axis = 2)
adjacency_matrix_209_np = np.stack([x.values for x in adjacency_matrix_209], axis = 2)
adjacency_matrix_214_np = np.stack([x.values for x in adjacency_matrix_214], axis = 2)

clindamycin_differential_adjacency_matrix = np.average(adjacency_matrix_205_np, axis = 2)/np.average(adjacency_matrix_214_np, axis = 2)
clindamycin_differential_adjacency_matrix = np.log10(clindamycin_differential_adjacency_matrix)
clinda_filter = np.isnan(clindamycin_differential_adjacency_matrix).all(axis = 0)
clindamycin_differential_adjacency_matrix_filtered = clindamycin_differential_adjacency_matrix[~clinda_filter,:]
clindamycin_differential_adjacency_matrix_filtered = clindamycin_differential_adjacency_matrix_filtered[:,~clinda_filter]
clindamycin_differential_adjacency_matrix_temp = clindamycin_differential_adjacency_matrix_filtered.copy()
clindamycin_differential_adjacency_matrix_temp[np.isnan(clindamycin_differential_adjacency_matrix_filtered)] = 0
clindamycin_differential_adjacency_matrix_temp[np.isinf(clindamycin_differential_adjacency_matrix_filtered)] = 0
clinda_most_changed = np.sum(np.abs(clindamycin_differential_adjacency_matrix_temp), axis = 0)
clinda_most_changed_index = np.argsort(clinda_most_changed)
clinda_small = clindamycin_differential_adjacency_matrix_filtered[clinda_most_changed_index < 20, :]
clinda_small = clinda_small[:,clinda_most_changed_index < 20]

folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/'
taxon_lookup = pd.read_csv('{}/taxon_color_lookup.csv'.format(folder), dtype = {'code': str})
label_codes = adjacency_matrix_205[0].columns[~clinda_filter]
label_codes = label_codes[clinda_most_changed_index < 20]
tick_labels = [taxon_lookup.loc[taxon_lookup.code.values == x, 'sci_name'].values[0] for x in label_codes.tolist()]
cmap = matplotlib.cm.RdBu
cmap.set_bad('black')
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(8.5),cm_to_inches(8.5))
ax = plt.Axes(fig, [0.01, 0.4, 0.59, 0.59])
fig.add_axes(ax)
mappable = ax.imshow(clinda_small, cmap = cmap)
plt.xticks(np.arange(clinda_small.shape[0]), tick_labels, rotation = 90, style = 'italic')
plt.yticks(np.arange(clinda_small.shape[1]), tick_labels, style = 'italic')
plt.tick_params(direction = 'in', length = 0, labelsize = 8)
ax.xaxis.tick_bottom()
ax.yaxis.tick_right()
cbaxes = inset_axes(ax, width="4%", height="30%", loc = 4, bbox_to_anchor = (0,-0.6,1.5,1.5), bbox_transform = ax.transAxes)
cbar = plt.colorbar(mappable, cax=cbaxes, orientation = 'vertical')
# cbar.ax.set_xticklabels([r'-$\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$'], color = 'white')
cbar.set_label(r'$\log_{10}$(Fold Change)', color = 'black', fontsize = 8)
cbar.ax.tick_params(labelsize = 8, direction = 'in', color = 'black', length = 3)
cbar.ax.yaxis.tick_left()
cbar.ax.yaxis.set_label_position('left')
plt.subplots_adjust(left = 0.01, bottom = 0.4, right = 0.6, top = 0.99)
plt.savefig('{}/clindamycin_differential_adjacency_matrix.pdf'.format(folder), dpi = 300, transparent = True)
plt.close()

folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/'
taxon_lookup = pd.read_csv('{}/taxon_color_lookup.csv'.format(folder), dtype = {'code': str})
label_codes = adjacency_matrix_205[0].columns[~clinda_filter]
label_codes = label_codes[clinda_most_changed_index < 20]
tick_labels = [taxon_lookup.loc[taxon_lookup.code.values == x, 'sci_name'].values[0] for x in label_codes.tolist()]
cmap = matplotlib.cm.RdBu
cmap.set_bad('grey')
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(12.5),cm_to_inches(12.5))
ax = plt.Axes(fig, [0.01, 0.4, 0.59, 0.59])
fig.add_axes(ax)
mappable = ax.imshow(clinda_small, cmap = cmap)
plt.xticks(np.arange(clinda_small.shape[0]), tick_labels, rotation = 90, style = 'italic', color = 'white')
plt.yticks(np.arange(clinda_small.shape[1]), tick_labels, style = 'italic', color = 'white')
plt.tick_params(direction = 'in', length = 0, labelsize = 10, colors = 'white')
ax.xaxis.tick_bottom()
ax.yaxis.tick_right()
cbaxes = inset_axes(ax, width="4%", height="30%", loc = 4, bbox_to_anchor = (0,-0.6,1.5,1.5), bbox_transform = ax.transAxes)
cbar = plt.colorbar(mappable, cax=cbaxes, orientation = 'vertical')
# cbar.ax.set_xticklabels([r'-$\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$'], color = 'white')
cbar.set_label(r'$\log_{10}$(Fold Change)', color = 'white', fontsize = 10)
cbar.ax.tick_params(labelsize = 10, direction = 'in', colors = 'white', length = 3)
cbar.ax.yaxis.tick_left()
cbar.ax.yaxis.set_label_position('left')
plt.subplots_adjust(left = 0.01, bottom = 0.4, right = 0.6, top = 0.99)
plt.savefig('{}/clindamycin_differential_adjacency_matrix_presentation.pdf'.format(folder), dpi = 300, transparent = True)
plt.close()


ciprofloxacin_differential_adjacency_matrix = np.average(adjacency_matrix_207_np, axis = 2)/np.average(adjacency_matrix_209_np, axis = 2)
ciprofloxacin_differential_adjacency_matrix = np.log10(ciprofloxacin_differential_adjacency_matrix)
cipro_filter = np.isnan(ciprofloxacin_differential_adjacency_matrix).all(axis = 0)
ciprofloxacin_differential_adjacency_matrix_filtered = ciprofloxacin_differential_adjacency_matrix[~cipro_filter,:]
ciprofloxacin_differential_adjacency_matrix_filtered = ciprofloxacin_differential_adjacency_matrix_filtered[:,~cipro_filter]
ciprofloxacin_differential_adjacency_matrix_temp = ciprofloxacin_differential_adjacency_matrix_filtered.copy()
ciprofloxacin_differential_adjacency_matrix_temp[np.isnan(ciprofloxacin_differential_adjacency_matrix_filtered)] = 0
ciprofloxacin_differential_adjacency_matrix_temp[np.isinf(ciprofloxacin_differential_adjacency_matrix_filtered)] = 0
cipro_most_changed = np.sum(np.abs(ciprofloxacin_differential_adjacency_matrix_temp), axis = 0)
cipro_most_changed_index = np.argsort(cipro_most_changed)
cipro_small = ciprofloxacin_differential_adjacency_matrix_filtered[cipro_most_changed_index < 20, :]
cipro_small = cipro_small[:,cipro_most_changed_index < 20]

folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/'
taxon_lookup = pd.read_csv('{}/taxon_color_lookup.csv'.format(folder), dtype = {'code': str})
label_codes = adjacency_matrix_205[0].columns[~cipro_filter]
label_codes = label_codes[cipro_most_changed_index < 20]
tick_labels = [taxon_lookup.loc[taxon_lookup.code.values == x, 'sci_name'].values[0] for x in label_codes.tolist()]
cmap = matplotlib.cm.RdBu
cmap.set_bad('black')
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(8.5),cm_to_inches(8.5))
ax = plt.Axes(fig, [0.01, 0.4, 0.59, 0.59])
fig.add_axes(ax)
mappable = ax.imshow(cipro_small, cmap = cmap)
plt.xticks(np.arange(cipro_small.shape[0]), tick_labels, rotation = 90, style = 'italic')
plt.yticks(np.arange(cipro_small.shape[1]), tick_labels, style = 'italic')
plt.tick_params(direction = 'in', length = 0, labelsize = 8)
ax.xaxis.tick_bottom()
ax.yaxis.tick_right()
cbaxes = inset_axes(ax, width="4%", height="30%", loc = 4, bbox_to_anchor = (0,-0.6,1.5,1.5), bbox_transform = ax.transAxes)
cbar = plt.colorbar(mappable, cax=cbaxes, orientation = 'vertical')
# cbar.ax.set_xticklabels([r'-$\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$'], color = 'white')
cbar.set_label(r'$\log_{10}$(Fold Change)', color = 'black', fontsize = 8)
cbar.ax.tick_params(labelsize = 8, direction = 'in', color = 'black', length = 3)
cbar.ax.yaxis.tick_left()
cbar.ax.yaxis.set_label_position('left')
plt.subplots_adjust(left = 0.01, bottom = 0.4, right = 0.6, top = 0.99)
plt.savefig('{}/ciprofloxacin_differential_adjacency_matrix.pdf'.format(folder), dpi = 300, transparent = True)
plt.close()

folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/05_09_2019_DSGN0567B/'
taxon_lookup = pd.read_csv('{}/taxon_color_lookup.csv'.format(folder), dtype = {'code': str})
label_codes = adjacency_matrix_205[0].columns[~cipro_filter]
label_codes = label_codes[cipro_most_changed_index < 20]
tick_labels = [taxon_lookup.loc[taxon_lookup.code.values == x, 'sci_name'].values[0] for x in label_codes.tolist()]
cmap = matplotlib.cm.RdBu
cmap.set_bad('grey')
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(12.5),cm_to_inches(12.5))
ax = plt.Axes(fig, [0.01, 0.4, 0.59, 0.59])
fig.add_axes(ax)
mappable = ax.imshow(cipro_small, cmap = cmap)
plt.xticks(np.arange(cipro_small.shape[0]), tick_labels, rotation = 90, style = 'italic', color = 'white')
plt.yticks(np.arange(cipro_small.shape[1]), tick_labels, style = 'italic', color = 'white')
plt.tick_params(direction = 'in', length = 0, labelsize = 10)
ax.xaxis.tick_bottom()
ax.yaxis.tick_right()
cbaxes = inset_axes(ax, width="4%", height="30%", loc = 4, bbox_to_anchor = (0,-0.6,1.5,1.5), bbox_transform = ax.transAxes)
cbar = plt.colorbar(mappable, cax=cbaxes, orientation = 'vertical')
# cbar.ax.set_xticklabels([r'-$\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$'], color = 'white')
cbar.set_label(r'$\log_{10}$(Fold Change)', color = 'white', fontsize = 10)
cbar.ax.tick_params(labelsize = 10, direction = 'in', colors = 'white', length = 3)
cbar.ax.yaxis.tick_left()
cbar.ax.yaxis.set_label_position('left')
plt.subplots_adjust(left = 0.01, bottom = 0.4, right = 0.6, top = 0.99)
plt.savefig('{}/ciprofloxacin_differential_adjacency_matrix_presentation.pdf'.format(folder), dpi = 300, transparent = True)
plt.close()


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

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(4),cm_to_inches(4))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((1000,1000)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 2, font_properties = {'size': 8})
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_1000_4_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(2.625),cm_to_inches(2.625))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((200,200)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 2, font_properties = {'size': 8})
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_200_2.625_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(1.75),cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((200,200)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, font_properties = {'size': 8}, height_fraction = 0.02)
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_200_1.75_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(5.5),cm_to_inches(5.5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((300,300)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, font_properties = {'size': 8})
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_300_5.5_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(5.5),cm_to_inches(5.5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((2000,2000)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, font_properties = {'size': 8})
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_2000_5.5_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(4),cm_to_inches(4))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((1000,1000)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, font_properties = {'size': 8})
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_1000_4_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(6),cm_to_inches(6))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((2000,2000)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 'upper left', font_properties = {'size': 8})
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_2000_6_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(1.875),cm_to_inches(1.875))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((200,200)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, label = None, label_formatter = lambda x, y:'', fixed_value = 5, height_fraction = 0.04)
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_200_1.875_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(1.75),cm_to_inches(1.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((200,200)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, label = None, label_formatter = lambda x, y:'', fixed_value = 5, height_fraction = 0.04)
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_200_1.75_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(4),cm_to_inches(4))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((2000,2000)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, label = None, label_formatter = lambda x, y:'', fixed_value = 20, height_fraction = 0.02)
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_2000_4_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(2.5),cm_to_inches(2.5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((200,200)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 1, label = None, label_formatter = lambda x, y:'', fixed_value = 5, height_fraction = 0.02)
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_200_2.5_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/06_14_2019_DSGN0549/03_15_2019_DSGN0549_C_fov_2'
image_registered = np.load('{}_registered.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_registered_average = np.average(image_registered, axis = 2)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(8),cm_to_inches(8))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_average, cmap = 'inferno')
segfilename = sample + '_average.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

image_seg_color = color.label2rgb(image_seg, bg_color = (0,0,0), bg_label = 0)
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(8),cm_to_inches(8))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_seg_color)
segfilename = sample + '_seg.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(8),cm_to_inches(8))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification)
segfilename = sample + '_identification.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(8),cm_to_inches(8))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_final_filtered, cmap = 'inferno')
segfilename = sample + '_enhanced.pdf'
fig.savefig(segfilename, dpi = 300)
plt.close()

sample = '/Users/hao/Documents/Research/Cornell/Projects/HIPRFISH/figures_dropbox'
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(8),cm_to_inches(8))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(np.zeros((2000,2000)), cmap = 'RdBu', alpha = 0)
scalebar = ScaleBar(0.0675, 'um', frameon = False, color = 'white', location = 2, font_properties = {'size': 8})
plt.gca().add_artist(scalebar)
plt.axis('off')
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
segfilename = sample + '/fov_size_2000_8_scalebar.pdf'
fig.savefig(segfilename, dpi = 300, transparent = True)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(25),cm_to_inches(25))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_average, cmap = 'inferno')
segfilename = sample + '_average.pdf'
fig.savefig(segfilename, dpi = 1000)
plt.close()

## make legend figure for Pseudopropionibacterium - Cardiobacterium - Schwartzia consortium
fig = plt.figure()
fig.set_size_inches(cm_to_inches(3.5), cm_to_inches(1))
plt.plot((0,0), color = (0,0.5,1), label = r'$\it{Pseudopropionibacterium}')
plt.plot((0,0), color = (1,0.5,0), label = r'$\it{Cardiobacterium}')
plt.plot((0,0), color = (0.5,1,0), label = r'$\it{Schwartzia}')
plt.legend(frameon = False, fontsize = 8)


def main():
    parser = argparse.ArgumentParser('Design FISH probes for a complex microbial community')
    parser.add_argument('input_folder', type = str, help = 'Input folder containing images of biological samples')
    parser.add_argument('-p', '--probe_design_filename', dest = 'probe_design_filename', type = str, nargs = '*', help = 'Input folder containing images of biological samples')
    parser.add_argument('-f', '--sample_prefix', dest = 'sample_prefix', type = str, nargs = '*', help = 'Input folder containing images of biological samples')
    args = parser.parse_args()
    summarize_error_rate(args.input_folder, args.probe_design_filename)
    plot_representative_cell_image(args.input_folder, args.probe_design_filename)
    return

if __name__ == '__main__':
    main()
