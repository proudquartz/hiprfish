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


ncbi = NCBITaxa()
javabridge.start_vm(class_path=bioformats.JARS)


sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P12_04_2017/03_16_2020_DSGN0673_P12_04_2017_fov_2'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_sum[1250:1550, 1150:1450], cmap = 'inferno')
image_filename = '{}_lautropia_image_registered_sum.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[1250:1550, 1150:1450])
image_filename = '{}_lautropia_image_identification.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P09_06_2018/03_16_2020_DSGN0673_P09_16_2018_fov_5'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_sum[350:650,550:850], cmap = 'inferno')
image_filename = '{}_lautropia_image_registered_sum.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[350:650,550:850])
image_filename = '{}_lautropia_image_identification.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_15_2020_DSGN0673/03_12_2020_DSGN0673_plaque_02_29_2020_fov_1'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {'cell_barcode':str})
image_registered_sum = np.sum(image_registered, axis = 2)



#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P09_06_2018/03_16_2020_DSGN0673_P09_16_2018_fov_5'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_sum, cmap = 'inferno')
image_filename = '{}_lautropia_image_registered_sum.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification)
image_filename = '{}_lautropia_image_identification.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_20_2019/03_16_2020_DSGN0673_P02_20_2019_fov_2'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_sum[1300:1600,600:900], cmap = 'inferno')
image_filename = '{}_lautropia_image_registered_sum.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_identification[1300:1600,600:900])
image_filename = '{}_lautropia_image_identification.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()

taxa_barcode_sciname = pd.read_csv('/workdir/hs673/Runs/V1/Samples/HIPRFISH_7/simulation/DSGN0673/DSGN0673_taxa_barcode.csv', dtype = {'code':str})

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_fov_12'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)
image_crop = image_registered_sum[1215:1325, 840:950]
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(2),cm_to_inches(2))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_crop, cmap = 'inferno')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0.65, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
crop_filename = '{}_1_confocal.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()


sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_fov_14'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)
image_crop = image_registered_sum[1215:1325, 840:950]

plt.imshow(image_seg)
cell_label = image_seg[1251,895]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(2),cm_to_inches(2))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_crop, cmap = 'inferno')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0.65, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
crop_filename = '{}_1_confocal.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_12_1_488'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
# check point: check segmentation to figure out which label correspond to the background
seg_mask = segmentation != 1
seg_labeled = skimage.measure.label(seg_mask)

# check point: check for the label corresponding to the desired cell
cell_seg_mask = seg_labeled == 1
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 50)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
intensity_profile = np.sum(straightened_image, axis = 0)
intensity_profile = intensity_profile/np.max(intensity_profile)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(3),cm_to_inches(4.5))
gs = GridSpec(2,1)
ax = plt.subplot(gs[0,0])
ax.imshow(straightened_image, cmap = 'inferno')
plt.axis('off')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
ax = plt.subplot(gs[1,0])
plt.plot(0.016*np.arange(0,intensity_profile.shape[0]), intensity_profile, color = (0,0.5,1))
plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
plt.xlabel(r'Length [$\mu$m]', fontsize = 8, color = theme_color)
plt.ylabel('Intensity', fontsize = 8, color = theme_color)
ax.spines['left'].set_color(theme_color)
ax.spines['bottom'].set_color(theme_color)
ax.spines['right'].set_color(theme_color)
ax.spines['top'].set_color(theme_color)
plt.subplots_adjust(left = 0.35, bottom = 0.25, right = 0.95, top = 0.95)
intensity_profile_filename = sample + '_intensity_profile.pdf'
fig.savefig(intensity_profile_filename, dpi = 300, transparent = True)
plt.close()
#####
cell_label = image_seg[1251,895]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_fov_14'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)

plt.imshow(image_seg)
cell_label = image_seg[1543,988]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_14_1_633'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

cell_seg_mask = seg_labeled == 11
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 50)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
intensity_profile = np.sum(straightened_image, axis = 0)
intensity_profile = intensity_profile/np.max(intensity_profile)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(3),cm_to_inches(4.5))
gs = GridSpec(2,1)
ax = plt.subplot(gs[0,0])
ax.imshow(straightened_image, cmap = 'inferno')
plt.axis('off')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
ax = plt.subplot(gs[1,0])
plt.plot(0.016*np.arange(0,intensity_profile.shape[0]), intensity_profile, color = (0,0.5,1))
plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
plt.xlabel(r'Length [$\mu$m]', fontsize = 8, color = theme_color)
plt.ylabel('Intensity', fontsize = 8, color = theme_color)
ax.spines['left'].set_color(theme_color)
ax.spines['bottom'].set_color(theme_color)
ax.spines['right'].set_color(theme_color)
ax.spines['top'].set_color(theme_color)
plt.subplots_adjust(left = 0.35, bottom = 0.25, right = 0.95, top = 0.95)
intensity_profile_filename = sample + '_intensity_profile.pdf'
fig.savefig(intensity_profile_filename, dpi = 300, transparent = True)
plt.close()
######
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_14_3_633'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

cell_label = image_seg[,]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled == 2
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 5)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
intensity_profile = np.sum(straightened_image, axis = 0)
intensity_profile = intensity_profile/np.max(intensity_profile)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(1.5*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(1.5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()
#########
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_14_4_633'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

cell_label = image_seg[819,945]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled == 1
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 50)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
intensity_profile = np.sum(straightened_image, axis = 0)
intensity_profile = intensity_profile/np.max(intensity_profile)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(3),cm_to_inches(4.5))
gs = GridSpec(2,1)
ax = plt.subplot(gs[0,0])
ax.imshow(straightened_image, cmap = 'inferno')
plt.axis('off')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
ax = plt.subplot(gs[1,0])
plt.plot(0.016*np.arange(0,intensity_profile.shape[0]), intensity_profile, color = (0,0.5,1))
plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
plt.xlabel(r'Length [$\mu$m]', fontsize = 8, color = theme_color)
plt.ylabel('Intensity', fontsize = 8, color = theme_color)
ax.spines['left'].set_color(theme_color)
ax.spines['bottom'].set_color(theme_color)
ax.spines['right'].set_color(theme_color)
ax.spines['top'].set_color(theme_color)
plt.subplots_adjust(left = 0.35, bottom = 0.25, right = 0.95, top = 0.95)
intensity_profile_filename = sample + '_intensity_profile.pdf'
fig.savefig(intensity_profile_filename, dpi = 300, transparent = True)
plt.close()

#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_14_13_488'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

cell_label = image_seg[603,1326]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled == 1
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 50)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
intensity_profile = np.sum(straightened_image, axis = 0)
intensity_profile = intensity_profile/np.max(intensity_profile)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(3.3),cm_to_inches(4.5))
gs = GridSpec(2,1)
ax = plt.subplot(gs[0,0])
ax.imshow(straightened_image, cmap = 'inferno')
plt.axis('off')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
ax = plt.subplot(gs[1,0])
plt.plot(0.016*np.arange(0,intensity_profile.shape[0]), intensity_profile, color = (0,0.5,1))
plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
plt.xlabel(r'Length [$\mu$m]', fontsize = 8, color = theme_color)
plt.ylabel('Intensity', fontsize = 8, color = theme_color)
ax.spines['left'].set_color(theme_color)
ax.spines['bottom'].set_color(theme_color)
ax.spines['right'].set_color(theme_color)
ax.spines['top'].set_color(theme_color)
plt.subplots_adjust(left = 0.35, bottom = 0.25, right = 0.95, top = 0.95)
intensity_profile_filename = sample + '_intensity_profile.pdf'
fig.savefig(intensity_profile_filename, dpi = 300, transparent = True)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_14_14_488'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

cell_label = image_seg[1154,1696]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled == 3
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 50)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
intensity_profile = np.sum(straightened_image, axis = 0)
intensity_profile = intensity_profile/np.max(intensity_profile)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(2.8),cm_to_inches(4.5))
gs = GridSpec(2,1)
ax = plt.subplot(gs[0,0])
ax.imshow(straightened_image, cmap = 'inferno')
plt.axis('off')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
ax = plt.subplot(gs[1,0])
plt.plot(0.016*np.arange(0,intensity_profile.shape[0]), intensity_profile, color = (0,0.5,1))
plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
plt.xlabel(r'Length [$\mu$m]', fontsize = 8, color = theme_color)
plt.ylabel('Intensity', fontsize = 8, color = theme_color)
ax.spines['left'].set_color(theme_color)
ax.spines['bottom'].set_color(theme_color)
ax.spines['right'].set_color(theme_color)
ax.spines['top'].set_color(theme_color)
plt.subplots_adjust(left = 0.35, bottom = 0.25, right = 0.95, top = 0.95)
intensity_profile_filename = sample + '_intensity_profile.pdf'
fig.savefig(intensity_profile_filename, dpi = 300, transparent = True)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P12_04_2017/03_16_2020_DSGN0673_P12_04_2017_fov_7'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P12_04_2017/03_16_2020_DSGN0673_P12_04_2017_airy_fov_7_1_561'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

cell_label = image_seg[,]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled > 0
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 50)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
intensity_profile = np.sum(straightened_image, axis = 0)
intensity_profile = intensity_profile/np.max(intensity_profile)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(7),cm_to_inches(4.5))
gs = GridSpec(2,1)
ax = plt.subplot(gs[0,0])
ax.imshow(straightened_image, cmap = 'inferno')
plt.axis('off')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
ax = plt.subplot(gs[1,0])
plt.plot(0.016*np.arange(0,intensity_profile.shape[0]), intensity_profile, color = (0,0.5,1))
plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
plt.xlabel(r'Length [$\mu$m]', fontsize = 8, color = theme_color)
plt.ylabel('Intensity', fontsize = 8, color = theme_color)
ax.spines['left'].set_color(theme_color)
ax.spines['bottom'].set_color(theme_color)
ax.spines['right'].set_color(theme_color)
ax.spines['top'].set_color(theme_color)
plt.subplots_adjust(left = 0.15, bottom = 0.25, right = 0.95, top = 0.95)
intensity_profile_filename = sample + '_intensity_profile.pdf'
fig.savefig(intensity_profile_filename, dpi = 300, transparent = True)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P12_04_2017/03_16_2020_DSGN0673_P12_04_2017_airy_fov_9_2_633'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

cell_label = image_seg[,]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled == 1
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 40)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
intensity_profile = np.sum(straightened_image, axis = 0)
intensity_profile = intensity_profile/np.max(intensity_profile)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(5),cm_to_inches(4.5))
gs = GridSpec(2,1)
ax = plt.subplot(gs[0,0])
ax.imshow(straightened_image, cmap = 'inferno')
plt.axis('off')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
ax = plt.subplot(gs[1,0])
plt.plot(0.016*np.arange(0,intensity_profile.shape[0]), intensity_profile, color = (0,0.5,1))
plt.tick_params(direction = 'in', length = 2, labelsize = 8, colors = theme_color)
plt.xlabel(r'Length [$\mu$m]', fontsize = 8, color = theme_color)
plt.ylabel('Intensity', fontsize = 8, color = theme_color)
ax.spines['left'].set_color(theme_color)
ax.spines['bottom'].set_color(theme_color)
ax.spines['right'].set_color(theme_color)
ax.spines['top'].set_color(theme_color)
plt.subplots_adjust(left = 0.15, bottom = 0.25, right = 0.95, top = 0.95)
intensity_profile_filename = sample + '_intensity_profile.pdf'
fig.savefig(intensity_profile_filename, dpi = 300, transparent = True)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P12_04_2017/03_16_2020_DSGN0673_P12_04_2017_airy_fov_8_1_561'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 2
seg_labeled = skimage.measure.label(seg_mask)

cell_seg_mask = seg_labeled == 1
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
intensity_profile = np.sum(straightened_image, axis = 0)
intensity_profile = intensity_profile/np.max(intensity_profile)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P12_04_2017/03_16_2020_DSGN0673_P12_04_2017_airy_fov_9_1_633'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 1
seg_labeled = skimage.measure.label(seg_mask)

cell_label = image_seg[,]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled == 1
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 40)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()

#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_fov_14'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_14_11_514'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

cell_seg_mask = seg_labeled == 1
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 75)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)

cell_label_1 = image_seg[148,831]
cell_barcode_1 = cell_info.loc[cell_info.label.values == cell_label_1, 'cell_barcode'].values[0]
taxa_name_1 = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode_1, 'sci_name']
cell_spec_1 = cell_info.loc[cell_info.label.values == cell_label_1, ['channel_{}'.format(i) for i in range(63)]].values.transpose()

cell_label_2 = image_seg[141,841]
cell_barcode_2 = cell_info.loc[cell_info.label.values == cell_label_2, 'cell_barcode'].values[0]
taxa_name_2 = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode_2, 'sci_name']
cell_spec_2 = cell_info.loc[cell_info.label.values == cell_label_2, ['channel_{}'.format(i) for i in range(63)]].values.transpose()

cell_seg_confocal_mask = (image_seg == cell_label_1)*1+(image_seg == cell_label_2)*1
cell_seg_confocal_mask_crop = cell_seg_confocal_mask[100:200,800:900]
medial_line_confocal, seg_mask_2_confocal = get_medial_line(cell_seg_confocal_mask_crop, 5, 10, 20)
image_registered_sum_crop = image_registered_sum[100:200,800:900]
straightened_image_confocal = get_straightened_image(image_registered_sum_crop*seg_mask_2_confocal, medial_line_confocal, 100)

confocal_size_row = int((straightened_image.shape[0]*0.016/0.070)/2)
confocal_size_col = int((straightened_image.shape[1]*0.016/0.070)/2)

image_confocal = straightened_image_confocal[50-confocal_size_row:50+confocal_size_row,23-confocal_size_col:23+confocal_size_col]
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*image_confocal.shape[1]/image_confocal.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_confocal, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_confocal_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()


cell_spec_avg = cell_spec_1 + cell_spec_2
cell_spec_avg_norm = cell_spec_avg/np.max(cell_spec_avg)
fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(1.5),cm_to_inches(0.75))
plt.plot(np.arange(0,23), cell_spec_avg_norm[0:23], color = 'limegreen')
plt.plot(np.arange(23,43), cell_spec_avg_norm[23:43], color = 'yellowgreen')
plt.plot(np.arange(43,57), cell_spec_avg_norm[43:57], color = 'darkorange')
plt.plot(np.arange(57,63), cell_spec_avg_norm[57:63], color = 'red')
plt.axis('off')
spec_snippet_filename = '{}_spec.pdf'.format(sample)
plt.subplots_adjust(left = 0, bottom = 0, right = 0.98, top = 0.98)
fig.savefig(spec_snippet_filename, dpi = 300, transparent = True)
plt.close()

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_fov_14'
image_registered = np.load('{}_registered.npy'.format(sample))
image_identification = np.load('{}_identification.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
cell_info = pd.read_csv('{}_cell_information.csv'.format(sample), dtype = {69:str})
image_registered_sum = np.sum(image_registered, axis = 2)

xml = bioformats.get_omexml_metadata('{}_488.czi'.format(sample))
ome = bioformats.OMEXML(xml)
pos_x = ome.image(0).Pixels.Plane().PositionX
pos_y = ome.image(0).Pixels.Plane().PositionY

#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_14_3_633'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

xml = bioformats.get_omexml_metadata('{}_Out.czi'.format(sample))
ome = bioformats.OMEXML(xml)
airy_x = ome.image(0).Pixels.Plane().PositionX
airy_y = ome.image(0).Pixels.Plane().PositionY

delta_pixel_x = (airy_x - pos_x)/(0.07)
delta_pixel_y = (airy_y - pos_y)/(0.07)
center_x = int(1000 + delta_pixel_x)
center_y = int(1000 + delta_pixel_y)
image_registered_channel_sum = np.sum(image_registered[:,:,57:63], axis = 2)
plt.imshow(image_seg[center_x-50:center_x+50,center_y-50:center_y+50])

cell_label = image_seg[503,1095]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled == 2
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 10)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image.shape[1]/straightened_image.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image, cmap = 'inferno')
scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()

#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_14_5_633'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

xml = bioformats.get_omexml_metadata('{}_Out.czi'.format(sample))
ome = bioformats.OMEXML(xml)
airy_x = ome.image(0).Pixels.Plane().PositionX
airy_y = ome.image(0).Pixels.Plane().PositionY

delta_pixel_x = (airy_x - pos_x)/(0.07)
delta_pixel_y = (airy_y - pos_y)/(0.07)
center_x = int(1000 + delta_pixel_x)
center_y = int(1000 + delta_pixel_y)
image_registered_channel_sum = np.sum(image_registered[:,:,57:63], axis = 2)
plt.imshow(image_registered_channel_sum[center_x-50:center_x+50,center_y-50:center_y+50])

cell_label = image_seg[1069,500]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled == 6
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20, 5)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
straightened_image_crop = straightened_image[:,60:]

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image_crop.shape[1]/straightened_image_crop.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image_crop, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_airy_fov_14_8_561'
image = bioformats.load_image('{}_Out.czi'.format(sample))
image = image[:,:,0]
segmentation = KMeans(n_clusters = 3, random_state = 0).fit_predict(image.reshape(np.prod(image.shape), 1)).reshape(image.shape)
seg_mask = segmentation != 0
seg_labeled = skimage.measure.label(seg_mask)

xml = bioformats.get_omexml_metadata('{}_Out.czi'.format(sample))
ome = bioformats.OMEXML(xml)
airy_x = ome.image(0).Pixels.Plane().PositionX
airy_y = ome.image(0).Pixels.Plane().PositionY

delta_pixel_x = (airy_x - pos_x)/(0.07)
delta_pixel_y = (airy_y - pos_y)/(0.07)
center_x = int(1000 + delta_pixel_x)
center_y = int(1000 + delta_pixel_y)
image_registered_channel_sum = np.sum(image_registered[:,:,43:57], axis = 2)
plt.imshow(image_registered_channel_sum[center_x-50:center_x+50,center_y-50:center_y+50])

cell_label = image_seg[964,784]
cell_barcode = cell_info.loc[cell_info.label.values == cell_label, 'cell_barcode'].values[0]
taxa_name = taxa_barcode_sciname.loc[taxa_barcode_sciname.code.values == cell_barcode, 'sci_name']

cell_seg_mask = seg_labeled == 1
medial_line, seg_mask_2 = get_medial_line(cell_seg_mask, 20, 20)
straightened_image = get_straightened_image(image*seg_mask_2, medial_line, 100)
straightened_image_crop = straightened_image[:,50:150]

fig = plt.figure(frameon = False)
fig.set_size_inches(cm_to_inches(0.75*straightened_image_crop.shape[1]/straightened_image_crop.shape[0]),cm_to_inches(0.75))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(straightened_image_crop, cmap = 'inferno')
# scalebar = ScaleBar(0.016, 'um', frameon = True, color = 'white', length_fraction = 0.3, height_fraction = 0.05, box_color = 'black', box_alpha = 0, location = 1, label_formatter = lambda x, y: '')
# plt.gca().add_artist(scalebar)
crop_filename = '{}_straightened.pdf'.format(sample)
fig.savefig(crop_filename, dpi = 300)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_16_2020_DSGN0673_P02_21_2018/03_16_2020_DSGN0673_P02_21_2018_fov_10'
image_registered = np.load('{}_registered.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
image_registered_sum = np.sum(image_registered, axis = 2)
image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_sum_norm[1000:1200,300:500], cmap = 'inferno')
image_filename = '{}_registered_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 2*image_noise_variance)
image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
image_lp = np.nan_to_num(image_lp)
image_lp_min = np.min(image_lp, axis = 3)
image_lp_max = np.max(image_lp, axis = 3)
image_lp_max = image_lp_max - image_lp_min
image_lp = image_lp - image_lp_min[:,:,:,None]
image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
image_lp_rnc = image_lp_rel_norm[:,:,:,5]
image_lprns = np.average(image_lp_rnc, axis = 2)
image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
image_lprn_qcv = np.zeros(image_lprn_uq.shape)
image_lprn_qcv_pre = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq + 1e-8)
image_lprn_qcv[image_lprn_uq > 0] = image_lprn_qcv_pre[image_lprn_uq > 0]
image_final = image_lprns*(1-image_lprn_qcv)
intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
image0 = image_final*(intensity_rough_seg == 0)
image1 = image_final*(intensity_rough_seg == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
intensity_rough_seg_mask = intensity_rough_seg == np.argmax([i0,i1])
image_lprns_rsfsm = skimage.morphology.remove_small_objects(intensity_rough_seg_mask, 10)
image_lprns_rsfsm_bfh = binary_fill_holes(image_lprns_rsfsm)
image_lprns_rsfsmbo = skimage.morphology.binary_opening(image_lprns_rsfsm_bfh)
image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
image0 = image_registered_sum*(image_bkg_filter == 0)
image1 = image_registered_sum*(image_bkg_filter == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
image_bkg_filter_mask = image_bkg_filter == np.argmax([i0,i1])
image_lprns_rsfsm = skimage.morphology.remove_small_objects(intensity_rough_seg_mask, 10)
image_lprns_rsfsm_bfh = binary_fill_holes(image_lprns_rsfsm)
image_lprns_rsfsmbo = skimage.morphology.binary_opening(image_lprns_rsfsm_bfh)
image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
image0 = image_registered_sum*(image_bkg_filter == 0)
image1 = image_registered_sum*(image_bkg_filter == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
image_bkg_filter_mask = image_bkg_filter == np.argmax([i0,i1])
image_watershed_mask = image_lprns_rsfsmbo*image_bkg_filter_mask
image_final_bkg_filtered = image_final*image_bkg_filter_mask
image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
image_watershed_seeds = skimage.measure.label(image_watershed_mask)
image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)

cells = skimage.measure.regionprops(image_seg)
cell_location_info = pd.DataFrame(columns = ['Label', 'X', 'Y'])
cell_location_info.loc[:, 'Label'] = [c.label for c in cells]
cell_location_info.loc[:, 'X'] = np.array([c.centroid[0] for c in cells])
cell_location_info.loc[:, 'Y'] = np.array([c.centroid[1] for c in cells])

fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_final_bkg_filtered[1000:1200,300:500], cmap = 'inferno')
image_filename = '{}_final_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()

colors = cm.get_cmap('tab10')
n = 10
color_list = [colors(i*(1/n)) for i in range(n)]
image_seg_color = color.label2rgb(image_seg, colors = color_list, bg_color = (0,0,0), bg_label = 0)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_seg_color[1000:1200,300:500,:])
image_filename = '{}_seg_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()

colors = cm.get_cmap('tab10')
n = 10
color_list = [colors(i*(1/n)) for i in range(n)]
image_seg_color = color.label2rgb(image_seg, colors = color_list, bg_color = (0,0,0), bg_label = 0)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_seg_color)


from matplotlib.widgets import Slider, Button, RadioButtons

fig = plt.figure()
ax = plt.subplot(111)
axz = plt.axes([0.25, 0.1, 0.65, 0.03])
zslider = Slider(axz, 'z-plane', 0, image_seg_color.shape[2], valinit = int(image_seg_color.shape[2]/2), valstep = 1)
ax.imshow(image_seg_color[200:300,300:400,int(image_seg_color.shape[2]/2)])

def update(val):
    z = zslider.val
    ax.imshow(image_seg_color[200:300,300:400,int(z)])
    fig.canvas.draw_idle()

inc_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
inc_button = Button(inc_ax, 'Increase')

dec_ax = plt.axes([0.6, 0.025, 0.1, 0.04])
dec_button = Button(dec_ax, 'Decrease')

def increase(event):
    z = zslider.val
    ax.imshow(image_seg_color[200:300,300:400,int(z+1)])
    fig.canvas.draw_idle()

def decrease(event):
    z = zslider.val
    ax.imshow(image_seg_color[200:300,300:400,int(z-1)])
    fig.canvas.draw_idle()

inc_button.on_clicked(increase)
dec_button.on_clicked(decrease)
zslider.on_changed(update)



edge_map = skimage.filters.sobel(image_seg > 0)
rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
cells = skimage.measure.regionprops(image_seg)
cell_location_info = pd.DataFrame(columns = ['Label', 'X', 'Y'])
cell_location_info.loc[:, 'Label'] = [c.label for c in cells]
cell_location_info.loc[:, 'X'] = np.array([c.centroid[0] for c in cells])
cell_location_info.loc[:, 'Y'] = np.array([c.centroid[1] for c in cells])
cell_location_info.loc[:, 'degree'] = 0
network = np.zeros(image_seg.shape)


for i in range(len(cells)):
    edges = list(rag.edges(i+1))
    for e in edges:
        node_1 = e[0]
        node_2 = e[1]
        if (node_1 > 0) & (node_2 > 0):
            node_1_x = cell_location_info.loc[cell_location_info.Label.values == node_1, 'X'].values[0]
            node_1_y = cell_location_info.loc[cell_location_info.Label.values == node_1, 'Y'].values[0]
            node_2_x = cell_location_info.loc[cell_location_info.Label.values == node_2, 'X'].values[0]
            node_2_y = cell_location_info.loc[cell_location_info.Label.values == node_2, 'Y'].values[0]
            rr,cc = skimage.draw.line(int(node_1_x), int(node_1_y), int(node_2_x), int(node_2_y))
            network[rr,cc] += 1

for i in range(len(cells)):
    cell_location_info.loc[cell_location_info.Label.values == cells[i].label, 'degree'] = rag.degree(cells[i].label)

cell_location_info['normalized_degree'] = cell_location_info.degree.values/np.max(cell_location_info.degree.values)

network_mask = 0.5*np.ones(network.shape)
network_mask[network > 0] = 0
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(network_mask, cmap = 'bwr', vmin = 0, vmax = 1)
plt.scatter(cell_location_info.Y.values, cell_location_info.X.values, s = 8, c = cell_location_info.normalized_degree.values, cmap = 'hsv')
plt.xlim(300,500)
plt.ylim(1200,1000)
plt.axis('off')
network_filename = '{}_adjacency_graph_example.pdf'.format(sample)
fig.savefig(network_filename, dpi = 300)
plt.close()



#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_15_2020_DSGN0673/03_12_2020_DSGN0673_plaque_02_29_2020_fov_1'
image_registered = np.load('{}_registered.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
image_registered_sum = np.sum(image_registered, axis = 2)
image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_sum_norm[400:800,1000:1400], cmap = 'inferno')
image_filename = '{}_registered_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 2*image_noise_variance)
image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
image_lp = np.nan_to_num(image_lp)
image_lp_min = np.min(image_lp, axis = 3)
image_lp_max = np.max(image_lp, axis = 3)
image_lp_max = image_lp_max - image_lp_min
image_lp = image_lp - image_lp_min[:,:,:,None]
image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
image_lp_rnc = image_lp_rel_norm[:,:,:,5]
image_lprns = np.average(image_lp_rnc, axis = 2)
image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
image_lprn_qcv = np.zeros(image_lprn_uq.shape)
image_lprn_qcv_pre = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq + 1e-8)
image_lprn_qcv[image_lprn_uq > 0] = image_lprn_qcv_pre[image_lprn_uq > 0]
image_final = image_lprns*(1-image_lprn_qcv)
intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
image0 = image_final*(intensity_rough_seg == 0)
image1 = image_final*(intensity_rough_seg == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
intensity_rough_seg_mask = intensity_rough_seg == np.argmax([i0,i1])
image_lprns_rsfsm = skimage.morphology.remove_small_objects(intensity_rough_seg_mask, 10)
image_lprns_rsfsm_bfh = binary_fill_holes(image_lprns_rsfsm)
image_lprns_rsfsmbo = skimage.morphology.binary_opening(image_lprns_rsfsm_bfh)
image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
image0 = image_registered_sum*(image_bkg_filter == 0)
image1 = image_registered_sum*(image_bkg_filter == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
image_bkg_filter_mask = image_bkg_filter == np.argmax([i0,i1])
image_watershed_mask = image_lprns_rsfsmbo*image_bkg_filter_mask
image_final_bkg_filtered = image_final*image_bkg_filter_mask
image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
image_watershed_seeds = skimage.measure.label(image_watershed_mask)
image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)

colors = cm.get_cmap('tab10')
n = 10
color_list = [colors(i*(1/n)) for i in range(n)]
image_seg_color = color.label2rgb(image_seg, colors = color_list, bg_color = (0,0,0), bg_label = 0)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_seg_color[1000:1200,300:500,:])
image_filename = '{}_seg_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()

fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_final_bkg_filtered[1000:1200,300:500], cmap = 'inferno')
image_filename = '{}_final_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()

edge_map = skimage.filters.sobel(image_seg > 0)
rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
cells = skimage.measure.regionprops(image_seg)
cell_location_info = pd.DataFrame(columns = ['Label', 'X', 'Y'])
cell_location_info.loc[:, 'Label'] = [c.label for c in cells]
cell_location_info.loc[:, 'X'] = np.array([c.centroid[0] for c in cells])
cell_location_info.loc[:, 'Y'] = np.array([c.centroid[1] for c in cells])
cell_location_info.loc[:, 'degree'] = 0
network = np.zeros(image_seg.shape)

for i in range(len(cells)):
    edges = list(rag.edges(i+1))
    for e in edges:
        node_1 = e[0]
        node_2 = e[1]
        if (node_1 > 0) & (node_2 > 0):
            node_1_x = cell_location_info.loc[cell_location_info.Label.values == node_1, 'X'].values[0]
            node_1_y = cell_location_info.loc[cell_location_info.Label.values == node_1, 'Y'].values[0]
            node_2_x = cell_location_info.loc[cell_location_info.Label.values == node_2, 'X'].values[0]
            node_2_y = cell_location_info.loc[cell_location_info.Label.values == node_2, 'Y'].values[0]
            rr,cc = skimage.draw.line(int(node_1_x), int(node_1_y), int(node_2_x), int(node_2_y))
            network[rr,cc] += 1

for i in range(len(cells)):
    cell_location_info.loc[cell_location_info.Label.values == cells[i].label, 'degree'] = rag.degree(cells[i].label)

cell_location_info['normalized_degree'] = cell_location_info.degree.values/np.max(cell_location_info.degree.values)

network_mask = 0.5*np.ones(network.shape)
network_mask[network > 0] = 0.1
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(network_mask, cmap = 'bwr', vmin = 0, vmax = 1)
plt.scatter(cell_location_info.Y.values, cell_location_info.X.values, s = 8, c = cell_location_info.normalized_degree.values, cmap = 'hsv')
plt.xlim(300,700)
plt.ylim(1400,1000)
plt.axis('off')
network_filename = '{}_adjacency_graph_example.pdf'.format(sample)
fig.savefig(network_filename, dpi = 300)
plt.close()

#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_15_2020_DSGN0673/03_12_2020_DSGN0673_plaque_02_29_2020_fov_4'
image_registered = np.load('{}_registered.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
image_registered_sum = np.sum(image_registered, axis = 2)
image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_sum_norm[400:800,1000:1400], cmap = 'inferno')
image_filename = '{}_registered_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 2*image_noise_variance)
image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
image_lp = np.nan_to_num(image_lp)
image_lp_min = np.min(image_lp, axis = 3)
image_lp_max = np.max(image_lp, axis = 3)
image_lp_max = image_lp_max - image_lp_min
image_lp = image_lp - image_lp_min[:,:,:,None]
image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
image_lp_rnc = image_lp_rel_norm[:,:,:,5]
image_lprns = np.average(image_lp_rnc, axis = 2)
image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
image_lprn_qcv = np.zeros(image_lprn_uq.shape)
image_lprn_qcv_pre = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq + 1e-8)
image_lprn_qcv[image_lprn_uq > 0] = image_lprn_qcv_pre[image_lprn_uq > 0]
image_final = image_lprns*(1-image_lprn_qcv)
intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
image0 = image_final*(intensity_rough_seg == 0)
image1 = image_final*(intensity_rough_seg == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
intensity_rough_seg_mask = intensity_rough_seg == np.argmax([i0,i1])
image_lprns_rsfsm = skimage.morphology.remove_small_objects(intensity_rough_seg_mask, 10)
image_lprns_rsfsm_bfh = binary_fill_holes(image_lprns_rsfsm)
image_lprns_rsfsmbo = skimage.morphology.binary_opening(image_lprns_rsfsm_bfh)
image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
image0 = image_registered_sum*(image_bkg_filter == 0)
image1 = image_registered_sum*(image_bkg_filter == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
image_bkg_filter_mask = image_bkg_filter == np.argmax([i0,i1])
image_final_bkg_filtered = image_final*image_bkg_filter_mask
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_final_bkg_filtered[1000:1200,300:500], cmap = 'inferno')
image_filename = '{}_final_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
#####
sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/03_15_2020_DSGN0673/03_12_2020_DSGN0673_plaque_02_29_2020_fov_13'
image_registered = np.load('{}_registered.npy'.format(sample))
image_seg = np.load('{}_seg.npy'.format(sample))
image_registered_sum = np.sum(image_registered, axis = 2)
image_registered_sum_norm = image_registered_sum/np.max(image_registered_sum)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_registered_sum_norm[200:600,0:400], cmap = 'inferno')
image_filename = '{}_registered_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()
image_noise_variance = skimage.restoration.estimate_sigma(image_registered_sum_norm)
image_registered_sum_nl = skimage.restoration.denoise_nl_means(image_registered_sum_norm, h = 2*image_noise_variance)
image_padded = skimage.util.pad(image_registered_sum_nl, 5, mode = 'edge')
image_lp = line_profile_2d_v2(image_padded.astype(np.float64), 11, 9)
image_lp = np.nan_to_num(image_lp)
image_lp_min = np.min(image_lp, axis = 3)
image_lp_max = np.max(image_lp, axis = 3)
image_lp_max = image_lp_max - image_lp_min
image_lp = image_lp - image_lp_min[:,:,:,None]
image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
image_lp_rnc = image_lp_rel_norm[:,:,:,5]
image_lprns = np.average(image_lp_rnc, axis = 2)
image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
image_lprn_qcv = np.zeros(image_lprn_uq.shape)
image_lprn_qcv_pre = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq + 1e-8)
image_lprn_qcv[image_lprn_uq > 0] = image_lprn_qcv_pre[image_lprn_uq > 0]
image_final = image_lprns*(1-image_lprn_qcv)
intensity_rough_seg = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_final.reshape(np.prod(image_final.shape), 1)).reshape(image_final.shape)
image0 = image_final*(intensity_rough_seg == 0)
image1 = image_final*(intensity_rough_seg == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
intensity_rough_seg_mask = intensity_rough_seg == np.argmax([i0,i1])
image_lprns_rsfsm = skimage.morphology.remove_small_objects(intensity_rough_seg_mask, 10)
image_lprns_rsfsm_bfh = binary_fill_holes(image_lprns_rsfsm)
image_lprns_rsfsmbo = skimage.morphology.binary_opening(image_lprns_rsfsm_bfh)
image_registered_sum_nl_log = np.log10(image_registered_sum_nl)
image_bkg_filter = KMeans(n_clusters = 2, random_state = 0).fit_predict(image_registered_sum_nl_log.reshape(np.prod(image_registered_sum_nl_log.shape), 1)).reshape(image_registered_sum_nl_log.shape)
image0 = image_registered_sum*(image_bkg_filter == 0)
image1 = image_registered_sum*(image_bkg_filter == 1)
i0 = np.average(image0[image0 > 0])
i1 = np.average(image1[image1 > 0])
image_bkg_filter_mask = image_bkg_filter == np.argmax([i0,i1])
image_watershed_mask = image_lprns_rsfsmbo*image_bkg_filter_mask
image_final_bkg_filtered = image_final*image_bkg_filter_mask
image_sum_bkg_filtered = image_registered_sum*image_bkg_filter_mask
image_watershed_mask = image_watershed_mask*image_bkg_filter_mask
image_watershed_mask = skimage.morphology.remove_small_objects(image_watershed_mask, 10)
image_watershed_seeds = skimage.measure.label(image_watershed_mask)
image_watershed_mask_bkg_filtered = intensity_rough_seg_mask*image_bkg_filter_mask
image_seg = skimage.morphology.watershed(-image_final_bkg_filtered, image_watershed_seeds, mask = image_watershed_mask_bkg_filtered)
adjacency_seg = skimage.morphology.watershed(-image_sum_bkg_filtered, image_watershed_seeds, mask = image_bkg_filter_mask)

fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_final_bkg_filtered[200:600,0:400], cmap = 'inferno')
image_filename = '{}_final_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()

colors = cm.get_cmap('tab10')
n = 10
color_list = [colors(i*(1/n)) for i in range(n)]
image_seg_color = color.label2rgb(image_seg, colors = color_list, bg_color = (0,0,0), bg_label = 0)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_seg_color[200:600,0:400,:])
image_filename = '{}_seg_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()

edge_map = skimage.filters.sobel(image_seg > 0)
rag = skimage.future.graph.rag_boundary(adjacency_seg, edge_map)
cells = skimage.measure.regionprops(image_seg)
cell_location_info = pd.DataFrame(columns = ['Label', 'X', 'Y'])
cell_location_info.loc[:, 'Label'] = [c.label for c in cells]
cell_location_info.loc[:, 'X'] = np.array([c.centroid[0] for c in cells])
cell_location_info.loc[:, 'Y'] = np.array([c.centroid[1] for c in cells])
cell_location_info.loc[:, 'degree'] = 0

network = np.zeros(image_seg.shape)

for i in range(len(cells)):
    edges = list(rag.edges(i+1))
    for e in edges:
        node_1 = e[0]
        node_2 = e[1]
        if (node_1 > 0) & (node_2 > 0):
            node_1_x = cell_location_info.loc[cell_location_info.Label.values == node_1, 'X'].values[0]
            node_1_y = cell_location_info.loc[cell_location_info.Label.values == node_1, 'Y'].values[0]
            node_2_x = cell_location_info.loc[cell_location_info.Label.values == node_2, 'X'].values[0]
            node_2_y = cell_location_info.loc[cell_location_info.Label.values == node_2, 'Y'].values[0]
            rr,cc = skimage.draw.line(int(node_1_x), int(node_1_y), int(node_2_x), int(node_2_y))
            network[rr,cc] += 1

for i in range(len(cells)):
    cell_location_info.loc[cell_location_info.Label.values == cells[i].label, 'degree'] = rag.degree(cells[i].label)

cell_location_info['normalized_degree'] = cell_location_info.degree.values/np.max(cell_location_info.degree.values)

network_mask = 0.5*np.ones(network.shape)
network_mask[network > 0] = 0.1
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(network_mask, cmap = 'bwr', vmin = 0, vmax = 1)
plt.scatter(cell_location_info.Y.values, cell_location_info.X.values, s = 8, c = cell_location_info.normalized_degree.values, cmap = 'hsv')
plt.xlim(0,400)
plt.ylim(600,200)
plt.axis('off')
network_filename = '{}_adjacency_graph_example.pdf'.format(sample)
fig.savefig(network_filename, dpi = 300)
plt.close()
#####

sample = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/01_31_2020_DSGN0641/01_31_2020_DSGN0641_285_fov_1'
image_seg = np.load('{}_seg.npy'.format(sample))
colors = cm.get_cmap('tab10')
n = 10
color_list = [colors(i*(1/n)) for i in range(n)]
image_seg_color = color.label2rgb(image_seg, colors = color_list, bg_color = (0,0,0), bg_label = 0)
fig = plt.figure()
fig.set_size_inches(cm_to_inches(5), cm_to_inches(5))
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)
ax.imshow(image_seg_color)
image_filename = '{}_seg_example.pdf'.format(sample)
fig.savefig(image_filename, dpi = 300)
plt.close()


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
