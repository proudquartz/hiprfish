"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import numpy as np
import pandas as pd

reference_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_reference_02_20_2019'
fret_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/hiprfish_fret'
sam_tab_bkg_filename = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging/images_table_1023_reference_bkg.csv'
data_folder = '/workdir/hs673/Runs/V1/Samples/HIPRFISH_imaging'

sam_tab_bkg_intensity = pd.read_csv(sam_tab_bkg_filename)
sample = sam_tab_bkg_intensity.loc[i,'SAMPLE']
image_name = sam_tab_bkg_intensity.loc[i,'IMAGES']
enc = re.sub('enc_', '', re.search('enc_[0-9]*', image_name).group(0))
barcode = re.sub('0b', '', format(int(enc), '#0' + str(nbit+2) + 'b'))
# cell_info = pd.read_csv('{}/{}/{}_avgint_ids.csv'.format(data_folder, sample, image_name), header = None, dtype = {100:str})
cell_info = pd.read_csv('{}/{}/{}_avgint_ids_replicate_{}.csv'.format(data_folder, sample, image_name, 0), header = None, dtype = {100:str})
