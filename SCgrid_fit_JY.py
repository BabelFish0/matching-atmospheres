# PSgrid_fit.py
# Goyal Planet Specific Grid fit
#  Written by H.R. Wakeford
# email: stellarplanet@gmail.com

import os
import sys
import numpy as np
import scipy.optimize as opt
import scipy.stats 

import glob 

from matplotlib import gridspec, rc, font_manager
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as grid


def model_to_data(p0, data, data_err, model):
	model_fit = model + p0[0]

	return np.sum((model_fit-data)**2/data_err**2)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


'''
What needs to be read in 
data file name and location (data needs to be in rp/rs^2 and microns)
name of the planet in the format used in the model grid files
location of the planet specific grid files
output folder location

'''
# Set up the data
data_file = '/Users/jude/Documents/School/Upper_School/Summer23/matching-atmospheres/HD209458b_transmission_Sing2016.txt'
data = np.loadtxt(data_file, dtype=float, comments='#')
data_wav = data[:,0]
data_waverr = data[:,1]
data_depth = data[:,2]
data_deptherr = data[:,3]

ntransmission = len(data_wav)
wav_low = data_wav-(data_waverr/2.0)
wav_high = data_wav+(data_waverr/2.0)

# User input folder location
grid_folder = 'testgrid_sc'
# User input planet name
planet_name = 'HD-209458'

model_files_loc = os.path.join(grid_folder,planet_name)
full_grid = glob.glob(model_files_loc+'/*.txt.gz')
print('grid loc: ', model_files_loc)
ngrid = len(full_grid)
print(ngrid)

# For JY reference -> pt-eqpt_PlanetName_Recirculationfactor_log(Metallicity)_C/ORatio_model.txt.gz
planet_recirc = []
planet_metal = []
planet_co = []

def get_bin_widths(values, val_min=None, val_max=None): #JY Added general func for above
    '''
    Find the difference between the midpoints of each consecutive value in `values`. If no upper or lower bound is given, the midpoint between element 0 and 1 will be assumed symmetrical. This is also true for element -1 and -2.

    -----
    ## Inputs
    - `values`: 1D ndarray of values of which the 'width' is to be determined.
    - `val_min`: lower limit - if the symmetric midpoint for the first or last element is less than this, `val_min` will be used instead to find the width.
    - `val_max`: upper limit - if the symmetric midpoint for the first or last element is greater than this, `val_max` will be used instead to find the width.
    -----
    ## Returns
    - `dx_values`: 1D ndarray of widths for every element in `values`.
    '''
    midpoints = .5 * (values[1:] + values[:-1])
    midpoints = np.insert(midpoints, 0, 2*values[0]-midpoints[0])
    midpoints = np.append(midpoints, 2*values[-1]-midpoints[-1])
    if val_min != None or val_max != None:
        midpoints = np.clip(midpoints, val_min, val_max)
    dx_values = midpoints[1:] - midpoints[:-1]
    return dx_values

def bin_width(i, bin_widths, values):
	unique_values = np.sort(np.unique(values).astype(float))
	return np.trim_zeros(np.where(unique_values==values[i], bin_widths, [0]*len(unique_values)))[0]

model_chi = np.array([])
model_alt = np.array([])

filenum=0
for model_file in full_grid:
	split_file = os.path.basename(model_file).split('_')
	planet_recirc.append(split_file[2])
	planet_metal.append(split_file[3])
	planet_co.append(split_file[4])
	print('--------------------------------------|', filenum)
	
	print(' Recirc= ', planet_recirc[-1], '[M/H]= ', planet_metal[-1], ', C/O= ', planet_co[-1])

	grid_point = os.path.join(model_files_loc,'trans-eqpt_'+planet_name+'_'+planet_recirc[-1]+'_'+planet_metal[-1]+'_'+planet_co[-1]+'_model.txt.gz')

	model = np.loadtxt(grid_point, dtype=float)
	planet_model_wav = model[:,0]
	planet_model = model[:,1]

	# Temp arrays for binning
	mod_wav = model[:,0]
	mod_depth = model[:,1]
	# now need to bin the model to the transmission spectrum of the planet
	planet_bin_model = []
	for i in range(0, ntransmission):
		bin_range = np.where((mod_wav < wav_high[i]) & (mod_wav > wav_low[i]))# [0] <- JY removed [0], maybe not necessary due to normalisation later??
		planet_bin_model.append(np.mean(mod_depth[bin_range]))

	# Temp array for use in function
	model_binned = np.array(planet_bin_model)

	p0 = [0.01]
	model_fit_info = opt.minimize(model_to_data, p0, (data_depth, data_deptherr, model_binned), method='L-BFGS-B')
	model_alt = np.append(model_alt, model_fit_info.x[0])
	model_chi = np.append(model_chi, model_fit_info.fun)

	filenum += 1

planet_recirc = np.array(planet_recirc).astype(float)
planet_metal = np.array(planet_metal).astype(float)
planet_co = np.array(planet_co).astype(float)
planet_recirc_dx = get_bin_widths(np.sort(np.unique(planet_recirc)))
planet_metal_dx = get_bin_widths(np.sort(np.unique(planet_metal)))
planet_co_dx = get_bin_widths(np.sort(np.unique(planet_co)))

print(planet_recirc_dx, planet_metal_dx, planet_co_dx)

print('--------------------------------------')
max_liklihood = np.amax(model_chi * (-0.5))

a = 0
for i, chi in np.ndenumerate(model_chi):
	i=i[0]
	model_evidence = - (chi / 2.0)
	beta_value = model_evidence - max_liklihood
	tot_evidence = np.exp(beta_value) * bin_width(i, planet_recirc_dx, planet_recirc) * bin_width(i, planet_metal_dx, planet_metal) * bin_width(i, planet_co_dx, planet_co)
	a += tot_evidence

print(a)
log_norm_grid = np.log10(a) + max_liklihood
				
norm_prob_density = [np.exp(-0.5 * chi - log_norm_grid) for chi in model_chi]
norm_prob = [norm_prob_density[i] * bin_width(i, planet_recirc_dx, planet_recirc) * bin_width(i, planet_metal_dx, planet_metal) * bin_width(i, planet_co_dx, planet_co) for i in range(len(norm_prob_density))]

print(np.amin(norm_prob_density), np.amax(norm_prob_density))
print(np.sum(norm_prob))

filename = 'test.npz'

np.savez(filename, model_chi=model_chi, model_alt=model_alt, planet_co=planet_co, planet_metal=planet_metal, planet_recirc=planet_recirc, planet_co_dx=planet_co_dx, planet_metal_dx=planet_metal_dx, planet_recirc_dx=planet_recirc_dx, planet_model=planet_model, planet_model_wav=planet_model_wav, planet_bin_model=planet_bin_model, data_wav=data_wav, data_waverr=data_waverr, data_depth=data_depth, data_deptherr=data_deptherr, norm_prob=norm_prob, norm_prob_density=norm_prob_density)










