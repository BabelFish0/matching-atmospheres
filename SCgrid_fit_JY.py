# SCgrid_fit.py
# Goyal Self Consistent Grid fit
# Written by J. Young based on PSgrid_fit.py by H. R. Wakeford.
# email: stellarplanet@gmail.com

import os
import sys
import numpy as np
import scipy.optimize as opt
import scipy.stats 

import glob 

import matplotlib
import matplotlib.pyplot as plt


'''
What needs to be read in 
data file name and location (data needs to be in rp/rs^2 and microns)
name of the planet in the format used in the model grid files
location of the planet specific grid files
output folder location
'''

# --- set up the data ---
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

def model_to_data(p0, data, data_err, model):
	'''
	This function is used to minimise systematic error in the observation.

	-----
	Return the quality of fit between the model transit depths and the observed transit depths.

	`p0` is a constant added to the model values. The optimiser calls this function to find the `p0` that minimises the fit error.
	'''
	model_fit = model + p0[0]
	return np.sum((model_fit-data)**2/data_err**2)

def get_bin_widths(values, val_min=None, val_max=None):
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
	'''Match a model parameter value to the correct dx value which has been produced by `get_bin_widths()`'''
	unique_values = np.sort(np.unique(values).astype(float))
	return np.trim_zeros(np.where(unique_values==values[i], bin_widths, [0]*len(unique_values)))[0]

# For JY reference -> pt-eqpt_PlanetName_Recirculationfactor_log(Metallicity)_C/ORatio_model.txt.gz
planet_recirc = []
planet_metal = []
planet_co = []
model_chi = np.array([])
'''chi squared'''
model_alt = np.array([])
'''added constant'''


'''
Iterate through every file in the grid. Record the model parameters and compare it to the observed data file. Record the fit data.
'''

filenum=0
for model_file in full_grid:
	split_file = os.path.basename(model_file).split('_')
	planet_recirc.append(split_file[2])
	planet_metal.append(split_file[3])
	planet_co.append(split_file[4])
	print('--------------------------------------|', filenum)
	print('Recirc= ', planet_recirc[-1], 'M/H= ', planet_metal[-1], ' C/O= ', planet_co[-1])

	model = np.loadtxt(model_file, dtype=float)
	planet_model_wav = model[:,0]
	planet_model = model[:,1]

	# temp arrays for binning
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
planet_recirc_dx = get_bin_widths(np.sort(np.unique(planet_recirc))) #find dx values for the model parameters using get_bin_widths
planet_metal_dx = get_bin_widths(np.sort(np.unique(planet_metal)))
planet_co_dx = get_bin_widths(np.sort(np.unique(planet_co)))
print(np.unique(planet_metal))
#print("dx values: ", planet_recirc_dx, planet_metal_dx, planet_co_dx)

# --- probabilites --- 
print('--------------------------------------')
max_likelihood = np.amax(model_chi * (-0.5))

a = 0
for i, chi in np.ndenumerate(model_chi):
	i=i[0]
	model_evidence = - (chi / 2.0)
	beta_value = model_evidence - max_likelihood
	tot_evidence = np.exp(beta_value) * bin_width(i, planet_recirc_dx, planet_recirc) * bin_width(i, planet_metal_dx, planet_metal) * bin_width(i, planet_co_dx, planet_co)
	a += tot_evidence

#print(a)
log_norm_grid = np.log10(a) + max_likelihood

norm_prob_density = [np.exp(-0.5 * chi - log_norm_grid) for chi in model_chi]
norm_prob = [norm_prob_density[i] * bin_width(i, planet_recirc_dx, planet_recirc) * bin_width(i, planet_metal_dx, planet_metal) * bin_width(i, planet_co_dx, planet_co) for i in range(len(norm_prob_density))]

min_index = np.argmin(norm_prob_density)
max_index = np.argmax(norm_prob_density)
print('MIN: Recirc=', planet_recirc[min_index], ' M/H=', planet_metal[min_index], ' C/O=', planet_co[min_index], np.amin(norm_prob), '\nMAX: Recirc=', planet_recirc[max_index], ' M/H=', planet_metal[max_index], ' C/O=', planet_co[max_index], np.amax(norm_prob))
print(np.sum(norm_prob))

# --- plot & save ---

# find models >=0.1 confidence
delta_chi2_max = 100
likely_models = np.array([[], []])
num_likely=-1
for i in np.where(model_chi - np.amin(model_chi)<=delta_chi2_max)[0]:
	num_likely += 1
	grid_point = os.path.join(model_files_loc, f'trans-eqpt_{planet_name}_{planet_recirc[i]:.2f}_{planet_metal[i]:+.1f}_{planet_co[i]:.2f}_model.txt.gz')
	model = np.loadtxt(grid_point, dtype=float)
	likely_models = np.append(likely_models, [model[:,0], model[:,1]+model_alt[i]], axis=1)
	
# Find file corresponding to max(norm_prob_density)
grid_point = os.path.join(model_files_loc, f'trans-eqpt_{planet_name}_{planet_recirc[max_index]:.2f}_{planet_metal[max_index]:+.1f}_{planet_co[max_index]:.2f}_model.txt.gz')
model = np.loadtxt(grid_point, dtype=float) 
model_wavelengths = model[:,0]
model_transit_depths = model[:,1]

fig, ax = plt.subplots(figsize=(14, 6))

plt.plot(likely_models[0,:], likely_models[1,:], color='C7', lw=0.5)
plt.plot(model_wavelengths,np.array(model_transit_depths)+model_alt[max_index], color='C1')
plt.errorbar(data_wav, data_depth, xerr=data_waverr, yerr=data_deptherr, ls='None', color='C0')

plt.xlim(0.3,5.1)
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.xscale('log')
majorLocator = matplotlib.pyplot.FixedLocator([0.2,0.6,1.0,1.4,2,3,5,8,12])
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Transit Depth')
plt.legend([f'{num_likely} additional models within $\Delta\chi^2$ {delta_chi2_max}', f'trans-eqpt_{planet_name}_{planet_recirc[max_index]:.2f}_{planet_metal[max_index]:+.1f}_{planet_co[max_index]:.2f}_model.txt.gz', 'observed'])
plt.title(planet_name, loc='left', weight='bold')
plt.show()

# --- junk for now v ---
filename = 'test.npz'

np.savez(filename, model_chi=model_chi, model_alt=model_alt, planet_co=planet_co, planet_metal=planet_metal, planet_recirc=planet_recirc, planet_co_dx=planet_co_dx, planet_metal_dx=planet_metal_dx, planet_recirc_dx=planet_recirc_dx, planet_model=planet_model, planet_model_wav=planet_model_wav, planet_bin_model=planet_bin_model, data_wav=data_wav, data_waverr=data_waverr, data_depth=data_depth, data_deptherr=data_deptherr, norm_prob=norm_prob, norm_prob_density=norm_prob_density)