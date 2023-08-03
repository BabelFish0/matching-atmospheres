import numpy as np
import matplotlib
from pylab import *
import gzip

### User Input Parameters
enter_file_location = '/Users/jude/Documents/School/Upper_School/Summer23/matching-atmospheres'
enter_file_name = 'HD209458b_transmission_Sing2016.txt'
#######


fin = open(enter_file_location + '/' + enter_file_name, 'rb')
planet_para = np.genfromtxt(fin, delimiter = '', dtype='float', usecols=(0,2))

#rcParams['figure.figsize'] = (25.0, 15.0)
lines = semilogx(planet_para[:,0],planet_para[:,1])
 
plt.setp(lines, linestyle='-',linewidth=3.0,color='b') 

xlabel('Wavelength ($\mu$m)',fontsize=18,fontweight='bold')
ylabel('R$_p$$^2$/R$_s$$^2$',fontsize=18,fontweight='bold')
xlim(0.2,12)

tick_params(axis='x',labelsize='16')
tick_params(axis='y',labelsize='16')

# #Set xaxis ticker format
ax = gca()
majorLocator = FixedLocator([0.2,0.6,1.0,1.4,2,3,5,8,12])
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
title('Transmission Spectrum',fontsize=15,fontweight='bold')
show()
