# Libraries
import os
import glob
from SPAD512S import SPAD512S
import numpy as np
import time
import matplotlib.pyplot  as plt
from scipy.stats import linregress


input("Here is a small example of how to use the function from SPAD512S library."
      "We'll take a look at the impact of the noise and dead pixel calibrations."
      " To do so, please delete the noisy_pixels and dead_pixels .txt files." 
      " WARNING : please put an objective on your camera to avoid getting too much light."
      " Once it's done, launch the software, wait for the camera to be ready, and press Enter.")

port = 9999 # Check the command Server in the setting tab of the software and change it if necessary
SPAD1 = SPAD512S(port)

# Get informations on the system used
info = SPAD1.get_info()
print("\nGeneral informations of the camera :")
print(info)
temp = SPAD1.get_temps() # Current temperatures of FPGAs, PCB and Chip
print("\nCurrent temperatures of FPGAs, PCB and Chip :")
print(temp)
freq = SPAD1.get_freq() # Operating frequencies (Laser and frame)
print("\nOperating frequencies (Laser and frame) :")
print(freq)

# # # Set the voltage to the maximum value
Vex = 7
SPAD1.set_Vex(Vex)



input("\nYou can put a bit of light in front of the detector before taking some images. "
      "Once it's done, press Enter to proceed.")

# Examples of how to take images with the function get_intensity
overlap = 1
timeout = 0
pileup = 1
im_width = 512
bitdepth = [4, 6, 7, 8, 9, 10, 11, 12]
intTime = np.array([10, 2, 4, 8, 16, 32, 64, 128]) # Integration time for the different bitdepth
nb_points = 10
iterations = 10
string_comment = ""
counter = 0
rowCol = 512 #Number of rows/columns

for i in range(len(bitdepth)):
    counts = SPAD1.get_intensity(iterations, intTime[i], bitdepth[i], overlap, timeout, pileup, im_width)
    if bitdepth[i] in (1,4):
        unit = "us"
        factor_unit = 1e-6
    else : 
        unit = "ms"
        factor_unit = 1e-3
        
    # Show the image after calibration
    plt.figure()
    plt.imshow(np.mean(counts, axis=2), cmap='gray') # Here we take the mean of counts over the number of iterations
    plt.colorbar()
    plt.title(f"{bitdepth[i]}-bit image with {intTime[i]}{unit} integration time.")
    plt.show()
    

    
    
input("\nPlease put the cap on the objective in order to continue with the calibration "
       "of the noise and the dead pixels. Once it's done, press Enter to proceed.")

# DCR graph before calibration
counts_DCR = SPAD1.get_intensity(iterations, intTime[i], bitdepth[i], overlap, timeout, pileup, im_width)
DCR_avg = np.average(counts_DCR, 2)
DCR_1D = np.reshape(DCR_avg, (1, rowCol**2))
DCR_sorted_rate = np.sort(DCR_1D)/(intTime[i]*factor_unit)



plt.figure()
plt.rcParams['font.size'] = '15'
plt.rcParams["font.family"] = "Arial"
xAxis = np.linspace(0, 100, rowCol**2 )
plt.plot(xAxis, DCR_sorted_rate[0,:], color=(0, 0.4470, 0.7410), linewidth=2.5)
plt.grid("ON", 'both')
plt.xlabel('Percentage [%]')
plt.ylabel('DCR [cps]')
plt.title(f"DCR values before calibration")
plt.show



# Do the calibration of the camera
msg = SPAD1.calib_noise()
print(msg)
msg = SPAD1.calib_dead()
print(msg)


# DCR graph after calibration
counts_DCR = SPAD1.get_intensity(iterations, intTime[i], bitdepth[i], overlap, timeout, pileup, im_width)
DCR_avg = np.average(counts_DCR, 2)
DCR_1D = np.reshape(DCR_avg, (1, rowCol**2))
DCR_sorted_rate = np.sort(DCR_1D)/(intTime[i]*factor_unit)

plt.figure()
plt.rcParams['font.size'] = '15'
plt.rcParams["font.family"] = "Arial"
xAxis = np.linspace(0, 100, rowCol**2 )
plt.plot(xAxis, DCR_sorted_rate[0,:], color=(0, 0.4470, 0.7410), linewidth=2.5)
plt.grid("ON", 'both')
plt.xlabel('Percentage [%]')
plt.ylabel('DCR [cps]')
plt.title(f"DCR values after calibration")
plt.show

input("\nNow you can remove the cap from the objective before taking some  more images. "
      "Once it's done, press Enter to proceed.")



# Take images with the function get_intensity after calibration

for i in range(len(bitdepth)):
    counts = SPAD1.get_intensity(iterations, intTime[i], bitdepth[i], overlap, timeout, pileup, im_width)
    if bitdepth[i] in (1,4):
        unit = "us"
        factor_unit = 1e-6
    else : 
        unit = "ms"
        factor_unit = 1e-3
        
    # Show the image after calibration
    plt.figure()
    plt.imshow(np.mean(counts, axis=2), cmap='gray') # Here we take the mean of counts over the number of iterations
    plt.colorbar()
    plt.title(f"{bitdepth[i]}-bit image with {intTime[i]}{unit} integration time.")
    plt.show()
    