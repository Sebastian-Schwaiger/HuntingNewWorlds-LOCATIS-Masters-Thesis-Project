#! /usr/bin/python
import operator
import sys
import os
from xml.parsers.expat import model
import certifi


# Fix SSL certificate issues for model downloads
os.environ['SSL_CERT_FILE'] = certifi.where()

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no GUI windows
import matplotlib.pyplot as plt
import time
import math
import random
import copy
from astropy.io import fits
from astropy.time import Time
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection
from LOCATIS_individual_column_plot import plot_newcolumn_subplots
import madys
from madys import IsochroneGrid
from madys import *


# --- ADDED: Import mass2mag from the other directory ---
import sys
# Add the path to the folder containing the 'mass2mag' package
sys.path.append('/Users/seschwaiger/Desktop/Master_Thesis_Sebastian/')
from mass2mag.mag2mass_OO import convert_mass2mag

# --- ADDED: Stellar Spectrum Configuration ---
USE_STELLAR_SPECTRUM_FILE = True
STELLAR_SPECTRUM_DIR = 'BT-NextGen/'
# Path to Vega spectrum
VEGA_SPECTRUM_FILE = 'BT-NextGen/VegaA/bt-nextgen_VegaA_range=(0.2, 20.0)µm_res=2000.fits'


###############################################################
# LOCATIS
#
# 2025.03.05 -> -New METIS detectability curves (from astropy.io import fits needed for it)
# 2025.03.01 -> -Added Xcoord and Ycoord as outputs of the function orbit_evolution()
#	         -New detectability plot x vs. y with the 2D representations of their orbits
# 2024.07.28 -> -Cleaned a bit the code. Now the functions get_detectability()
# 		and minimum_contrast() are the ones encapsulating
#		all the instrument-related detectability criteria.
#		  Modify these functions to play around with the
#		  instrument sensitivity.
#		-Still built to read the orbital parameters of
#		known exoplanets from the NASA Exoplanet Archive. 
#		  The read_table_exoplanets_NASAArchive function
#		  can be adapted or substituted to change the input
#		  orbital parameters of whatever planet one wants
#		  to simulate (either real or synthetic).
#
# 2022.02.07 -> Updated to use the new data columns in the NASA
# Exoplanet Archive after the Confirmed Exoplanets table was 
# deprecated in April 2021.
#
# 2020.06.26 -> First version
# 
# Oscar Carrion-Gonzalez
###############################################################


def orbperiod_2_orbsemimajorax(orbperiod, Mstar, mplanet, count):
	#For those exoplanets without semimajor axis in the catalog

	if mplanet != '' and Mstar != '':			#We know both the planet and stellar mass
		orbperiod = float(orbperiod)*24.*3600.	#s
		Mstar = float(Mstar)*Msun		#kg
		mplanet = float(mplanet)*Mjup		#kg

		a_meter = np.cbrt(((orbperiod)**2.*G_constant*(Mstar+mplanet))/(4.*np.pi**2.))
		a_AU = a_meter/AU_constant

		return a_AU, count, ''

	elif Mstar != '' and mplanet == '' and orbperiod != '':	#We know stellar mass and the orbital period of the planet, but not the planet mass
		orbperiod = float(orbperiod)*24.*3600.	#s
		Mstar = float(Mstar)*Msun		#kg

		a_meter = np.cbrt(((orbperiod)**2.*G_constant*(Mstar))/(4.*np.pi**2.))
		a_AU = a_meter/AU_constant

		return a_AU, count, '*'

	else:
		count = count + 1
		return '', count, ''



def orbsemimajorax_2_orbperiod(orbsmaxis, Mstar, mplanet, count):
	#For those exoplanets without the orbital period in the catalog

	if mplanet != '' and Mstar != '':
		a_meter = float(orbsmaxis)*AU_constant	#m
		Mstar = float(Mstar)*Msun		#kg
		mplanet = float(mplanet)*Mjup		#kg

		orbper_s = np.sqrt(((a_meter)**3.*4.*np.pi**2.)/(G_constant*(Mstar+mplanet)))
		orbper = orbper_s/(24.*3600.)		#days

		return orbper, count, ''

	elif Mstar != '' and mplanet == '' and orbsmaxis != '':
		a_meter = float(orbsmaxis)*AU_constant	#m
		Mstar = float(Mstar)*Msun		#kg

		orbper_s = np.sqrt(((a_meter)**3.*4.*np.pi**2.)/(G_constant*Mstar))
		orbper = orbper_s/(24.*3600.)		#days

		return orbper, count, '*'

	else:
		count = count + 1
		return '', count, ''


def read_table_exoplanets_NASAArchive(filetab):
	#Reads the table of the NASA exoplanet archive https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS or https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars
	
	infile = open(filetab, 'r')

	#First ruling out the lines that begin with #
	content = infile.readlines()
	count = 0
	for i in range(len(content)):
		num = content[i].split()
		if num[0] == "#":
			count = count+1
	for i in range(0, count, 1):
		infile.readline()
	infile.readline()

	#Now, reading the useful data
	params_arrs = []
	dictio_setup = {}	#This will be the dictio initialized with all the parameters. For each planet, I will copy it 

	dictionary = {}
	for i in range(count, len(content), 1):
		num = content[i].split('\t')
		if i == count:
			params_labels = num		#First line, with all the names of the variables
			for param in params_labels:
				dictio_setup[param] = ''	#I initialize the dictionary with all the variables
		else:
			dictioi = dictio_setup.copy()
			for j in range(len(num)):	
				dictioi[params_labels[j]]=num[j]	#This builds a dictionary for each particular planet
			dictionary[dictioi['pl_name']] = dictioi.copy()	#This ,,dictionary,, contains, for each planet, a dictionary ,,dictioi,, with all the parameters 

	return dictionary


def compute_maxangproj(ecc, aorbit, dist):
	#Returns the maximum angular projection (aorbit in AU; dist in pc)
	
	return 1000.*aorbit*(1.+ecc)/dist #maxangproj [mas] (Traub & Oppenheimer (2010))


def compute_maxangproj_trueanom(ecc, aorbit, dist, longperiast, incl):
	#Returns the maximum angular separation (aorbit in AU; dist in pc)

	longperiast = longperiast*np.pi/180.		#Transform to radians
	incl = incl*np.pi/180.				#Transform to radians
	trueanom = np.arange(-180.,181.,1.)*np.pi/180.	#Transform to radians

	angproj = ((aorbit*(1.-ecc**2.))/(dist*(1.+ecc*np.cos(trueanom))))*np.sqrt((np.cos(trueanom+longperiast))**2.+(np.sin(trueanom+longperiast))**2.*(np.cos(incl))**2.)
	angproj = angproj*1000.				#Transform to m.a.s.

	return np.amax(angproj)



def filter_planets_detections(method, dictionary):
	#We want to reduce the dictionary to only those planets found with a certain method
	
	count = 0
	new_dict = {}
	for key in dictionary:	
		if dictionary[key]['discoverymethod'] == method:
			new_dict[key] = dictionary[key]
			count = count + 1

	return new_dict


def sorted_planets(dictio):
	#I sort the names of planets in the dictionary. Then I use this ordered set of planet to print the results table of the manuscript
	dicprov = {}
	for key in dictio:
		#dicprov[key] = float(dictio[key]['sy_dist'])		#Sorting by how close are they to the Sun
		#dicprov[key] = dictio[key]['pl_maxangsep']		#Sorting by maximum angular separation
		dicprov[key] = float(dictio[key]['pl_detectrate'])	#Sorting by detectability rate

	dict_sorted_plnames = sorted(dicprov.items(), key=operator.itemgetter(1))

	#sorted_plnames = [x[0] for x in dict_sorted_plnames]		#Sorting lower values first (e.g. for sy_dist)
	sorted_plnames = [x[0] for x in dict_sorted_plnames][::-1]	#Sorting higher values first (e.g. for pl_detectrate)
	return sorted_plnames
	

def Mp_2_Rp(massmj):
	#We compute the radius of the planet from its mass, by assuming different densities

	coeff1_uncert_giants = random.uniform(-0.03, +0.03)
	coeff2_uncert_giants = random.uniform(-0.03, +0.03)

	coeff1_uncert_volatiles = random.uniform(-0.11, +0.11)
	coeff2_uncert_volatiles = random.uniform(-0.04, +0.04)

	coeff1_uncert_rocky = random.uniform(-0.02, +0.02)
	coeff2_uncert_rocky = random.uniform(-0.01, +0.01)


	#This is the new implementation from Hatzes & Rauer (2015) + Otegi et al. (2020)
	if massmj > 0.3:
		logdens_gp = (1.15+coeff1_uncert_giants)*np.log10(massmj)-(0.11+coeff2_uncert_giants)	#density is given in g/cm^3 and Rjup in m
		return np.cbrt((massmj*Mjup*1000.)/((4./3.)*np.pi*np.power(10.,logdens_gp)))/100./Rjup	#I have to adapt the value of Mjup (in kg) and the resulting Rp (in cm)
	elif massmj <= 0.3 and massmj > 9.798E-3:
		massmE = massmj*Mjup/Mear
		return (Rear/Rjup)*((0.70+coeff1_uncert_volatiles)*np.power(massmE, (0.63+coeff2_uncert_volatiles)))
	else:
		massmE = massmj*Mjup/Mear
		return (Rear/Rjup)*((1.03+coeff1_uncert_rocky)*np.power(massmE, (0.29+coeff2_uncert_rocky)))



def give_planet_radius(planet, incl):
	#We filter out those exoplanets with no value of Rp or without a value of Mp to compute Rp
	incl *= np.pi/180.	#Radians

	if planet['pl_radj'] != '':						#Already has a value
		planet['pl_radj_run_n'] = planet['pl_radj']
		planet['pl_radj_flag_computedOCG'] = ''
	elif planet['pl_radj'] == '' and planet['pl_bmassj'] != '':		#No Rp value but Mp to compute it from
		if planet['pl_bmassprov']=='Msini':
			Mp = float(planet['pl_bmassj'])/np.sin(incl)
		else:
			Mp = float(planet['pl_bmassj'])
		planet['pl_radj_run_n'] = Mp_2_Rp(Mp)
		planet['pl_radj_flag_computedOCG'] = '*'
	else:									#No info. Cannot compute Rp
		planet['pl_radj_run_n'] = ''
		planet['pl_radj_flag_computedOCG'] = ''
		
	return planet




def orbit_evolution(aorbit, incl, ecc, longperiast, Rp, Ag, dist):
	
	BigOmega = 0.
	trueanom = np.arange(-180., 181., 1. )*np.pi/180.#Here you can change how maany points per orbit np.arange(-180., 181., 0.1)
	#trueanom = np.arange(-180., 180.1, 0.1)*np.pi/180.	
	#trueanom = np.arange(-180., 180.5, 0.5)*np.pi/180.	
	incl *= np.pi/180.
	longperiast *= np.pi/180.

	dist_pl_st = (aorbit*(1.-ecc**2)) / (1. + ecc* np.cos(trueanom))

	alpha = np.arccos(np.sin(incl) * np.sin(trueanom+longperiast))

	Xcoord = dist_pl_st*(np.cos(BigOmega)*np.cos(trueanom+longperiast)-np.sin(BigOmega)*np.sin(trueanom+longperiast)*np.cos(incl))
	Ycoord = dist_pl_st*(np.sin(BigOmega)*np.cos(trueanom+longperiast)+np.cos(BigOmega)*np.sin(trueanom+longperiast)*np.cos(incl))
	Zcoord = dist_pl_st*np.sin(trueanom+longperiast)*np.sin(incl)

	#angproj = (dist_pl_st/dist)*np.sqrt((np.cos(trueanom+longperiast))**2.+(np.sin(trueanom+longperiast))**2.*(np.cos(incl))**2.)
	angproj = (1./dist)*np.sqrt(Xcoord**2.+Ycoord**2.)
	angproj = angproj*1000.			#Transform to m.a.s.

	dist_pl_st = dist_pl_st*AU_constant	#meters
	Rp = Rp*Rjup				#meters

	t_tp = (1./(2.*np.pi))*(-((ecc*np.sin(trueanom)*np.sqrt(1.-ecc**2.))/(1.+ecc*np.cos(trueanom)))+2.*np.arctan(np.sqrt((1.-ecc)/(1.+ecc))*np.tan(trueanom/2.)))

	E_anom = 2.*np.arctan(np.sqrt((1.-ecc)/(1.+ecc))*np.tan(trueanom/2.))
	M_anom = 2*np.pi*t_tp

	lambert_scatt = (np.sin(alpha)+(np.pi-alpha)*np.cos(alpha)) / np.pi

	Fp_Fstar = (Rp/dist_pl_st)**2. * Ag * lambert_scatt

	return Xcoord, Ycoord, t_tp, trueanom, angproj, Fp_Fstar, alpha, dist_pl_st/AU_constant

##################Sebastian############


def thermal_flux_planet(Teq_orbit, R_p, st_radi, st_teff, wav_thermal, Transmission_curve):
	global _MASS_MAG_ERROR_SHOWN

	R_p_m = R_p * Rjup
	st_radi_m = st_radi * Rsun

	# Initialize arrays for wavelength-dependent calculations
	B_planet = np.zeros((len(wav_thermal)))
	B_star = np.zeros((len(wav_thermal)))
	Fp_Fstar_thermal = np.zeros(len(Teq_orbit))
	F_planet_thermal_convolution_before_integration = np.zeros((len(Teq_orbit), len(wav_thermal)))
	F_planet_thermal_convolution = np.zeros(len(Teq_orbit))
	F_star_thermal_convolution_before_integration = np.zeros((len(Teq_orbit), len(wav_thermal)))
	F_star_thermal_convolution = np.zeros(len(Teq_orbit))
	Fp_Fstar_thermal_unconcolved = np.zeros((len(Teq_orbit), len(wav_thermal)))



	B_star = (2*h*c**2) / (wav_thermal**5) / (np.exp(h*c/(wav_thermal*k*st_teff)) - 1)

	# Calculate Planck function for planet at each orbital position and wavelength
	for i in range(len(Teq_orbit)):
		B_planet = (2*h*c**2) / (wav_thermal**5) / (np.exp(h*c/(wav_thermal*k*Teq_orbit[i])) - 1)
		F_planet = B_planet * np.pi * (R_p_m**2)  # Total power emitted by the planet per wavelength
		F_star = B_star * np.pi * (st_radi_m**2)  # Total power emitted by the star per wavelength
		
		Fp_Fstar_thermal_unconcolved[i] = (B_planet * (R_p_m**2)) / (B_star * (st_radi_m**2))

		normalization_thermal = np.trapezoid(Transmission_curve, wav_thermal)  # Normalization factor

		Fp_Fstar_thermal[i] = np.trapezoid(Fp_Fstar_thermal_unconcolved[i] * Transmission_curve, wav_thermal) / normalization_thermal # Integrate over wavelength unnormalized



	return Fp_Fstar_thermal


def thermal_flux_planet_mass2mag(Teq_orbit, sy_dist, band, st_mag, mass_mjup, age_gyr):

	Fp_Fstar_thermal = np.zeros(len(Teq_orbit))

	# --- ADDED: Calculate Absolute Magnitude from Mass ---
	# 1. Get Mass (Jupiter Masses)
	if mass_mjup is not None:

	
		
		# 3. Define Model and Filter (Adjust these names as needed)
		model_name = "Sonorabobcat+0.0"
		
		# Use the mass2mag filter name from the global configuration
		# Note: mass2mag expects filter names like "L'", "M'", "W3", etc.
		filter_name = mass2mag_filter_name
		
		# approximation using Sonora Bobcat models in L' and M' band of BRIKS telescope
		#most optimal for ELT METIS L,M,N bands calculated magnitudes by Bex for lower masses (up to ~3 Mjup) and higher Atmo models 
		
		# 4. Call the function hasnet worked with try earlier then just take that loop out
		
		abs_mag_planet, temp_planet = convert_mass2mag([mass_mjup], age_gyr, filter_name=filter_name, model=model_name, return_temp  = True) 
		#print(f"Run: Mass={mass_mjup:.4f} Mjup, Abs_Mag={abs_mag_planet}, Temp={temp_planet} K")

		# 5. transform into Apparent Magnitude
		# Apparent Magnitude = Absolute Magnitude + 5 * log10(Distance [pc]) - 5
		# sy_dist is expected to be in parsecs.
		app_mag_planet = abs_mag_planet + 5 * np.log10(sy_dist) - 5
		#print(f"Run : Mass={mass_mjup:.4f} Mjup, rel_Mag={app_mag_planet}")


		# 6. calculate the contrast Fp/Fstar from the magnitude difference
		# Fp/Fstar = 10^(-0.4 * (m_planet - m_star))
		
		# Calculate Contrast
		#get source for this function 
		contrast_val = 10**(-0.4 * (app_mag_planet - st_mag))
		
		# Fill the array (assuming constant contrast for the orbit if mass-derived)
		# If Teq_orbit varies, mass2mag doesn't account for it (it uses mass/age). 
		Fp_Fstar_thermal[:] = contrast_val

		#print(f"Star Mag ({band})={st_mag:.2f}, Contrast={contrast_val}")

	return Fp_Fstar_thermal, temp_planet, abs_mag_planet


def thermal_flux_planet_MADYS(Teq_orbit, sy_dist, band, st_mag, mass_mjup, age_gyr, Madys_Modell_selection=None):

#bex-atmo2023-ceq (Chemical Equilibrium)
#Assumption: The atmosphere is stable. Chemical time scales are much shorter than mixing time scales.
#Physics: The chemical composition at any layer is determined solely by the local temperature and pressure (Chemical Equilibrium).Result: Standard baseline model.

#bex-atmo2023-neq-w (Non-Equilibrium - Weak Mixing)
#Assumption: There is Weak vertical mixing (turbulence). #Physics: The vertical diffusion coefficient (Kzz) is low/moderate. This "dredges up" deep gases (CO) to cooler upper layers where Methane (CH4)  normally dominate s. #Result: Changes the spectral features slightly (e.g., strong CO absorption bands where you wouldn't expect them in equilibrium).

#bex-atmo2023-neq-s (Non-Equilibrium - Strong Mixing)
#Assumption: There is Strong vertical mixing. #Physics: The vertical diffusion coefficient (Kzz) is high (strong turbulence). strongly alters the chemical abundance profiles compared to equilibrium.
#Result: Significant changes in spectra and magnitudes, especially in infrared bands affected by CO, CH4, and NH3.often more representative of young, giant planets which are convective and turbulent.

	#Minimum mass in this model bex-atmo2023-ceq & neq-s, neq-s is 0.15 Mjup at 100M years
	


	Fp_Fstar_thermal = np.zeros(len(Teq_orbit))
	#temp_planet = 1

	# --- ADDED: Calculate Absolute Magnitude from Mass ---
	# 1. Get Mass (Jupiter Masses)
	#mass_array has to be in units of M_sun
	if mass_mjup is not None:

		#print(ModelHandler.available('full_model_list'))
		#print(info_filters())
		#print(mass_mjup, age_gyr, 'here is the mass')

		# Filter METIS MADYS.   METIS_Lp, METIS_Mp, METIS_N1, METIS_N2

		mass_sun = 0.000954588 * mass_mjup

		#print(mass_mjup)

		mass_range = np.array([mass_sun, mass_sun])

		age_range_Myr = np.array([age_gyr*1000, age_gyr*1000])





		# Use the MADYS filter name from the global configuration
		your_filters_list = [madys_filter_name, "logT"]
		#madys.ModelHandler.download_model('atmo2023')

		if Madys_Modell_selection is None:
			Madys_Modell_selection = 'bex-atmo2023-ceq'
			print('no model choosen default model bex-atmo is used')

		# 3. Define Model and Filter (Adjust these names as needed)
		iso = IsochroneGrid(Madys_Modell_selection, your_filters_list, age_range = age_range_Myr, mass_range = mass_range, n_steps=[1,1])

		#TESTTTTTT BOOOTTTHHHI-band filter
		#Reference: Bessell, PASP 102, 1181 (1990)
		#mass_range = np.array([0.001, 0.001])
		#Available in the following models: bex(bex-helios-clear,bex-petitcode-clear'/ 'bex-petitcode-cloudy), 
		#bhac15, bt-settl, dartmouth, geneva, mist, parsec, parsec2, pm13, spots, starevol
		#iso = IsochroneGrid('atmo2020-ceq', ['NIRCAM_p_F090W',"logT"], age_range = age_range_Myr, mass_range = mass_range, n_steps=[1,1])
		#print(iso.data)
		#print(iso)
		#ModelHandler.available('atmo')
		#madys.info_filters()
		#quit()

		abs_mag_planet = iso.data[0, 0, 0]  #(Masses, Ages, Filters) we take the first mass because values for both ages and masses are the same
		#print(abs_mag_planet)

		log10_temp = iso.data[0, 0, 1]
		#print(log10_temp)
		temp_planet = 10**log10_temp
		#print(temp_planet)


		#print(iso.masses)
		#print(iso.ages)
		#ModelHandler.available()
		#madys.info_filters('METIS_Lp')


		# 5. transform into Apparent Magnitude
		# Apparent Magnitude = Absolute Magnitude + 5 * log10(Distance [pc]) - 5

		app_mag_planet = abs_mag_planet + 5 * np.log10(sy_dist) - 5
		#print(f"Run : Mass={mass_mjup:.4f} Mjup, rel_Mag={app_mag_planet}")


		# 6. calculate the contrast Fp/Fstar from the magnitude difference
		# Fp/Fstar = 10^(-0.4 * (m_planet - m_star))
		
		contrast_val = 10**(-0.4 * (app_mag_planet - st_mag))
		
		# Fill the array (assuming constant contrast for the orbit if mass-derived)
		# If Teq_orbit varies doesnt change this value (it uses mass/age). 
		Fp_Fstar_thermal[:] = contrast_val

		#print(f"Star Mag ({band})={st_mag:.2f}, Planetmass={mass_mjup:.4f}, Contrast={contrast_val}")

	return Fp_Fstar_thermal, temp_planet, abs_mag_planet




	




def find_interval_lims(arr):
	#This is a routine to find the intervals contained in an array which has NaNs in it (to plot the detectability windows)

	flagleftlim = 0
	flagrightlim = 0
	flaginit = 0		#This will mark in case the first truanom is already observable (because, as it is cyclic, it should join the interval of alphas given by the last truanom)
	maxvalue = 0
	minvalue= 0
	leftlim, rightlim = [], []
	minvalues, maxvalues = [], []
	valuesprov, valuesprovinit = [], []		#Used just to find the minimum and maximum values within an interval
	for i in range(len(arr)):
		if i==len(arr)-1:	#If we are on the last value of the considered array
			if flagleftlim==1:
				rightlim.append(arr[i])
				if flaginit == 1:
					minvalues.append(np.nanmin(valuesprovinit+valuesprov))
					maxvalues.append(np.nanmax(valuesprovinit+valuesprov))
				else:
					minvalues.append(np.nanmin(valuesprov))
					maxvalues.append(np.nanmax(valuesprov))
											
		else:
			if arr[i]==arr[i] and arr[i+1]==arr[i+1] and flagleftlim==0:	#Found the beginning of an interval
				leftlim.append(arr[i])
				flagleftlim = 1
				valuesprov.append(arr[i])
				if i==0:
					flaginit = 1
			elif arr[i]==arr[i] and arr[i+1]!=arr[i+1] and flagleftlim==0:	#If only one point is observable for this interval (probably meaning I need a denser grid)
				leftlim.append(arr[i])
				rightlim.append(arr[i])
				minvalues.append(arr[i])
				maxvalues.append(arr[i])
			elif arr[i]==arr[i] and arr[i+1]!=arr[i+1] and flagleftlim==1:	#Found the end of an interval
				rightlim.append(arr[i])
				if flaginit == 1:		#In case first and last part of the orbit belongs to the same interval
					valuesprovinit = valuesprov[:]
				else:
					minvalues.append(np.nanmin(valuesprov))
					maxvalues.append(np.nanmax(valuesprov))
				valuesprov = []	#Re-initialising for next interval
				flagleftlim = 0	#Re-initialising for next interval
			elif arr[i]==arr[i] and arr[i+1]==arr[i+1] and flagleftlim == 1:#We are still in an interval
				valuesprov.append(arr[i])

	return leftlim, rightlim, minvalues, maxvalues



def bootstrapping(keys, planet, planet_aux):
	#At each of the runs, I select a value of the parameters to within the errorbars

	for x in keys:
		if x in planet and planet[x] != '': # Check if key exists in dictionary
			if planet[x+'err2']!= '' and planet[x+'err1']!= '':		#Checking that errorbars are available
				if float(planet[x+'err2'])<=0 and float(planet[x+'err1'])>=0:		#Checking that the +/- signs are correct
					planet_aux[x] = random.uniform(float(planet[x])+float(planet[x+'err2']), float(planet[x])+float(planet[x+'err1']))
					#print(planet_aux[x])
				else:
					print("\nSome error with the +/- uncertainties for %s [%s]\n"%(planet['pl_name'], x))
			else:
				# If it's a magnitude parameter, print a warning
				if 'mag' in x and nrun==0:
					print(f"Warning: No uncertainties for {x} in {planet['pl_name']}. Using nominal value: {planet[x]}")
				planet_aux[x] = planet[x]
		else:
			# Key might not exist in this planet's dictionary (e.g. missing mag column)
			pass
			
	return planet_aux

# Gaussian sampling instead of Bootstrapping 
def gaussian_sampling(keys, planet, planet_aux):
	#At each of the runs, I select a value of the parameters using a gaussian distribution

	for x in keys:
		if x in planet and planet[x] != '': # Check if key exists in dictionary
			if planet[x+'err2']!= '' and planet[x+'err1']!= '':		#Checking that errorbars are available
				if float(planet[x+'err2'])<=0 and float(planet[x+'err1'])>=0:		#Checking that the +/- signs are correct
					sigma = (float(planet[x+'err1']) - float(planet[x+'err2'])) / 2.
					#skewed gaussian distribution
					planet_aux[x] = random.gauss(float(planet[x]), sigma)
					
					#could say: if dict_limits[sy_dist]: (0, False)
					#if dict_limits[x]
					#if planet_aux < dict_limits[x][0] or planet_aux > dict_limits[x][1]
						#redraw

				else:
					print("\nSome error with the +/- uncertainties for %s [%s]\n"%(planet['pl_name'], x))
			else:
				# If it's a magnitude parameter, print a warning
				if 'mag' in x:
					print(f"Warning: No uncertainties for {x} in {planet['pl_name']}. Using nominal value: {planet[x]}")
				planet_aux[x] = planet[x]
		else:
			# Key might not exist in this planet's dictionary (e.g. missing mag column)
			pass
			
	return planet_aux


#Normal distribution  to both sides of the distribution beeing different
def split_normal_sampling(keys, planet, planet_aux):
    #At each of the runs, I select a value of the parameters using a split normal distribution

    for x in keys:
        if x in planet and planet[x] != '': # Check if key exists in dictionary
            if planet[x+'err2']!= '' and planet[x+'err1']!= '':     #Checking that errorbars are available
                if float(planet[x+'err2'])<=0 and float(planet[x+'err1'])>=0:       #Checking that the +/- signs are correct
                    
                    mu = float(planet[x])
                    sigma_plus = float(planet[x+'err1'])

                    sigma_minus = abs(float(planet[x+'err2'])) # Take absolute magnitude of the negative error
                    #print(sigma_plus, sigma_minus)
                    
                    # Draw standard normal
                    z = random.gauss(0, 1)
                    #print(z)
                    
                    # Split Normal Logic: Scale by sigma_plus if z>0, sigma_minus if z<0
                    if z >= 0:
                        planet_aux[x] = mu + (z * sigma_plus)
                    else:
                        planet_aux[x] = mu + (z * sigma_minus)
                        
                else:
                    print("\nSome error with the +/- uncertainties for %s [%s]\n"%(planet['pl_name'], x))
            else:
                # If it's a magnitude parameter, print a warning
                if 'mag' in x and nrun==0:
                    print(f"Warning: No uncertainties for {x} in {planet['pl_name']}. Using nominal value: {planet[x]}")
                planet_aux[x] = planet[x]
        else:
            # Key might not exist in this planet's dictionary (e.g. missing mag column)
            pass
            
    return planet_aux


def does_it_transit(planet):
	#Checking if the orbital configuration is observable in transit
	incl = float(planet['pl_orbincl'])*np.pi/180.
	orblper = float(planet['pl_orblper'])*np.pi/180.

	b_impact = (float(planet['pl_orbsmax'])*AU_constant/(float(planet['st_rad'])*Rsun))*((1.-float(planet['pl_orbeccen'])**2.)/(1.-float(planet['pl_orbeccen'])*np.sin(orblper)))*np.cos(incl)

	condition = (float(planet['st_rad'])*Rsun-float(planet['pl_radj_run_n'])*Rjup)/(float(planet['st_rad'])*Rsun)

	if np.absolute(b_impact)<condition:
		return 1
	else:
		return 0

	
def minimum_contrast_PCS(sep):
	 #Detectability of PCS @ ELT
	mincontrast = np.copy(sep)*0.#np.zeros(len(sep))
	# Set values for x < 15
	mincontrast[sep < 15] = 1E100
	# Linear interpolation between 1e-8 and 1e-9 for 15 <= x <= 100
	mask_linear_interpolation = (15 <= sep) & (sep <= 100)
	loginterp = np.interp(np.log10(sep[mask_linear_interpolation]), [np.log10(15), np.log10(100)], [np.log10(1e-8), np.log10(1e-9)])
	mincontrast[mask_linear_interpolation] = 10**loginterp
	# Set values for x > 100
	mincontrast[sep > 100] = 1e-9
	return mincontrast
	
def minimum_contrast_Roman_pessim(sep):
	#Detectability of Roman (pessimistic)
	D_WFIRST = 2.4	#m
	IWA = (4*wav*1.E-9/D_WFIRST)*(180.*3600.*1000./np.pi)	#Trauger et al. (2016)
	OWA = (8*wav*1.E-9/D_WFIRST)*(180.*3600.*1000./np.pi)	
	Cmin = 5.E-9
	mincontrast = np.zeros(len(sep))
	mincontrast[sep < IWA] = 1E100
	mincontrast[(IWA<=sep) & (sep<=OWA)] = Cmin
	mincontrast[sep > OWA] = 1E100
	return mincontrast
	
def minimum_contrast_Roman_optim(sep):
	#Detectability of Roman (optimistic)
	global IWA
	global OWA
	D_WFIRST = 2.4	#m
	IWA = (3*wav*1.E-9/D_WFIRST)*(180.*3600.*1000./np.pi)	#Trauger et al. (2016)
	OWA = (9*wav*1.E-9/D_WFIRST)*(180.*3600.*1000./np.pi)	
	Cmin = 1.E-9
	#mincontrast = np.zeros(len(sep))
	mincontrast = np.copy(sep)*0.
	mincontrast[sep < IWA] = 1E100
	mincontrast[(IWA<=sep) & (sep<=OWA)] = Cmin
	mincontrast[sep > OWA] = 1E100
	return mincontrast
	
def minimum_contrast_Roman_WFoV(sep):
	#Detectability of Roman (optimistic)
	global IWA
	global OWA
	D_WFIRST = 2.4	#m
	IWA = (6*wav*1.E-9/D_WFIRST)*(180.*3600.*1000./np.pi)	#Trauger et al. (2016)
	OWA = (20*wav*1.E-9/D_WFIRST)*(180.*3600.*1000./np.pi)	
	Cmin = 1.E-9
	#mincontrast = np.zeros(len(sep))
	mincontrast = np.copy(sep)*0.
	mincontrast[sep < IWA] = 1E100
	mincontrast[(IWA<=sep) & (sep<=OWA)] = Cmin
	mincontrast[sep > OWA] = 1E100
	return mincontrast
	







# Global cache for METIS contrast curves
METIS_DATA_CACHE = {}
_METIS_WARNINGS_SHOWN = set()


def load_contrast_curves():
	"""
	Loads all METIS contrast curves into a global dictionary.
	Structure: METIS_DATA_CACHE[band][magnitude] = (separation_mas, contrast)
	"""






	global METIS_DATA_CACHE

	# Determine which band to load based on FILTER_USED
	# Get OWA extension from config
	
	if FILTER_USED in FILTER_CONFIGS:
		config = FILTER_CONFIGS[FILTER_USED]
		target_owa = config.get('owa_extension')
		integration_time = config.get('integration_time')
		contrast_assumption = config.get('contrast_assumption')
		
	files = []

	# Define file mappings for each filter: (magnitude, filename)
	if FILTER_USED == 'METIS_L_BAND':
		# Only load if not already cached
		if FILTER_USED in METIS_DATA_CACHE:
			return
		files = [
			(0, 'cc_adi_bckg0_L_CVC_all_effects.fits'),
			(7, 'cc_adi_bckg1_mag7_L_CVC_all_effects.fits'),
			(8, 'cc_adi_bckg1_mag8_L_CVC_all_effects.fits'),
			(9, 'cc_adi_bckg1_mag9_L_CVC_all_effects.fits'),
			(10, 'cc_adi_bckg1_mag10_L_CVC_all_effects.fits'),
			(11, 'cc_adi_bckg1_mag11_L_CVC_all_effects.fits'),
			(12, 'cc_adi_bckg1_mag12_L_CVC_all_effects.fits')
		]

	elif FILTER_USED == 'METIS_M_BAND':
		# Only load if not already cached
		if FILTER_USED in METIS_DATA_CACHE:
			return
		files = [
			(0, 'cc_adi_bckg0_M_CVC_all_effects.fits'),
			(7, 'cc_adi_bckg1_mag7_M_CVC_all_effects.fits'),
			(8, 'cc_adi_bckg1_mag8_M_CVC_all_effects.fits'),
			(9, 'cc_adi_bckg1_mag9_M_CVC_all_effects.fits'),
			(10, 'cc_adi_bckg1_mag10_M_CVC_all_effects.fits')
		]

	elif FILTER_USED == 'METIS_N_BAND':
		# Only load if not already cached
		if FILTER_USED in METIS_DATA_CACHE:
			return
		# These are pretty old N band contrast curves: 5 to 8 not the sweet spot in terms of magnitudes for N-band HCI 
		# (consider coronagraphy to be useful/realistic for magnitudes brighter than 3 in N). 
		# Curves include water vapour seeing (close seperations), but not our latest estimates which are less pessimistic.
		# Much older contrast curves (from five years ago) for a few bright targets ranging from Nmag = -1.5 to 1.5 (not regularly spaced). 
		# They do not consider water vapour seeing at all, so you should definitely not rely on them. 
		# I would tend to think that only the background-limited sensitivity can be more or less trusted here.
		# (low seperation not trustable, high ones can be trusted more)
		files = [
			(-1.478, 'cc_adi_bckg1_mag-1.478_N2_CVC_fullM1_all_effects.fits'),
			(-1.312, 'cc_adi_bckg1_mag-1.312_N2_CVC_fullM1_all_effects.fits'),
			(-0.633, 'cc_adi_bckg1_mag-0.633_N2_CVC_fullM1_all_effects.fits'),
			(-0.592, 'cc_adi_bckg1_mag-0.592_N2_CVC_fullM1_all_effects.fits'),
			(0.267, 'cc_adi_bckg1_mag0.267_N2_CVC_fullM1_all_effects.fits'),
			(1.691, 'cc_adi_bckg1_mag1.691_N2_CVC_fullM1_all_effects.fits'),
			(5, 'cc_adi_bckg1_mag5_N2_IMG_exeter_all_effects.fits'),
			(5.5, 'cc_adi_bckg1_mag5.5_N2_IMG_exeter_all_effects.fits'),
			(6, 'cc_adi_bckg1_mag6_N2_IMG_exeter_all_effects.fits'),
			(6.5, 'cc_adi_bckg1_mag6.5_N2_IMG_exeter_all_effects.fits'),
			(7, 'cc_adi_bckg1_mag7_N2_IMG_exeter_all_effects.fits'),
			(7.5, 'cc_adi_bckg1_mag7.5_N2_IMG_exeter_all_effects.fits'),
			(8, 'cc_adi_bckg1_mag8_N2_IMG_exeter_all_effects.fits'),
		]

	elif FILTER_USED == 'ROMAN_F1':
		# Only load if not already cached
		if FILTER_USED in METIS_DATA_CACHE:
			return
		
		# Default files list
		files = []

		if contrast_assumption == 'opti':
			if integration_time == 'short':
				files = [(5, 'Roman_pred_imaging_short_opti.txt')]
			elif integration_time == 'medium':
				files = [(5, 'Roman_pred_imaging_medium_opti.txt')]
			elif integration_time == 'long':
				files = [(5, 'Roman_pred_imaging_long_opti.txt')]
				
		elif contrast_assumption == 'cons':
			if integration_time == 'short':
				files = [(5, 'Roman_pred_imaging_short_cons.txt')]
			elif integration_time == 'medium':
				files = [(5, 'Roman_pred_imaging_medium_cons.txt')]
			elif integration_time == 'long':
				files = [(5, 'Roman_pred_imaging_long_cons.txt')]
		else:
			print(f"Warning: Unknown contrast assumption ROMAN: {contrast_assumption}")

	elif FILTER_USED == 'ROMAN_F2':
		# Only load if not already cached
		if FILTER_USED in METIS_DATA_CACHE:
			return
		
		# Default files list
		files = []
		
		if contrast_assumption == 'opti':
			if integration_time == 'short':
				# Assuming no file for short as requested in original hardcoded logic
				files = [(5, 'Roman_pred_spec_short_opti.txt')]
			elif integration_time == 'medium':
				files = [(5, 'Roman_pred_spec_medium_opti.txt')]
			elif integration_time == 'long':
				files = [(5, 'Roman_pred_spec_long_opti.txt')]

		elif contrast_assumption == 'cons':
			if integration_time == 'short':
				files = [(5, 'Roman_pred_spec_short_cons.txt')]
			elif integration_time == 'medium':
				files = [(5, 'Roman_pred_spec_medium_cons.txt')]
			elif integration_time == 'long':
				files = [(5, 'Roman_pred_spec_long_cons.txt')]
		else:
			print(f"Warning: Unknown contrast assumption ROMAN: {contrast_assumption}")

	elif FILTER_USED == 'ROMAN_F3':
		# Only load if not already cached
		if FILTER_USED in METIS_DATA_CACHE:
			return
		
		# Default files list
		files = []

		if contrast_assumption == 'opti':
			if integration_time == 'short':
				files = [(5, 'Roman_pred_spec_short_opti.txt')]
			elif integration_time == 'medium':
				files = [(5, 'Roman_pred_spec_medium_opti.txt')]
			elif integration_time == 'long':
				files = [(5, 'Roman_pred_spec_long_opti.txt')]
				
		elif contrast_assumption == 'cons':
			if integration_time == 'short':
				files = [(5, 'Roman_pred_spec_short_cons.txt')]
			elif integration_time == 'medium':
				files = [(5, 'Roman_pred_spec_medium_cons.txt')]
			elif integration_time == 'long':
				files = [(5, 'Roman_pred_spec_long_cons.txt')]
		else:
			print(f"Warning: Unknown contrast assumption ROMAN: {contrast_assumption}")

	elif FILTER_USED == 'ROMAN_F4':
		# Only load if not already cached
		if FILTER_USED in METIS_DATA_CACHE:
			return
		
		# Default files list
		files = []
		
		if contrast_assumption == 'opti':
			if integration_time == 'short':
				files = [(5, 'Roman_pred_wideFOVimaging_short_opti.txt')]
			elif integration_time == 'medium':
				files = [(5, 'Roman_pred_wideFOVimaging_medium_opti.txt')]
			elif integration_time == 'long':
				files = [(5, 'Roman_pred_wideFOVimaging_long_opti.txt')]

		elif contrast_assumption == 'cons':
			if integration_time == 'short':
				files = [(5, 'Roman_pred_wideFOVimaging_short_cons.txt')]
			elif integration_time == 'medium':
				files = [(5, 'Roman_pred_wideFOVimaging_medium_cons.txt')]
			elif integration_time == 'long':
				files = [(5, 'Roman_pred_wideFOVimaging_long_cons.txt')]
		else:
			print(f"Warning: Unknown contrast assumption ROMAN: {contrast_assumption}")

	

	else:
		print(f"Warning: Unknown filter {FILTER_USED}")
		return

	if not files:
		print(f"Warning: No contrast curve files defined for filter {FILTER_USED}")
		return

	# Initialize cache for the selected filter
	METIS_DATA_CACHE[FILTER_USED] = {}

	base_path = os.path.normpath(os.path.expandvars('$HOME/desktop/Master_Thesis_Sebastian/Contrast Curves/aaryn/'))

	# Load files for the selected band
	for mag, filename in files:
		full_path = os.path.join(base_path, filename)
		if os.path.exists(full_path):
			try:
				if FILTER_USED == 'METIS_L_BAND' or FILTER_USED == 'METIS_M_BAND' or FILTER_USED == 'METIS_N_BAND':
					data = fits.getdata(full_path)
					# data[0] is separation in arcsec, data[1] is contrast
					# Convert separation to mas immediately
					sep_mas = data[0] * 1000.0
					contrast = data[1]
				
				if FILTER_USED == 'ROMAN_F1' or FILTER_USED == 'ROMAN_F2' or FILTER_USED == 'ROMAN_F3' or FILTER_USED == 'ROMAN_F4':
					data = np.loadtxt(full_path, skiprows=2)
					# data[0] is separation in arcsec, data[1] is contrast
					# Convert separation to mas immediately
					D_WFIRST = 2.4	#m
					sep_mas = (data[:,0] * 1.E-9 * data[:,2]/ D_WFIRST)* (180.*3600.*1000./np.pi)
					contrast = data[:,1]	





				# Extend contrast curve to target OWA if needed
				# Assumption: constant sensitivity beyond measured data
				if target_owa is not None and sep_mas[-1] < target_owa:
					# Calculate median spacing from original data
					median_spacing = np.median(np.diff(sep_mas))
					
					# Create extended separation array from max to target OWA
					extended_sep = np.arange(sep_mas[-1] + median_spacing, 
											target_owa + median_spacing, 
											median_spacing)
					
					# Combine original and extended separations
					sep_mas = np.concatenate([sep_mas, extended_sep])
					
					# Pad contrast with last measured value (constant sensitivity assumption)
					contrast = np.concatenate([contrast, 
												np.full(len(extended_sep), contrast[-1])])
				
				METIS_DATA_CACHE[FILTER_USED][mag] = (sep_mas, contrast)
			except Exception as e:
				print(f"Warning: Could not load {filename}: {e}")
		else:
			print(f"Warning: File not found: {full_path}") 


def minimum_contrast_METIS(sep, band, mag, kmag=None):
    """
    Calculates the minimum contrast for a given separation, band, and stellar magnitude.
    Interpolates between available contrast curves based on magnitude.

    Parameters:
    -----------
    kmag : float, optional
        K-band stellar magnitude. When a METIS filter is active, used for
        contrast curve selection and interpolation, since the curve files are
        labelled by K-band magnitude. Falls back to mag if not provided.
    """
    load_contrast_curves()
    global _METIS_WARNINGS_SHOWN
    global IWA
    global OWA

    if band not in METIS_DATA_CACHE or not METIS_DATA_CACHE[band]:
        # Fallback or error if no data for this band
        print(f"Warning: No data available for band {band}")
        return np.ones_like(sep) * 1e-9 
        
    # Get all available magnitudes (include all, no filtering)
    available_mags = sorted(METIS_DATA_CACHE[band].keys())
    
    if not available_mags:
        # No data available
        print('no contrast curves where found')
        return np.ones_like(sep) * 1e-9

    min_mag = available_mags[0]
    max_mag = available_mags[-1]

    # For METIS filters, the contrast curve files are labelled by K-band magnitude.
    # Use sy_kmag (passed as kmag) for curve selection/interpolation; fall back to mag if not available.
    interp_mag = kmag if (FILTER_USED.startswith('METIS') and kmag is not None) else mag

    # Clamp magnitude to available range with warnings
    if interp_mag < min_mag:
        warning_key = (band, 'min', interp_mag)
        if warning_key not in _METIS_WARNINGS_SHOWN:
            print(f"Warning: Star magnitude {interp_mag:.2f} is brighter than available curves ({min_mag}). Clamping to {min_mag}.")
            _METIS_WARNINGS_SHOWN.add(warning_key)
        interp_mag = min_mag

    elif interp_mag > max_mag:
        warning_key = (band, 'max', interp_mag)
        if warning_key not in _METIS_WARNINGS_SHOWN:
            print(f"Warning: Star magnitude {interp_mag:.2f} is fainter than available curves ({max_mag}). Clamping to {max_mag}.")
            _METIS_WARNINGS_SHOWN.add(warning_key)
        interp_mag = max_mag
        
    # Find bracketing magnitudes
    mag_low = min_mag
    mag_high = max_mag
    
    for m in available_mags:
        if m <= interp_mag:
            mag_low = m
        if m >= interp_mag:
            mag_high = m
            break
            
    # Get curves
    sep_low, contrast_low = METIS_DATA_CACHE[band][mag_low]
    sep_high, contrast_high = METIS_DATA_CACHE[band][mag_high]
    
    # Interpolate contrast at requested separation for both curves
    # Note: sep is in mas, stored data is in mas
    C_low = np.interp(sep, sep_low, contrast_low)
    C_high = np.interp(sep, sep_high, contrast_high)
    
    # Linear interpolation between magnitudes
    if mag_high == mag_low:
        C_final = C_low
    else:
        fraction = (interp_mag - mag_low) / (mag_high - mag_low)
        C_final = C_low + fraction * (C_high - C_low)
    
    # Define IWA and OWA from the data
    # IWA: minimum separation from the curve data
    IWA = min(sep_low[0], sep_high[0])  # Use minimum from both bracketing curves
    # OWA: use owa_extension from config

    if owa_extension == None:
        OWA = max(sep_low[0], sep_high[0])
    else:
        OWA = owa_extension
	
    
    # Apply IWA/OWA masking
    mincontrast = np.copy(sep) * 0.
    mincontrast[sep < IWA] = 1E100
    mincontrast[(IWA <= sep) & (sep <= OWA)] = C_final[(IWA <= sep) & (sep <= OWA)]
    mincontrast[sep > OWA] = 1E100
    
    return mincontrast




def load_transmission_curve(transmission_file, skiprows):
	"""
	Load transmission curve for the selected filter.
	
	Parameters:
	-----------
	transmission_file : str
		Filename of the transmission curve file
	skiprows : int
		Number of rows to skip when loading the file
		
	Returns:
	--------
	numpy array : Transmission curve data or None if file not found
	"""
	if transmission_file is None:
		return None
		
	original_dir = os.getcwd()
	try:
		#Transmission curves for METIS
		os.chdir(os.path.normpath(os.path.expandvars('$HOME/desktop/Master_Thesis_Sebastian/Filter Transmission/')))
		if transmission_file.lower().endswith('.csv'):
			Transmission_curve = np.loadtxt(transmission_file, skiprows=skiprows, delimiter=',')
			Transmission_curve[:,0] = Transmission_curve[:,0] / 1000.0  # Convert nm -> µm to match .dat files
		else:
			Transmission_curve = np.loadtxt(transmission_file, skiprows=skiprows)
		os.chdir(original_dir)
		return Transmission_curve
	except Exception as e:
		os.chdir(original_dir)
		print(f"Warning: Could not load transmission curve {transmission_file}: {e}")
		return None
	



def calculate_effective_albedo(Transmission_curve, Ag_spectrum=None):
	"""
	Calculates the effective albedo for the selected filter given its transmission curve.
	
	Parameters:
	-----------
	Transmission_curve : numpy array
		Transmission curve for the selected filter
	Ag_spectrum : numpy array, optional
		A 2D array [wavelengths, albedos]. 
		If None, a constant albedo of 0.3 is used.
		If provided, it will be interpolated to the transmission curve wavelengths.
	
	Returns:
	--------
	float : The effective albedo for the selected filter. Returns None if no transmission curve.
	"""
	if Transmission_curve is None:
		return None
		
	wav_thermal = Transmission_curve[:,0] * 1e-6  # Convert microns to meters
	
	if Ag_spectrum is None:
		# Use constant albedo of 0.3 if no spectrum provided
		Ag_arr = 0.3 * np.ones(len(Transmission_curve[:,0]))
	else:
		# Interpolate provided albedo spectrum to match transmission curve wavelengths
		# Assuming Ag_spectrum[:,0] is in microns, same as Transmission_curve[:,0]
		Ag_arr = np.interp(Transmission_curve[:,0], Ag_spectrum[:,0], Ag_spectrum[:,1])

	# Normalization factor (integral of transmission curve)
	normalization_Ag = np.trapezoid(Transmission_curve[:,1], wav_thermal)
	
	# Integrate albedo weighted by transmission curve
	Ag_val = np.trapezoid(Ag_arr * Transmission_curve[:,1], wav_thermal) / normalization_Ag
	return Ag_val



	



def get_detectability(contrast,sep):
	min_contrast_arr = minimum_contrast(sep)
	return (contrast > min_contrast_arr).astype(int)

def get_detectability_band(contrast, sep, band, mag, kmag=None):
	"""
    Get detectability for a specific METIS band.
   
    Parameters:
    -----------
    contrast : array
        Thermal contrast (Fp/Fstar) values
    sep : array
        Angular separations in mas
    band : str
        'N', 'L', or 'M'
    mag : float
        Stellar magnitude (band-appropriate, used for flux calculations)
    kmag : float, optional
        K-band stellar magnitude, used for METIS contrast curve interpolation
    
    Returns:
    --------
    array of 1s and 0s indicating detectability
    """
	min_contrast_arr = minimum_contrast_METIS(sep, band, mag, kmag=kmag)
    
	return (contrast > min_contrast_arr).astype(int)




def colored_line(x, y, c, ax, **lc_kwargs):

	# Default the capstyle to butt so that the line segments smoothly line up
	default_kwargs = {"capstyle": "butt"}
	default_kwargs.update(lc_kwargs)

	# Compute the midpoints of the line segments. Include the first and last points twice so we don't need any special syntax later to handle them.
	x = np.asarray(x)
	y = np.asarray(y)
	x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
	y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

	# Determine the start, middle, and end coordinate pair of each line segment. Use the reshape to add an extra dimension so each pair of points is in its own list. Then concatenate them.
	coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
	coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
	coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
	segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

	lc = LineCollection(segments, **default_kwargs)
	lc.set_array(c)  # set the colors of each segment

	return ax.add_collection(lc)



# Global cache for stellar spectra and Vega flux to avoid re-reading files in loops
_STELLAR_SPECTRUM_CACHE = {}
_VEGA_FLUX_CACHE = {}

def load_stellar_spectrum(star_name, verbose=False):
	"""
	Loads stellar spectrum from Working/BT-NextGen/{StarName}A/bt-nextgen_{StarName}A_range=(0.2, 20.0)µm_res=2000.fits.
	Returns: wav (microns), flx (W/m2/um)
	"""
	# Check cache first
	if star_name in _STELLAR_SPECTRUM_CACHE:
		return _STELLAR_SPECTRUM_CACHE[star_name]

	# Construct paths
	# Remove spaces from star name, e.g. "Beta Pic" -> "BetaPic"
	star_clean_name = star_name.replace(" ", "")
	
	# Special handling for Vega if needed, but assuming star_name='Vega' works for VegaA folder
	# If star_name is 'Vega', star_clean_name is 'Vega', folder is 'VegaA'
	
	star_folder_name = star_clean_name + "A"
	
	# Try loading from the configured directory
	if os.path.isabs(STELLAR_SPECTRUM_DIR):
		base_path = os.path.join(STELLAR_SPECTRUM_DIR, star_folder_name)
	else:
		base_path = os.path.join(os.getcwd(), STELLAR_SPECTRUM_DIR, star_folder_name)
	
	# Check if folder exists
	if not os.path.isdir(base_path):
		if verbose: print(f"  Stellar spectrum folder not found: {base_path}")
		return None, None

	# Find the FITS file inside
	spectrum_file = None
	# Try specifically with star name first
	for f in os.listdir(base_path):
		if f.startswith(f"bt-nextgen_{star_clean_name}") and f.endswith(".fits"):
			spectrum_file = os.path.join(base_path, f)
			break
	
	if spectrum_file is None:
		# Fallback: any bt-nextgen file
		for f in os.listdir(base_path):
			if f.startswith("bt-nextgen_") and f.endswith(".fits"):
				spectrum_file = os.path.join(base_path, f)
				break
			
	if spectrum_file is None:
		if verbose: print(f"  No bt-nextgen FITS file found in {base_path}")
		return None, None

	try:
		if verbose: print(f"  Loading stellar spectrum from {spectrum_file}")
		with fits.open(spectrum_file) as hdul:
			data = hdul[1].data
			# Columns: 'WAV' (um), 'FLX' (W/m2/um)
			# Copy data to memory to close file safety
			# Check column names if needed, assuming user confirmation
			# WAV and FLX
			wav = data['WAV'].copy()
			flx = data['FLX'].copy()
			
			_STELLAR_SPECTRUM_CACHE[star_name] = (wav, flx)
			return wav, flx
	except Exception as e:
		print(f"  Error loading stellar spectrum: {e}")
		return None, None

def get_vega_zero_point_flux_interpolated(filt_wav, filt_trans, verbose=False):
	"""
	Calculates the integrated flux of Vega through the filter.
	Cached by filter hash.
	Uses load_stellar_spectrum('Vega') to get the spectrum in consistent units.
	"""
	# Simple caching based on filter properties
	cache_key = (len(filt_wav), np.sum(filt_wav), np.sum(filt_trans))
	if cache_key in _VEGA_FLUX_CACHE:
		return _VEGA_FLUX_CACHE[cache_key]

	# Load Vega spectrum using the standard loader (handles path and units)
	# Assumes 'Vega' maps to 'Working/BT-NextGen/VegaA/bt-nextgen_VegaA...'
	v_wav_microns, v_flx_wm2um = load_stellar_spectrum('Vega', verbose=verbose)
	
	if v_wav_microns is None or v_flx_wm2um is None:
		if verbose: print("  Error: Could not load Vega spectrum.")
		return None

	try:
		# Check bounds before interpolation
		#print(v_wav_microns[0], filt_wav[-1],v_wav_microns[-1], filt_wav[0])
		if v_wav_microns[0] > filt_wav[-1] or v_wav_microns[-1] < filt_wav[0]:
			if verbose: print("  Error: Vega spectrum does not overlap with filter.")
			return None

		f_vega_interp = interp1d(v_wav_microns, v_flx_wm2um, bounds_error=False, fill_value=0.0)(filt_wav)
		flux_vega_integrated = np.trapezoid(f_vega_interp * filt_trans, x=filt_wav)
		
		_VEGA_FLUX_CACHE[cache_key] = flux_vega_integrated
		return flux_vega_integrated

	except Exception as e:
		if verbose: print(f"  Error processing Vega spectrum: {e}")
		return None


def get_stellar_magnitude_for_band(planet_dict, mag_priority_list, Transmission_curve=None, verbose=False):
    """
    Determines the best stellar magnitude to use for the selected filter.
    Uses the priority list from the filter configuration.
    If USE_STELLAR_SPECTRUM_FILE is True, it calculates the magnitude from the stellar spectrum.
    
    Parameters:
    -----------
    planet_dict : dict
        Dictionary containing planet/star parameters
    mag_priority_list : list
        List of magnitude keys in priority order (e.g., ['sy_w1mag', 'sy_kmag', 'sy_vmag'])
    Transmission_curve : array (optional)
        Transmission curve [wavelength_um, transmission]
    verbose : bool, optional
        Print debug information
    
    Returns:
    --------
    float : Stellar magnitude or None if not found
    """
    
    # 1. Try calculation from Spectrum File if enabled
    if USE_STELLAR_SPECTRUM_FILE and Transmission_curve is not None:
        star_name = planet_dict.get('pl_hostname', '')
        # Try fallback to pl_name if hostname missing (remove planet letter)
        if not star_name and 'pl_name' in planet_dict and planet_dict['pl_name']:
             # Heuristic: Remove trailing letters ' b', ' c', ' d' if present
             try:
                 star_name = planet_dict['pl_name']
                 for suffix in [' b', ' c', ' d', ' e', ' f', ' g', ' h']:
                     star_name = star_name.replace(suffix, '')
             except:
                 pass

        if star_name:
             wav, flx = load_stellar_spectrum(star_name, verbose=verbose)
             if wav is not None and flx is not None:
                 filt_wav = Transmission_curve[:, 0]
                 filt_trans = Transmission_curve[:, 1]
                 
                 # Calculate Vega Flux (Zero Point)
                 flux_vega_integrated = get_vega_zero_point_flux_interpolated(filt_wav, filt_trans, verbose=verbose)
                 
                 if flux_vega_integrated is not None and flux_vega_integrated > 0:
                     # Calculate Star Flux
                     if wav[0] <= filt_wav[-1] and wav[-1] >= filt_wav[0]:
                         f_star_interp = interp1d(wav, flx, bounds_error=False, fill_value=0.0)(filt_wav)
                         flux_star_integrated = np.trapezoid(f_star_interp * filt_trans, x=filt_wav)
                         
                         if flux_star_integrated > 0:
                             mag = -2.5 * np.log10(flux_star_integrated / flux_vega_integrated)
                             if verbose:
                                 print(f"  Calculated magnitude {mag:.3f} from stellar spectrum for {star_name}")
                             return mag
                         elif verbose:
                             print(f"  Warning: Integrated flux for {star_name} is non-positive.")
                     elif verbose:
                         print(f"  Warning: Star spectrum for {star_name} does not overlap with filter.")
                 elif verbose:
                     print("  Warning: Could not calculate Vega zero point.")
             elif verbose:
                 print(f"  No stellar spectrum file found for {star_name}. Falling back to catalog magnitude.")

    # 2. Fallback to Catalog Values
    for mag_key in mag_priority_list:
        if mag_key in planet_dict and planet_dict[mag_key] != '':
            if verbose:
                print(f"  Found magnitude {planet_dict[mag_key]} from {mag_key}")
            return float(planet_dict[mag_key])
    
    # If no magnitude found in priority list, return None
    if verbose:
        print(f"  Warning: No stellar magnitude found in priority list {mag_priority_list}")
    return None



    
    #return found_mag if found_mag is not None else mag

def run_multiple_orbital_simulations(dictionary, dict_aux, key, use_mass2mag=False, use_madys=False):
	#Before the 21.05.2025, this function was part of the main body. 
	#Now it is a function to make it easier to run several orbital configurations for a given planet (e.g. before and after some astrometry measurements)
	dictionary[key]['pl_orbsmax_flag_computedOCG'] = ''
	dictionary[key]['pl_orbper_flag_computedOCG'] = ''
	dictionary[key]['pl_eqt_flag_computedOCG'] = ''
	detectable, ptransit_arr = [], []
	Xcoord_arr, Ycoord_arr, Fp_Fstar_max_arr, Fp_Fstar_arr, angproj_arr, alphas_arr, trueanom_arr, t_tp_arr, observ_alpha, observ_time, observ_dist, observ_Teq, NOTobserv_Teq, observ_X, observ_Y, NOTobserv_X, NOTobserv_Y = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
	Teq_arr, observ_Teq_arr, NOTobserv_Teq_arr = [], [], []
	dates_arr, observ_angproj, observ_Fp_Fstar, NOTobserv_angproj, NOTobserv_Fp_Fstar, values_orbper, values_orbtper = [], [], [], [], [], [], []
	# Single-band thermal arrays (no longer separate N, L, M)
	####SEBASTIAN######
	Fp_Fstar_thermal_arr = []
	detectable_thermal = []
	observ_alpha_thermal = []
	observ_X_thermal = []
	observ_Y_thermal = []
	NOTobserv_X_thermal = []
	NOTobserv_Y_thermal = []
	Fp_Fstar_total_arr, observ_Fp_Fstar_total, NOTobserv_Fp_Fstar_total = [], [], []
	observ_angproj_thermal, NOTobserv_angproj_thermal, observ_Fp_Fstar_thermal, NOTobserv_Fp_Fstar_thermal = [], [], [], []
	
	# Load transmission curve for the selected filter
	Transmission_curve = load_transmission_curve(transmission_curve_file, transmission_skiprows) 
	#Fp_Fstar_arr_iconstrained, angproj_arr_iconstrained, observ_alpha_iconstrained, observ_X_iconstrained, observ_Y_iconstrained, NOTobserv_X_iconstrained, NOTobserv_Y_iconstrained = [], [], [], [], [], [], [], []
	values_ecc, values_incl, values_orblper = [], [], []
	values_orbsmax, values_orbper, values_Mp, values_Rp, values_eqt, values_eqtOCG, values_st_mass, values_st_teff, values_sy_dist, values_st_rad = [], [], [], [], [], [], [], [], [], []
	values_Temp = [] # Store temperatures from mass2mag
	values_abs_mag = [] # Store absolute magnitudes from mass2mag
	values_age_gyr = [] # Store ages from mass2mag
	values_sy_vmag, values_st_optmag, values_sy_kmag, values_sy_w1mag, values_sy_w2mag, values_sy_w3mag = [], [], [], [], [], []


	keys = ['sy_dist', 'pl_orbper', 'pl_orbsmax', 'pl_bmassj', 'pl_radj', 'pl_orbincl', 'pl_orbeccen', 'pl_orblper', 'pl_eqt', 'st_teff', 'st_mass', 'st_age', 'sy_vmag', 'st_optmag', 'sy_kmag', 'sy_w1mag', 'sy_w2mag', 'sy_w3mag'] 
	#define the physical limmits for each key.  'st_rad'
	dict_limits = {}
	values_params = [values_sy_dist, values_orbper, values_orbsmax, values_Mp, values_Rp, values_incl, values_ecc, values_orblper, values_eqt, values_st_teff, values_st_mass, values_age_gyr]  # values_st_rad removed (12 params for 4x3 grid)
	paramlabels = [
    r"$d$ [pc]", r"$P$ [days]", r"$a$ [AU]", r"$M_p$ [$M_J$]", r"$R_p$ [$R_J$]", r"$i$ [deg]", r"$e$", r"$\omega$ [deg]", r"$T_{eq}$ [K]", r"$T_\star$ [K]", r"$M_\star$ [$M_\odot$]", r"Age [Gyr]"  # r"$R_\star$ [$R_\odot$]" removed
]
	#print(values_Mp)

	count_noplorbmax = 0
	count_noplorbper = 0

	for n in range(nrun):
		#New orbital realization. 
		#For details, see Sect. 4 of of Carrion-Gonzalez et al (2021), A&A 651, A7
		if SPLIT_GAUSSIAN == False:
			#sprint('using bootstraping')
			dict_aux[key] = bootstrapping(keys, dictionary[key], dict_aux[key])
		if SPLIT_GAUSSIAN == True:
			#print('using split normal sampling')
			dict_aux[key] = split_normal_sampling(keys, dictionary[key], dict_aux[key])



		#First, checking if there is an stored orbsmax value for this planet. If the field is empty, I try to compute it from the period
		if dictionary[key]['pl_orbsmax'] == '':		
			dict_aux[key]['pl_orbsmax'], count_noplorbmax, flag_nomassplanet = orbperiod_2_orbsemimajorax(dict_aux[key]['pl_orbper'], dict_aux[key]['st_mass'], dict_aux[key]['pl_bmassj'], count_noplorbmax)
			if dict_aux[key]['pl_orbsmax'] != '':
				dictionary[key]['pl_orbsmax_flag_computedOCG'] = '*'+flag_nomassplanet
			else:
				dictionary[key]['pl_orbsmax_flag_computedOCG'] = ''
			flag_nomassplanet = ''

		#Now, I check if we have a value for the orbital period. If not, I try to compute it from the semimajor axis
		if dictionary[key]['pl_orbper'] == '':		
			dict_aux[key]['pl_orbper'], count_noplorbper, flag_nomassplanet = orbsemimajorax_2_orbperiod(dict_aux[key]['pl_orbsmax'], dict_aux[key]['st_mass'], dict_aux[key]['pl_bmassj'], count_noplorbper)
			if dict_aux[key]['pl_orbper'] != '':
				dictionary[key]['pl_orbper_flag_computedOCG'] = '*'+flag_nomassplanet
			else:
				dictionary[key]['pl_orbper_flag_computedOCG'] = ''
			flag_nomassplanet = ''


		################################################################
		if dictionary[key]['pl_orbeccen'] == '':	#If no value of e is in the NASA catalogue, we take one from our distribution
			dict_aux[key]['pl_orbeccen'] = random.uniform(0.,1.)
		else:
			if dict_aux[key]['pl_orbeccen'] != '' and float(dict_aux[key]['pl_orbeccen']) < 0.:
				dict_aux[key]['pl_orbeccen'] = 0.
	
		if dictionary[key]['pl_orbincl'] == '':		#Same for i
			dict_aux[key]['pl_orbincl'] = np.arccos(random.uniform(-1.,1.))*180./np.pi

		if dictionary[key]['pl_orblper'] == '':		#Same for omega
			dict_aux[key]['pl_orblper'] = random.uniform(0.,2.*np.pi)*180./np.pi

		#Here making sure that the new value of incl affects the mass of the planet if this was given as Msini in the catalogue
		if dictionary[key]['pl_bmassprov'] == 'Msini' and dictionary[key]['pl_bmassj']!='' and dict_aux[key]['pl_bmassj'] != '' and dict_aux[key]['pl_orbincl'] != '':
			dict_aux[key]['pl_bmassj'] = float(dict_aux[key]['pl_bmassj'])/np.sin(float(dict_aux[key]['pl_orbincl'])*np.pi/180.)
			dict_aux[key]['pl_bmassprov'] = 'Msin(i)/sin(i)' 

		#Here checking whether we have a value of Rp or a value of Mp to compute the radius
		if dict_aux[key]['pl_orbincl'] != '':
			dict_aux[key] = give_planet_radius(dict_aux[key], float(dict_aux[key]['pl_orbincl']))
		else:
			dict_aux[key] = give_planet_radius(dict_aux[key], 0.0)  # Use 0 as default if inclination is missing
		if dict_aux[key]['pl_radj_run_n']=='': 
			break	#If I could not obtain a value of Rp, we cannot process this planet

		#Here computing the Teq at the semimajor axis
		if dict_aux[key]['st_rad'] != '' and dict_aux[key]['st_teff'] != '' and dict_aux[key]['pl_orbsmax'] != '':
			dict_aux[key]['pl_eqtOCG'] = np.power((1.-Abond)/4., 0.25)*np.power(float(dict_aux[key]['st_rad'])*Rsun/(float(dict_aux[key]['pl_orbsmax'])*AU_constant), 0.5)*float(dict_aux[key]['st_teff'])
		else:
			dict_aux[key]['pl_eqtOCG'] = ''


		values_orbsmax.append(float(dict_aux[key]['pl_orbsmax']))
		values_Rp.append(float(dict_aux[key]['pl_radj_run_n']))
		if dict_aux[key]['pl_orbper']!='':
			values_orbper.append(float(dict_aux[key]['pl_orbper']))
		if dict_aux[key]['pl_orbtper']!='':
			values_orbtper.append(float(dict_aux[key]['pl_orbtper']))
		if dict_aux[key]['pl_bmassj']!='':
			values_Mp.append(float(dict_aux[key]['pl_bmassj']))
		if dict_aux[key]['pl_eqt']!='':
			values_eqt.append(float(dict_aux[key]['pl_eqt']))
		if dict_aux[key]['pl_eqtOCG']!='':
			values_eqtOCG.append(float(dict_aux[key]['pl_eqtOCG']))
		if dict_aux[key]['st_mass']!='':
			values_st_mass.append(float(dict_aux[key]['st_mass']))
		if dict_aux[key]['st_rad']!='':
			values_st_rad.append(float(dict_aux[key]['st_rad']))
		if dict_aux[key]['st_teff']!='':
			values_st_teff.append(float(dict_aux[key]['st_teff']))
		values_sy_dist.append(float(dict_aux[key]['sy_dist']))
		values_ecc.append(dict_aux[key]['pl_orbeccen'])
		values_incl.append(dict_aux[key]['pl_orbincl'])
		values_orblper.append(dict_aux[key]['pl_orblper'])
		

		#Computing the transit probability
		if dictionary[key]['st_rad']!='':
			ptransit_arr.append(does_it_transit(dict_aux[key]))







		#Computing along the orbit Fp/Fstar, delta_theta etc. for this orbital realization
		if dict_aux[key]['sy_dist']!='':
			#Xcoord, Ycoord, t_tpi, trueanomi, angproji, Fp_Fstari, alphasi, dist_pl_st = orbit_evolution(aorbit, incl, ecc, longperiast, Rp, Ag, dist)




			
			# Calculate effective albedo for the selected filter
			Ag = calculate_effective_albedo(Transmission_curve, Ag_spectrum)
			if Ag is None:
				#print("Effective albedo calculation returned None, using default value 0.3")
				Ag = 0.3  # Default fallback
			#print(f"Using Albedo: {Ag}")
			Xcoordi, Ycoordi, t_tpi, trueanomi, angproji, Fp_Fstari, alphasi, dist_pl_st = orbit_evolution(float(dict_aux[key]['pl_orbsmax']), float(dict_aux[key]['pl_orbincl']), float(dict_aux[key]['pl_orbeccen']), float(dict_aux[key]['pl_orblper']), float(dict_aux[key]['pl_radj_run_n']), Ag, float(dict_aux[key]['sy_dist']))
			Xcoord_arr.append(Xcoordi)
			Ycoord_arr.append(Ycoordi)
			Fp_Fstar_max_arr.append(np.amax(Fp_Fstari))
			Fp_Fstar_arr.append(Fp_Fstari)
			#print(Fp_Fstar_arr)
			angproj_arr.append(angproji)
			alphas_arr.append(alphasi)
			trueanom_arr.append(trueanomi)
			t_tp_arr.append(t_tpi)
			if dictionary[key]['pl_orbtper']!='':
				datesi = np.asarray(t_tpi)*float(dict_aux[key]['pl_orbper'])+float(dict_aux[key]['pl_orbtper'])	#Dates at an arbitrary date given by the pl_orbtper
				dates_arr.append(datesi)
			if dict_aux[key]['st_rad'] != '' and dict_aux[key]['st_teff'] != '' and dict_aux[key]['pl_orbsmax'] != '':
			#if len(values_st_rad)>1 and len(values_st_teff)>1:
				st_radi = float(dict_aux[key]['st_rad'])#values_params[keys.index('st_rad')][i]
				st_teffi = float(dict_aux[key]['st_teff'])#values_params[keys.index('st_teff')][i]
				Teq_orbit = np.power((1.-Abond)/4., 0.25)*np.power(float(st_radi)*Rsun/(dist_pl_st*AU_constant), 0.5)*float(st_teffi)
			else:
				Teq_orbit = np.ones(len(dist_pl_st))
			
			# Get stellar magnitude for the selected filter
			st_mag = get_stellar_magnitude_for_band(dictionary[key], stellar_mag_priority, Transmission_curve=Transmission_curve, verbose=(n==0))
			if st_mag is None:
				print('no stellar magnitude found using 5 mag')
				st_mag = 5.0  # Default fallback

			# K-band magnitude for METIS contrast curve interpolation (curves are labelled by K-mag)
			kmag_val = float(dictionary[key]['sy_kmag']) if dictionary[key].get('sy_kmag', '') != '' else None
				
			# Determine band letter for detectability check
			band_letter = FILTER_USED
			#Computing the thermal Flux of Star to Planet 

			# Get Mass and Age for mass2mag if needed
			mass_mjup = None
			#age_gyr = 4.85
			if  mass_mjup is None:
				if dict_aux[key]['pl_bmassj'] != '':
					mass_mjup = float(dict_aux[key]['pl_bmassj'])
					
				if 'st_age' in dict_aux[key]:
					if dict_aux[key]['st_age'] != '':
						age_gyr = float(dict_aux[key]['st_age'])
					#print('system age [gyr]', age_gyr)
			
			
			values_age_gyr.append(age_gyr)

			

			# Calculate thermal flux for the selected filter
			if FILTER_USED.startswith('METIS') and kmag_val is not None:
				if n == 0:
					print(f"Using K-band Magnitude {kmag_val:.2f} for {FILTER_USED} contrast curve interpolation.")
			else:
				if n == 0:
					print(f"Using Apparent Magnitude {st_mag:.2f} for {FILTER_USED} contrast curve interpolation.")
				
			temp_current_run = None
			abs_mag_current_run = None
			
			if use_madys:
				thermal_orbiti, temp_planet, abs_mag_planet = thermal_flux_planet_MADYS(Teq_orbit, float(dict_aux[key]['sy_dist']), band_letter, st_mag, mass_mjup, age_gyr, Madys_Modell_selection)
				temp_current_run = temp_planet
				abs_mag_current_run = abs_mag_planet
				#print(thermal_orbiti)
			elif use_mass2mag:
				thermal_orbiti, temp_planet, abs_mag_planet = thermal_flux_planet_mass2mag(Teq_orbit, float(dict_aux[key]['sy_dist']), band_letter, st_mag, mass_mjup, age_gyr)
				temp_current_run = temp_planet[0] if isinstance(temp_planet, (list, np.ndarray)) else temp_planet
				abs_mag_current_run = abs_mag_planet[0] if isinstance(abs_mag_planet, (list, np.ndarray)) else abs_mag_planet
			elif use_blackbody:
				if Transmission_curve is not None:
					wav_thermal = Transmission_curve[:,0] * 1e-6  # Convert microns to meters
					thermal_orbiti = thermal_flux_planet(Teq_orbit, float(dict_aux[key]['pl_radj_run_n']), float(dict_aux[key]['st_rad']), float(dict_aux[key]['st_teff']), wav_thermal, Transmission_curve[:,1])
				else:
					print(f"Warning: No transmission curve available for {FILTER_USED}, skipping thermal calculation")
					thermal_orbiti = np.zeros(len(Teq_orbit))
			else:
				# No thermal calculation method selected
				thermal_orbiti = np.zeros(len(Teq_orbit))
			
			Fp_Fstar_thermal_arr.append(thermal_orbiti)
			if RUN_REFLECTED_LIGHT:
				Fp_Fstari_total = Fp_Fstari + thermal_orbiti
				Fp_Fstar_total_arr.append(Fp_Fstari_total)
				#print(Fp_Fstar_arr)
			else:
				Fp_Fstar_arr.append(Fp_Fstari)

			


			if RUN_REFLECTED_LIGHT:
				# Total Detectability
				SNRTi_total = get_detectability_band(np.asarray(Fp_Fstari_total), np.asarray(angproji), band_letter, st_mag, kmag=kmag_val)
				observ_arr_total = np.where(np.asarray(SNRTi_total)==1.)[0]
				
				# Reflected Only Detectability
				SNRTi_reflected = get_detectability_band(np.asarray(Fp_Fstari), np.asarray(angproji), band_letter, st_mag, kmag=kmag_val)
				observ_arr_reflected = np.where(np.asarray(SNRTi_reflected)==1.)[0]

				observ_arr = observ_arr_total # Maintain original variable name for compatibility

				observ_alpha.append([alphasi[x] if x in observ_arr else np.nan for x in range(len(alphasi))])
				observ_X.append([Xcoordi[x] if x in observ_arr else np.nan for x in range(len(Xcoordi))])
				NOTobserv_X.append([Xcoordi[x] if x not in observ_arr else np.nan for x in range(len(Xcoordi))])
				observ_Y.append([Ycoordi[x] if x in observ_arr else np.nan for x in range(len(Ycoordi))])
				NOTobserv_Y.append([Ycoordi[x] if x not in observ_arr else np.nan for x in range(len(Ycoordi))])
				observ_angproj.append([angproji[x] if x in observ_arr else np.nan for x in range(len(angproji))])
				NOTobserv_angproj.append([angproji[x] if x not in observ_arr else np.nan for x in range(len(angproji))])
				
				# Store based on TOTAL detectability (for plots)
				observ_Fp_Fstar_total.append([Fp_Fstari_total[x] if x in observ_arr else np.nan for x in range(len(Fp_Fstari_total))])
				NOTobserv_Fp_Fstar_total.append([Fp_Fstari_total[x] if x not in observ_arr else np.nan for x in range(len(Fp_Fstari_total))])
				
				# OLD: Store reflected component of TOTAL detectability
				# observ_Fp_Fstar.append([Fp_Fstari[x] if x in observ_arr else np.nan for x in range(len(Fp_Fstari))])

				observ_Fp_Fstar.append([Fp_Fstari[x] if x in observ_arr_reflected else np.nan for x in range(len(Fp_Fstari))])

				NOTobserv_Fp_Fstar.append([Fp_Fstari[x] if x not in observ_arr_reflected else np.nan for x in range(len(Fp_Fstari))])				
			else:
				observ_arr = []
				observ_alpha.append([np.nan for x in range(len(alphasi))])
				observ_X.append([np.nan for x in range(len(Xcoordi))])
				NOTobserv_X.append([Xcoordi[x] for x in range(len(Xcoordi))])
				observ_Y.append([np.nan for x in range(len(Ycoordi))])
				NOTobserv_Y.append([Ycoordi[x] for x in range(len(Ycoordi))])
				observ_angproj.append([np.nan for x in range(len(angproji))])
				NOTobserv_angproj.append([angproji[x] for x in range(len(angproji))])
				observ_Fp_Fstar_total.append([np.nan for x in range(len(Fp_Fstari_total))])
				NOTobserv_Fp_Fstar_total.append([Fp_Fstari_total[x] for x in range(len(Fp_Fstari_total))])
				observ_Fp_Fstar.append([np.nan for x in range(len(Fp_Fstari))])
				NOTobserv_Fp_Fstar.append([Fp_Fstari[x] for x in range(len(Fp_Fstari))])	

			'''if dict_aux[key]['pl_orbincl_newconstraint']!='':
				if float(dict_aux[key]['pl_orbincl']) <= (float(dict_aux[key]['pl_orbincl_newconstraint'])+float(dict_aux[key]['pl_orbincl_newconstrainterr1'])) and float(dict_aux[key]['pl_orbincl']) >= (float(dict_aux[key]['pl_orbincl_newconstraint'])+float(dict_aux[key]['pl_orbincl_newconstrainterr2'])):
					Fp_Fstar_arr_iconstrained.append(Fp_Fstari)
					angproj_arr_iconstrained.append(angproji)
					observ_alpha_iconstrained.append([alphasi[x] if x in observ_arr else np.nan for x in range(len(alphasi))])
					observ_X_iconstrained.append([Xcoordi[x] if x in observ_arr else np.nan for x in range(len(Xcoordi))])
					NOTobserv_X_iconstrained.append([Xcoordi[x] if x not in observ_arr else np.nan for x in range(len(Xcoordi))])
					observ_Y_iconstrained.append([Ycoordi[x] if x in observ_arr else np.nan for x in range(len(Ycoordi))])
					NOTobserv_Y_iconstrained.append([Ycoordi[x] if x not in observ_arr else np.nan for x in range(len(Ycoordi))])'''
								
			if 'Teq_orbit' in locals():#len(values_st_rad)>1 and len(values_st_teff)>1:
				observ_Teq = [Teq_orbit[x] if x in observ_arr else np.nan for x in range(len(Teq_orbit))]
				NOTobserv_Teq = [Teq_orbit[x] if x not in observ_arr else np.nan for x in range(len(Teq_orbit))]
			if len(observ_arr)>0:
				detectable.append(1)
			else:
				detectable.append(0)
		if 'Teq_orbit' in locals():#len(values_st_rad)>1 and len(values_st_teff)>1:
		#if dict_aux[key]['st_rad'] != '' and dict_aux[key]['st_teff'] != '' and dict_aux[key]['pl_orbsmax'] != '':
			Teq_arr.append(Teq_orbit)
			observ_Teq_arr.append(observ_Teq)
			NOTobserv_Teq_arr.append(NOTobserv_Teq)

			
			#print(Fp_Fstar_thermal_arr)

			SNRTi = get_detectability_band(np.asarray(thermal_orbiti), np.asarray(angproji), band_letter, st_mag, kmag=kmag_val)
			observ_arr_thermal = np.where(np.asarray(SNRTi) == 1.)[0]
			
			observ_alpha_thermal.append([alphasi[x] if x in observ_arr_thermal else np.nan for x in range(len(alphasi))])
			observ_X_thermal.append([Xcoordi[x] if x in observ_arr_thermal else np.nan for x in range(len(Xcoordi))])
			NOTobserv_X_thermal.append([Xcoordi[x] if x not in observ_arr_thermal else np.nan for x in range(len(Xcoordi))])
			observ_Y_thermal.append([Ycoordi[x] if x in observ_arr_thermal else np.nan for x in range(len(Ycoordi))])
			NOTobserv_Y_thermal.append([Ycoordi[x] if x not in observ_arr_thermal else np.nan for x in range(len(Ycoordi))])
			observ_angproj_thermal.append([angproji[x] if x in observ_arr_thermal else np.nan for x in range(len(angproji))])
			NOTobserv_angproj_thermal.append([angproji[x] if x not in observ_arr_thermal else np.nan for x in range(len(angproji))])
			observ_Fp_Fstar_thermal.append([thermal_orbiti[x] if x in observ_arr_thermal else np.nan for x in range(len(thermal_orbiti))])
			NOTobserv_Fp_Fstar_thermal.append([thermal_orbiti[x] if x not in observ_arr_thermal else np.nan for x in range(len(thermal_orbiti))])	
			
			if len(observ_arr_thermal) > 0:
				detectable_thermal.append(1)
			else:
				detectable_thermal.append(0)

			if temp_current_run is not None:
				values_Temp.append(temp_current_run)
			
			if abs_mag_current_run is not None:
				values_abs_mag.append(abs_mag_current_run)




			



		#print(len(t_tpi), len(Teq_orbit), len(observ_Teq), len(NOTobserv_Teq))


	observ_X = np.asarray(observ_X)
	NOTobserv_X = np.asarray(NOTobserv_X)
	observ_Y = np.asarray(observ_Y)
	NOTobserv_Y = np.asarray(NOTobserv_Y)
	#print("Planet %s done. Detectable in %d out of %d orbital simulations.\n"%(dictionary[key]['pl_name'], sum(detectable), nrun))
	print("Planet %s done." % dictionary[key]['pl_name'])
	
	print("  %d/%d orbital realizations are detectable at some point of their orbit (%s - Thermal)" % (sum(detectable_thermal), nrun, FILTER_USED))
	
	if RUN_REFLECTED_LIGHT:
		print("  %d/%d orbital realizations are detectable at some point of their orbit (for total flux)" % (sum(detectable), nrun))

	
	if PLOT_NEWCOLUMN_SUBPLOTS == True:
		# Grid plot: 4 rows x 3 columns (12 subplots, no empty cells)
		fig2, axs2 = plt.subplots(4, 3, sharex=False, sharey=False, figsize=(9, 8))
		axs2_flat = axs2.flatten()
		fig2, axs2_flat = plot_newcolumn_subplots(fig2, axs2_flat, 1, values_params, values_Temp, paramlabels, dictionary[key], dict_aux[key], keys)
		fig2.suptitle(dictionary[key]['pl_name'], fontsize=12)
		plt.tight_layout()
		figname2 = '%s/%s_onecolumnplot_nrun_%s'%(route, dictionary[key]['pl_name'], str(nrun))
		fig2.savefig(figname2+'.png', bbox_inches='tight')
		fig2.savefig(figname2+'.pdf', bbox_inches='tight')
		plt.close(fig2)
	
	return detectable, angproj_arr, Fp_Fstar_arr, observ_alpha, observ_X, observ_Y, NOTobserv_X, NOTobserv_Y, t_tp_arr, Teq_arr, observ_Teq_arr, NOTobserv_Teq_arr, Fp_Fstar_thermal_arr, detectable_thermal, observ_alpha_thermal, observ_X_thermal, observ_Y_thermal, NOTobserv_X_thermal, NOTobserv_Y_thermal, values_Mp, values_Temp, values_ecc, values_incl, values_orblper, values_abs_mag, values_age_gyr, Transmission_curve, dates_arr, observ_angproj, observ_Fp_Fstar, NOTobserv_angproj, NOTobserv_Fp_Fstar, values_orbper, values_orbtper, observ_angproj_thermal, NOTobserv_angproj_thermal, observ_Fp_Fstar_thermal, NOTobserv_Fp_Fstar_thermal, Fp_Fstar_total_arr, observ_Fp_Fstar_total, NOTobserv_Fp_Fstar_total


def detectability_windows(t_tp_arr, dates_arr, angproj_arr, Fp_Fstar_arr, observ_angproj, observ_Fp_Fstar, NOTobserv_angproj, NOTobserv_Fp_Fstar, values_orbper, values_orbtper):#startdate, enddate, ):
	#Plotting the detectability windows vs. time
	
	textsize = 16
	figname = '%s/%s_nrun%s_TIME-dep_%s_DetectWindow%s'%(route, dictionary[key]['pl_name'], str(nrun), namerun, Time(startdetectwindow, format='jd').isot[0:7:1])
	if 'dictio_alternative' in locals():
		figname = '%s/%s_nrun%s_TIME-dep_%s_DetectWindow%s_WITH-i-constraints'%(route, dictionary[key]['pl_name'], str(nrun), namerun, Time(startdetectwindow, format='jd').isot[0:7:1])
	if RUN_REFLECTED_LIGHT == False: 
		figname = figname + "_thermal"

	
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))
	ax1.set_title(dictionary[key]['pl_name'], fontsize=textsize+2)

	'''dates_iso_full = Time(dates_arr[0], format='jd')
	print(dates_arr[0])
	print()
	print(dates_iso_full.isot)
	print()
	dates_iso = [date[0:7:1] for date in dates_iso_full.isot]
	print(dates_iso)
	plt.plot(dates_iso, angproj_arr[0], '-k', alpha=0.1)'''
	

	for i in range(len(t_tp_arr)):
		n_orbper = int((startdetectwindow - dates_arr[i][0])/values_orbper[i])
		#print(t_tp_arr[i], values_orbper[i], values_orbtper[i], n_orbper, values_orbper[i])	
		dateswindow1 = t_tp_arr[i]*values_orbper[i]+values_orbtper[i]+(n_orbper*values_orbper[i])	#The above array of dates, multiplied by an integer number of orbital periods such that the initial date of the detectability window is included
		dateswindow2 = t_tp_arr[i]*values_orbper[i]+values_orbtper[i]+((n_orbper+1)*values_orbper[i])
		dateswindow3 = t_tp_arr[i]*values_orbper[i]+values_orbtper[i]+((n_orbper+2)*values_orbper[i])
		dateswindow = np.concatenate((dateswindow1,dateswindow2,dateswindow3))
		#For now, we concatenate two orbits because sometimes the large uncertainties (e.g. for pl_orbtper or pl_orbper) might mislead the final detectability plots. Better to be sure that no orbital simulations are suddenly stopping at the middle of the desired window, misleading making us think it would not be detectable
		#Probably this option to concatenate orbits should be done automatically, in such a way that we concatenate orbits until we hit the end of the desired window.

		#ax1.plot(dates_arr[i], NOTobserv_angproj[i], '-k', alpha=0.1)
		#ax1.plot(dates_arr[i], observ_angproj[i], linestyle='-', color='limegreen', alpha=0.1)
		#ax2.plot(dates_arr[i], NOTobserv_Fp_Fstar[i], '-k', alpha=0.1)
		#ax2.plot(dates_arr[i], observ_Fp_Fstar[i], linestyle='-', color='limegreen', alpha=0.1)
		ax1.plot(dateswindow, np.concatenate((NOTobserv_angproj[i],NOTobserv_angproj[i],NOTobserv_angproj[i])), '-k', alpha=0.1)
		ax1.plot(dateswindow, np.concatenate((observ_angproj[i], observ_angproj[i], observ_angproj[i])), linestyle='-', color='limegreen', alpha=0.1)
		ax2.plot(dateswindow, np.concatenate((NOTobserv_Fp_Fstar[i], NOTobserv_Fp_Fstar[i], NOTobserv_Fp_Fstar[i])), '-k', alpha=0.1)
		ax2.plot(dateswindow, np.concatenate((observ_Fp_Fstar[i], observ_Fp_Fstar[i], observ_Fp_Fstar[i])), linestyle='-', color='limegreen', alpha=0.1)
	#plt.show()
	ax2.set_xlim(xmin=startdetectwindow)
	xmaxi = ax2.get_xlim()[1]
	if xmaxi > enddetectwindow:
		ax2.set_xlim(xmax=enddetectwindow)
	ax2.minorticks_on()
	fig.savefig(figname+'.png', bbox_inches='tight')
	dates_labels = ax2.get_xticklabels()
	#print([float(str(d).replace('Text(', '').split(',')[0]) for d in dates_labels])
	dates_iso_full = Time([float(str(d).replace('Text(', '').split(',')[0]) for d in dates_labels], format='jd')
	#print(dates_iso_full)
	#print()
	dates_iso = [date[0:7:1] for date in dates_iso_full.isot]
	#print(dates_iso)
	ax2.set_xticklabels(dates_iso)
	ax1.axhline(IWA, color='g', linestyle='--')
	if OWA < 1.2* np.max(observ_angproj):
		ax1.axhline(OWA, color='g', linestyle='--')
	#ax2.axhline(Cmin, color='g', linestyle='--')
	ax1.set_ylabel("Ang. sep [mas]", fontsize=textsize)
	ax2.set_ylabel("$F_p$/$F_s$", fontsize=textsize)
	ax2.set_xlabel("Date", fontsize=textsize)
	ax1.tick_params(labelsize=textsize-2)
	ax2.tick_params(labelsize=textsize-2)
	ymini, ymaxi = ax2.get_ylim()

	#if ymaxi/Cmin > 1E4:
	#	ax2.set_yscale('log')
	#plt.show()
	fig.savefig(figname+'.png', bbox_inches='tight')
	fig.savefig(figname+'.pdf', bbox_inches='tight')
	print("SAVED: %s.pdf"%figname)
	plt.close(fig)
	#quit()

def analyze_march_2027(planet_name, dates_arr, 
					   thermal_flux_arr, reflected_flux_arr, total_flux_arr, 
					   observ_thermal_arr, observ_reflected_arr, observ_total_arr,
					   csv_filename='March2027_Analysis.csv', run_reflected=False,
					   split_gaussian=False, filter_name='Unknown', entropy_model='Unknown'):
	
	import csv
	
	# March 2027 JD Range
	# Calculated earlier: 2461465.5 to 2461496.5
	MARCH_START = 2461465.5
	MARCH_END = 2461496.5
	MARCH_MID = 2461481.0
	
	all_thermal = []
	all_reflected = []
	all_total = []
	
	total_points_count = 0
	detectable_points_thermal = 0
	detectable_points_reflected = 0
	detectable_points_total = 0
	
	n_runs = len(dates_arr)
	
	if n_runs == 0:
		return

	# Check if dates exist (if pl_orbtper was present, dates_arr[0] should be non-empty)
	if len(dates_arr) == 0 or len(dates_arr[0]) == 0:
		print(f"Skipping March 2027 analysis for {planet_name}: No Time of Periastron (dates array empty).")
		return

	for i in range(n_runs):
		dates = np.array(dates_arr[i])
		
		# Indices in March 2027
		idx_march = np.where((dates >= MARCH_START) & (dates <= MARCH_END))[0]
		
		if len(idx_march) == 0:
			# Find closest to mid-march
			# "Just use the 2 closest positions before and after March in that case"
			#print('no values found in march 2027, could increase 360 points per orbit')
			idx_closest = np.searchsorted(dates, MARCH_MID)
			
			# Handle edge cases
			if idx_closest == 0:
				indices_to_use = [0, 1] if len(dates) > 1 else [0]
			elif idx_closest >= len(dates):
				indices_to_use = [len(dates)-2, len(dates)-1] if len(dates) > 1 else [len(dates)-1]
			else:
				indices_to_use = [idx_closest-1, idx_closest]
				
			idx_march = np.array(indices_to_use)

		# Collect Fluxes
		# thermal_flux_arr[i] is the array for run i
		thermal_vals = np.asarray(thermal_flux_arr[i])[idx_march]
		all_thermal.extend(thermal_vals)
		
		# Check detectability for these points
		# observable arrays contain NaNs for non-observable points
		obs_thermal_vals = np.asarray(observ_thermal_arr[i])[idx_march]
		detectable_points_thermal += np.sum(~np.isnan(obs_thermal_vals))
		
		if run_reflected:
			reflected_vals = np.asarray(reflected_flux_arr[i])[idx_march]
			all_reflected.extend(reflected_vals)
			
			total_vals = np.asarray(total_flux_arr[i])[idx_march]
			all_total.extend(total_vals)
			
			obs_reflected_vals = np.asarray(observ_reflected_arr[i])[idx_march]
			detectable_points_reflected += np.sum(~np.isnan(obs_reflected_vals))
			
			obs_total_vals = np.asarray(observ_total_arr[i])[idx_march]
			detectable_points_total += np.sum(~np.isnan(obs_total_vals))

		total_points_count += len(idx_march)

	# Statistics

	# Toggle for plotting
	MAKE_MARCH_PLOTS = True 

	def get_stats_basic(data):
		if len(data) == 0: return 0, 0, 0, None
		
		# Robust statistics
		median_val = np.nanmedian(data)
		p16 = np.nanpercentile(data, 16)
		p84 = np.nanpercentile(data, 84)
		
		return median_val, p16, p84, data

	# Calculate stats for all components
	med_th, p16_th, p84_th, data_th = get_stats_basic(all_thermal)
	
	if run_reflected:
		med_ref, p16_ref, p84_ref, data_ref = get_stats_basic(all_reflected)
		med_tot, p16_tot, p84_tot, data_tot = get_stats_basic(all_total)
	else:
		med_ref, p16_ref, p84_ref, data_ref = 0,0,0, []
		med_tot, p16_tot, p84_tot, data_tot = 0,0,0, []

	# Use the global plotfigs variable or the local toggle
	if MAKE_MARCH_PLOTS and (len(all_thermal) > 0):
		try:
			# Setup 3 subplots vertically
			fig, axs = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)
			fig.suptitle(f'{planet_name} - March 2027 Contrast Distribution', fontsize=16)

			# Helper to plot on a specific axis
			def plot_on_axis(ax, data, median_val, p16, p84, label_name, color):
				if len(data) == 0:
					ax.text(0.5, 0.5, "No Data", ha='center', va='center')
					return
				
				# Histogram
				ax.hist(data, bins=30, density=False, alpha=0.6, color=color, label='Data')
				
				# Vertical lines for statistics
				ax.axvline(median_val, color='k', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.2e}')
				ax.axvline(p16, color='k', linestyle=':', linewidth=1, label=f'16th: {p16:.2e}')
				ax.axvline(p84, color='k', linestyle=':', linewidth=1, label=f'84th: {p84:.2e}')
				
				ax.set_title(f'{label_name} Contrast', fontsize=14)
				ax.set_ylabel('Number of values', fontsize=12)
				ax.legend(loc='upper right', fontsize=10)
				ax.grid(alpha=0.3)
				# scientific notation for x axis
				ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

			# Plot Thermal
			plot_on_axis(axs[0], data_th, med_th, p16_th, p84_th, "Thermal", "salmon")
			
			if run_reflected:
				plot_on_axis(axs[1], data_ref, med_ref, p16_ref, p84_ref, "Reflected", "skyblue")
				plot_on_axis(axs[2], data_tot, med_tot, p16_tot, p84_tot, "Total", "lightgreen")
			else:
				axs[1].text(0.5, 0.5, "Reflected Light Not Run", ha='center', va='center')
				axs[2].text(0.5, 0.5, "Total Flux Not Calculated", ha='center', va='center')

			axs[2].set_xlabel('Contrast (Flux Ratio)', fontsize=12)

			# Save
			csv_dir = os.path.dirname(csv_filename)
			if csv_dir == '': csv_dir = '.'
			safe_pname = '%s_nrun%s_%s_%s_%s' % (dictionary[key]['pl_name'], str(nrun), namerun, Madys_Modell_selection, dist_str)
				
			plot_path = os.path.join(csv_dir, f"March2027_Dist_{safe_pname}_Combined.png")
			
			plt.savefig(plot_path, dpi=150)
			plt.close(fig)
			print(f"Saved distribution plot to {plot_path}")
		except Exception as e:
			import traceback
			traceback.print_exc()
			print(f"Error plotting March 2027 distributions: {e}")

		mean_ref, mu_ref, sig_p_ref, sig_m_ref = 0,0,0,0
		mean_tot, mu_tot, sig_p_tot, sig_m_tot = 0,0,0,0
		
	# Detection Percentages
	# "percentage of orbital positions in March 2027 that are detectable"
	pct_det_th = (detectable_points_thermal / total_points_count * 100) if total_points_count > 0 else 0
	pct_det_ref = (detectable_points_reflected / total_points_count * 100) if total_points_count > 0 else 0
	pct_det_tot = (detectable_points_total / total_points_count * 100) if total_points_count > 0 else 0
	
	import csv
	import datetime
	
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	# Write to CSV
	row = [
		planet_name,
		# Thermal
		med_th, p16_th, p84_th, pct_det_th,
		# Reflected
		med_ref, p16_ref, p84_ref, pct_det_ref,
		# Total
		med_tot, p16_tot, p84_tot, pct_det_tot,
		total_points_count,
		split_gaussian, filter_name, entropy_model,
		timestamp
	]
	
	header = [
		"Planet", 
		"Median_Thermal", "P16_Thermal", "P84_Thermal", "Pct_Det_Thermal",
		"Median_Reflected", "P16_Reflected", "P84_Reflected", "Pct_Det_Reflected",
		"Median_Total", "P16_Total", "P84_Total", "Pct_Det_Total",
		"Total_Points_Sampled",
		"Split_Gaussian_Sampling", "Filter", "Thermal_Model", "Calculation_Date"
	]
	
	file_exists = os.path.isfile(csv_filename)
	
	with open(csv_filename, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(header)
		writer.writerow(row)
	
	print(f"March 2027 Analysis saved to {csv_filename} (N={total_points_count} points)")

if __name__ == '__main__':

	time0 = time.time()
	print

	global Mjup
	global Rjup
	global Mear
	global Rear
	global Msun
	global G_constant
	global AU_constant
	global namerun
	global route
	global IWA
	global OWA
	global Cmin
	global instrument
	global g
	global nrun
	global wav
	global minimum_contrast
	global startdetectwindow
	global enddetectwindow





#INPUT AREA

	# ============================================================================
	# FILTER CONFIGURATION SECTION
	# ============================================================================
	# Define all available filters and their parameters
	# To add a new filter, add an entry to this dictionary with all parameters
	
	FILTER_CONFIGS = {
		'METIS_L_BAND': {
			'owa_extension': 4500,  # Target OWA for extending contrast curves (mas)
			'transmission_curve_file': 'TC_filter_HCI_L_long.dat',
			'transmission_skiprows': 13,
			'madys_filter_name': 'METIS_Lp',
			'mass2mag_filter_name': 'METIS_L',
		'stellar_mag_priority': ['sy_w1mag', 'sy_w2mag', 'sy_kmag']
	},
	
	'METIS_M_BAND': {
		'owa_extension': 4500,
		'transmission_curve_file': 'TC_filter_HCI_M.dat',
		'transmission_skiprows': 15,
		'madys_filter_name': 'METIS_Mp',
		'mass2mag_filter_name': 'METIS_M',
		'stellar_mag_priority': ['sy_w2mag', 'sy_w1mag', 'sy_w3mag', 'sy_kmag']
	},


	'METIS_N_BAND': {
		'owa_extension': 3000,
		'transmission_curve_file': 'TC_filter_N2.dat',
		'transmission_skiprows': 13,
		'madys_filter_name': 'METIS_N2',
		'mass2mag_filter_name': 'METIS_N',
		'stellar_mag_priority': ['sy_w3mag', 'sy_w2mag', 'sy_kmag']
	},
	
	# Roman Space Telescope Filters - TO BE FILLED IN
	'ROMAN_F1': {
		'owa_extension': 440,
		'integration_time' : 'short', #'short' = 25hr, 'medium' = 100, 'long' = 10000 (infinite)
		'contrast_assumption': 'opti', #'opti', 'cons'
		'transmission_curve_file': 'transmission_ID-01_1F_v0.csv',  # TO BE FILLED
		'transmission_skiprows': 4,
		'madys_filter_name': 'Roman_CGI_1F',  # May not be applicable, hst / hr H_F606W or gaia: Gbp 
		'mass2mag_filter_name': None,  # TO BE FILLED
		'stellar_mag_priority': ['sy_vmag']  # TO BE REFINED
	},
	
	'ROMAN_F2': {
		'owa_extension': 440,
		'integration_time' : 'short', #'short' = 25hr, 'medium' = 100, 'long' = 10000 (infinite)
		'contrast_assumption': 'opti', #'opti', 'cons'
		'transmission_curve_file': 'transmission_ID-02_2F_v0.csv',
		'transmission_skiprows': 4,
		'madys_filter_name': 'Roman_CGI_2F',
		'mass2mag_filter_name': 'METIS_N',
		'stellar_mag_priority': ['sy_vmag', 'sy_kmag']
	},
	
	'ROMAN_F3': {
		'owa_extension': 558,
		'integration_time' : 'short', #'short' = 25hr, 'medium' = 100, 'long' = 10000 (infinite)
		'contrast_assumption': 'opti', #'opti', 'cons'
		'transmission_curve_file': 'transmission_ID-03_3F_v0.csv',
		'transmission_skiprows': 4,
		'madys_filter_name': 'Roman_CGI_3F',
		'mass2mag_filter_name': None,
		'stellar_mag_priority': ['sy_vmag']
	},
	
	'ROMAN_F4': {
		'owa_extension': 1404,
		'integration_time' : 'short', #'short' = 25hr, 'medium' = 100, 'long' = 10000 (infinite)
		'contrast_assumption': 'opti', #'opti', 'cons'
		'transmission_curve_file': 'transmission_ID-04_4F_v0.csv',
		'transmission_skiprows': 4,
		'madys_filter_name': 'Roman_CGI_4F',
		'mass2mag_filter_name': 'METIS_N',
		'stellar_mag_priority': ['sy_vmag', 'sy_kmag']
	}
}
	
	#============================================================================
	# MANUAL INPUT OVERRIDES FOR PLANET/STAR PARAMETERS
	# ============================================================================
	# Manually set planet/star parameters to override NASA Archive values
	# Leave as None to use values from NASA Archive
	# For uncertainties: set _UPPER for +error and _LOWER for -error (e.g., 151.1 +5.3 -1.8)
	# These will be applied to the currently selected planet
	
	MANUAL_ST_AGE = None  # Stellar age (Gyr)
	MANUAL_ST_AGE_UPPER = None  # Stellar age upper uncertainty (+)
	MANUAL_ST_AGE_LOWER = None  # Stellar age lower uncertainty (-)
	
	MANUAL_ST_MASS = None  # Stellar mass (solar masses)
	MANUAL_ST_MASS_UPPER = None  # Stellar mass upper uncertainty (+)
	MANUAL_ST_MASS_LOWER = None  # Stellar mass lower uncertainty (-)

	MANUAL_ST_TEFF = None  # Stellar temperature (K)
	MANUAL_ST_TEFF_UPPER = None  # Stellar temperature upper uncertainty (+)
	MANUAL_ST_TEFF_LOWER = None  # Stellar temperature lower uncertainty (-)
	
	MANUAL_ST_RAD = None  # Stellar radius (solar radii)
	MANUAL_ST_RAD_UPPER = None  # Stellar radius upper uncertainty (+)
	MANUAL_ST_RAD_LOWER = None  # Stellar radius lower uncertainty (-)
	
	MANUAL_ST_SPECTYPE = None  # Stellar spectral type (string)
	
	MANUAL_SY_DIST = None  # System distance (pc)
	MANUAL_SY_DIST_UPPER = None  # System distance upper uncertainty (+)
	MANUAL_SY_DIST_LOWER = None  # System distance lower uncertainty (-)
	
	MANUAL_SY_VMAG = None  # V magnitude
	MANUAL_SY_VMAG_UPPER = None  # V magnitude upper uncertainty (+)
	MANUAL_SY_VMAG_LOWER = None  # V magnitude lower uncertainty (-)
	
	MANUAL_SY_KMAG = None  # K magnitude
	MANUAL_SY_KMAG_UPPER = None  # K magnitude upper uncertainty (+)
	MANUAL_SY_KMAG_LOWER = None  # K magnitude lower uncertainty (-)
	
	MANUAL_SY_W1MAG = None  # W1 magnitude (WISE)
	MANUAL_SY_W1MAG_UPPER = None  # W1 magnitude upper uncertainty (+)
	MANUAL_SY_W1MAG_LOWER = None  # W1 magnitude lower uncertainty (-)
	
	MANUAL_SY_W2MAG = None  # W2 magnitude (WISE)
	MANUAL_SY_W2MAG_UPPER = None  # W2 magnitude upper uncertainty (+)
	MANUAL_SY_W2MAG_LOWER = None  # W2 magnitude lower uncertainty (-)
	
	MANUAL_SY_W3MAG = None  # W3 magnitude (WISE)
	MANUAL_SY_W3MAG_UPPER = None  # W3 magnitude upper uncertainty (+)
	MANUAL_SY_W3MAG_LOWER = None  # W3 magnitude lower uncertainty (-)
	
	MANUAL_PL_BMASSJ = None  # Planet mass (Jupiter masses)
	MANUAL_PL_BMASSJ_UPPER = None  # Planet mass upper uncertainty (+)
	MANUAL_PL_BMASSJ_LOWER = None  # Planet mass lower uncertainty (-)
	
	MANUAL_PL_RADJ = None  # Planet radius (Jupiter radii)
	MANUAL_PL_RADJ_UPPER = None  # Planet radius upper uncertainty (+)
	MANUAL_PL_RADJ_LOWER = None  # Planet radius lower uncertainty (-)
	
	MANUAL_PL_ORBSMAX = None  # Semi-major axis (AU)
	MANUAL_PL_ORBSMAX_UPPER = None  # Semi-major axis upper uncertainty (+)
	MANUAL_PL_ORBSMAX_LOWER = None  # Semi-major axis lower uncertainty (-)
	
	MANUAL_PL_ORBPER = None  # Orbital period (days)
	MANUAL_PL_ORBPER_UPPER = None  # Orbital period upper uncertainty (+)
	MANUAL_PL_ORBPER_LOWER = None  # Orbital period lower uncertainty (-)
	
	MANUAL_PL_ORBECCEN = None  # Orbital eccentricity
	MANUAL_PL_ORBECCEN_UPPER = None  # Orbital eccentricity upper uncertainty (+)
	MANUAL_PL_ORBECCEN_LOWER = None  # Orbital eccentricity lower uncertainty (-)
	
	MANUAL_PL_ORBINCL = None  # Orbital inclination (degrees)
	MANUAL_PL_ORBINCL_UPPER = None  # Orbital inclination upper uncertainty (+)
	MANUAL_PL_ORBINCL_LOWER = None  # Orbital inclination lower uncertainty (-)
	
	MANUAL_PL_ORBLPER = None  # Argument of periastron (degrees)
	MANUAL_PL_ORBLPER_UPPER = None  # Argument of periastron upper uncertainty (+)
	MANUAL_PL_ORBLPER_LOWER = None  # Argument of periastron lower uncertainty (-)
	
	MANUAL_PL_EQT = None  # Equilibrium temperature (K)
	MANUAL_PL_EQT_UPPER = None  # Equilibrium temperature upper uncertainty (+)
	MANUAL_PL_EQT_LOWER = None  # Equilibrium temperature lower uncertainty (-)


	#Manual parameter inputs for 51 Eri b parameters from Balmer 2025
	"""
	MANUAL_ST_MASS = 1.55  # Stellar mass (solar masses)
	MANUAL_ST_MASS_UPPER = 0.01  # Stellar mass upper uncertainty (+)
	MANUAL_ST_MASS_LOWER = -0.01  # Stellar mass lower uncertainty (-)

	MANUAL_PL_BMASSJ = 0.88  # Planet mass (Jupiter masses)
	MANUAL_PL_BMASSJ_UPPER = 2.71  # Planet mass upper uncertainty (+)
	MANUAL_PL_BMASSJ_LOWER = -0.67  # Planet mass lower uncertainty (-)

	MANUAL_PL_ORBSMAX = 9.58  # Semi-major axis (AU)
	MANUAL_PL_ORBSMAX_UPPER = 1.61  # Semi-major axis upper uncertainty (+)
	MANUAL_PL_ORBSMAX_LOWER = -0.42  # Semi-major axis lower uncertainty (-)

	MANUAL_PL_ORBECCEN = 0.57  # Orbital eccentricity
	MANUAL_PL_ORBECCEN_UPPER = 0.03  # Orbital eccentricity upper uncertainty (+)
	MANUAL_PL_ORBECCEN_LOWER = -0.09  # Orbital eccentricity lower uncertainty (-)
	
	MANUAL_PL_ORBINCL = 151.1  # Orbital inclination (degrees)
	MANUAL_PL_ORBINCL_UPPER = 5.3  # Orbital inclination upper uncertainty (+)
	MANUAL_PL_ORBINCL_LOWER = -11.8  # Orbital inclination lower uncertainty (-)
	"""

	#Manual parameter inputs for AB Pic c parameters from chat with pauline for hypothetical planet
	"""
	MANUAL_PL_BMASSJ = 6  # Planet mass (Jupiter masses)
	MANUAL_PL_BMASSJ_UPPER = 4  # Planet mass upper uncertainty (+)
	MANUAL_PL_BMASSJ_LOWER = -4  # Planet mass lower uncertainty (-)

	MANUAL_PL_ORBSMAX = 7  # Semi-major axis (AU)
	MANUAL_PL_ORBSMAX_UPPER = 5  # Semi-major axis upper uncertainty (+)
	MANUAL_PL_ORBSMAX_LOWER = -5  # Semi-major axis lower uncertainty (-)

	MANUAL_ST_AGE = 0.0133  # Stellar age (Gyr)
	MANUAL_ST_AGE_UPPER = 0.0011  # Stellar age upper uncertainty (+)
	MANUAL_ST_AGE_LOWER = -0.0006  # Stellar age lower uncertainty (-)
	
	"""


	# ============================================================================
	# MAIN CONFIGURATION
	# ============================================================================

	#namerun = 'PCS'	#Choose your run name
	#minimum_contrast = minimum_contrast_PCS
	#namerun = 'Roman-pessimistic'
	#minimum_contrast = minimum_contrast_Roman_pessim
	#namerun = 'Roman-optimistic'
	#minimum_contrast = minimum_contrast_Roman_optim
	#namerun = 'Roman-WFoV'
	#minimum_contrast = minimum_contrast_Roman_WFoV
	#namerun = 'METIS'
	minimum_contrast = minimum_contrast_METIS
	

	#OLD FILES FROM NASA EXOPLANET ARCHIVE
	#routeNASAarchive = '/Users/seschwaiger/Desktop/Sebastian Locatis/PS_2025.02.05_11.54.53.tab'
	#routeNASAarchiveCOMPOS = '/Users/seschwaiger/Desktop/Sebastian Locatis/PSCompPars_2025.02.05_11.54.18.tab'


	route = '/Users/seschwaiger/Desktop/Sebastian Locatis/Working' #Choose the directory to plot your results
	routeNASAarchive = '/Users/seschwaiger/Desktop/Sebastian Locatis/PS_2026.02.23_10.45.26.tab'
	routeNASAarchiveCOMPOS = '/Users/seschwaiger/Desktop/Sebastian Locatis/PSCompPars_2026.02.23_10.47.38.tab'
	nrun = 1000		#Number of orbital simulations for each planet in the bootstrap-like statistical methodology
	plotfigs = True		#Do we plot the Fp/Fstar vs. angular separation plots for each of the simulated planets ?
	WRITE_SUMMARY_CSV = True	#Write a CSV log with detection probability, mean contrast, timestamp for each planet
	WRITE_INPUT_TABLE_CSV = True      # Write appendix CSV: NASA archive input parameters for every simulated planet
	WRITE_OUTPUT_TABLE_CSV = True     # Write main-text CSV: LOCATIS results for planets above det_prob threshold
	COMBINE_OUTPUT_TABLES_PATH = None #'/Users/seschwaiger/Desktop/Sebastian Locatis/Working/LOCATIS_output_table_METIS_L_BAND_bex-atmo2023-ceq_2026-03-26.csv' # Set to path of the *other* band's output table to produce a merged L+M CSV
	                                  # e.g. '/path/to/LOCATIS_output_table_METIS_M_BAND_..._<date>.csv'

	# ============================================================================
	# FILTER SELECTION
	# ============================================================================

	#METIS_L_BAND, METIS_M_BAND, METIS_N_BAND filters are available in 
	#Available in models: bex-atmo2023-neq-s, bex-atmo2023-neq-w, bex-atmo2023-ceq & atmo2023-ceq, atmo2023-neq-s, atmo2023-neq-w

	#ROMAN_F1, ROMAN_F2, ROMAN_F3, ROMAN_F4 filters are available in 
	#'sonora-flame-skimmer-ceq' for chemical equilibrium and 'sonora-flame-skimmer-neq' for chemical disequilibrium. 

	#The filters are called: 'Roman_CGI_1A', 'Roman_CGI_1B', 'Roman_CGI_1C', 'Roman_CGI_1F','Roman_CGI_2A', 'Roman_CGI_2B',
	# 'Roman_CGI_2C', 'Roman_CGI_2F','Roman_CGI_3A', 'Roman_CGI_3B', 'Roman_CGI_3C', 'Roman_CGI_3D','Roman_CGI_3E',
	# 'Roman_CGI_3F', 'Roman_CGI_3G', 'Roman_CGI_4A','Roman_CGI_4B', 'Roman_CGI_4C', 'Roman_CGI_4F', 'Roman_WFI_F062',
	# 'Roman_WFI_F087', 'Roman_WFI_F106', 'Roman_WFI_F129','Roman_WFI_F146', 'Roman_WFI_F158', 'Roman_WFI_F184','Roman_WFI_F213' 
	# (you can use the function info_filters() to have info about the wavelength, the extinction coefficient, etc)



	
	Madys_Modell_selection = 'sonora-flame-skimmer-ceq'
	# ============================================================================
	# FILTER SELECTION
	# ============================================================================
	# Choose which filter to use from FILTER_CONFIGS dictionary
	# Available filters: METIS_L_BAND, METIS_M_BAND, METIS_N_BAND, 
	#                    ROMAN_F1, ROMAN_F2, ROMAN_F3, ROMAN_F4

	FILTER_USED = 'ROMAN_F1'  # Set to the desired filter key from FILTER_CONFIGS

	# ============================================================================
	# LOAD SELECTED FILTER CONFIGURATION
	# ============================================================================
	
	# Get the configuration for the selected filter
	if FILTER_USED not in FILTER_CONFIGS:
		raise ValueError(f"Filter '{FILTER_USED}' not found in FILTER_CONFIGS. Available filters: {list(FILTER_CONFIGS.keys())}")
	
	config = FILTER_CONFIGS[FILTER_USED].copy()
	
	# Extract parameters from config (will be used throughout the script)
	owa_extension = config['owa_extension']
	transmission_curve_file = config['transmission_curve_file']
	transmission_skiprows = config['transmission_skiprows']
	madys_filter_name = config['madys_filter_name']
	mass2mag_filter_name = config['mass2mag_filter_name']
	stellar_mag_priority = config['stellar_mag_priority']
	
	print(f"\n{'='*60}")
	print(f"FILTER CONFIGURATION LOADED: {FILTER_USED}")
	print(f"{'='*60}")
	print(f"Transmission Curve: {transmission_curve_file}")
	print(f"MADYS Filter: {madys_filter_name}")
	print(f"mass2mag Filter: {mass2mag_filter_name}")
	print(f"OWA Extension: {owa_extension} mas")
	print(f"Stellar Mag Priority: {stellar_mag_priority}")
	print(f"{'='*60}\n")

	namerun = FILTER_USED	#Choose your run name

	#Choose whether you want the Reflected light curves as well or not
	RUN_REFLECTED_LIGHT = True 


	#Decide if we use the stellar spectra
	USE_STELLAR_SPECTRUM_FILE = True


	# Choose whether to use mass2mag for thermal flux calculation
	use_mass2mag = False # Set to False with madys to use the old thermal_flux_planet function
	
	# Choose whether to use Madys for thermal flux calculation
	use_madys = True # Set to False with mass2mag to use the old thermal_flux_planet function
	
	#Choose blackbodycalculations
	use_blackbody = False

	#if set to True we have a split normal distribution used for all inputed parameters, if set to False we use bootstrapping (uniform distribution inbetween uncertainties)
	SPLIT_GAUSSIAN = True




	# Choose whether to plot the Full long distribution histogram from Oscars LOCATIS_individual_column_plot.py 
	PLOT_NEWCOLUMN_SUBPLOTS = False



	PLOT_COMBINED_SUMMARY = True	#Do we plot the combined summary plot (Flux + Orbit) for each of the simulated planets?


	# --- PLOT SWITCHES (set False to save RAM) ---
	# Contrast vs. angular sep + phase-angle histogram (reflected light only)
	PLOT_CONTRAST_ALPHAS_SUBPLOTS = True
	# Detectability windows vs. time (requires pl_orbtper in catalog)
	PLOT_DETECTABILITY_WINDOWS = True
	# 2D orbit map (ΔRA vs ΔDec) with IWA/OWA shading (thermal)
	PLOT_2D_ANGSEP_MAP = True
	# Thermal flux ratio vs. angular separation
	PLOT_THERMAL_FLUX_VS_ANGSEP = True
	# Reflected + thermal combined flux vs. angular separation
	PLOT_REFLECTED_PLUS_THERMAL = True

	#ModelHandler.available('atmo2023') # Check if the model is available before running simulations



	# Choose whether to plot the mass distribution histogram
	PLOT_MASS_HISTOGRAM = False
	# Choose whether to plot the eccentricity distribution histogram
	PLOT_ECC_HISTOGRAM = False
	# Choose whether to plot the inclination distribution histogram
	PLOT_INCL_HISTOGRAM = False
	# Choose whether to plot the temperature distribution histogram
	PLOT_TEMP_HISTOGRAM = False
	# Choose whether to plot the absolute magnitude distribution histogram
	PLOT_ABS_MAG_HISTOGRAM = False
	# Choose whether to plot the age distribution histogram
	PLOT_AGE_HISTOGRAM = False

	# -------------------------------------------------------------------------
	# SUMMARY CONTRAST vs. SEPARATION PLOT
	# Plots all planets from the CSV that have det_prob_total_% > 25%,
	# colour-coded by age, with error bars on contrast and angular separation.
	# Set SUMMARY_CSV_RERUN_PATH to a path string to skip simulations and only
	# regenerate this plot from a previously produced CSV file.
	# -------------------------------------------------------------------------
	PLOT_SUMMARY_CONTRAST_SEP = False
	PLOT_SUMMARY_FLAG_CONTRAST = False       # flag-coloured planet contrast bar chart (horizontal: names on y, contrast on x)
	PLOT_SUMMARY_DETPROB_NAMES = False       # flag-coloured detection probability bar chart (horizontal: names on y, det_prob on x)
	PLOT_SUMMARY_DIST_VS_SMA = False         # distance [pc] vs semi-major axis [AU] scatter plot
	PLOT_SUMMARY_ABSMAG_VS_AGE = False       # absolute magnitude vs planet age scatter plot
	PLOT_SUMMARY_TEMP_VS_AGE = False         # effective temperature vs age scatter plot
	PLOT_SUMMARY_TEMP_VS_MASS = False        # effective temperature vs mass scatter plot
	PLOT_SUMMARY_A_VS_D = False              # semi-major axis [AU] vs. distance [pc], all archive (hollow) + detectable (filled), coloured by discovery method
	PLOT_STELLAR_PROPERTIES_HISTOGRAM = False # spectral type / age / stellar mass histograms (all vs detectable)
	SUMMARY_CONTRAST_DET_THRESHOLD = 25.0   # % threshold for det_prob_total_%
	SUMMARY_CSV_RERUN_PATH = None#'/Users/seschwaiger/Desktop/Communication_Master/Tables with METIS  Longrun/LOCATIS_run_summary_METIS_L_BAND_bex-atmo2023-ceq_2026-03-06.csv'#'/Users/seschwaiger/Desktop/Communication_Master/Tables with METIS  Longrun/LOCATIS_run_summary_L_Band_1.3_Mag_12.csv'            # e.g. '/path/to/LOCATIS_run_summary_METIS_L_BAND_bex-atmo2023-ceq.csv'




	Mear = 5.972E24				#Earth mass [kg]
	Rear = 6371000.				#Earth radius [m]
	Mjup = 1.898E27				#kg
	Rjup = 69911000.			#m
	Msun = 1.989E30				#kg
	Rsun = 696340000.			#m
	Lsun = 3.828E26				#W
	G_constant = 6.67408E-11		#m**3 kg**-1 s**-2
	AU_constant = 149597870700.		#m

	startdetectwindow = 2461284.5 #2026.09.01
	enddetectwindow = 2464937.5 #2036.09.01



	#######Sebastian#####

	c = 299792458 #[m/s] speed of light
	k = 1.380649E-23   #[J/K] boltzmann constant
	h = 6.62607015E-34 #[J/s] Planck constant



	print("nrun = ", nrun)

	#Ag had to be moved into orbital simulations
	# Better definition based on standard usage [wav, albedo]
	Ag_spectrum = np.array([
		[1, 0.3],   # At 1 microns, Albedo is 0.3 ? (Example values) 
		[10, 0.3],
		[100.0, 0.3]  # At 100 microns, Albedo is 0.1
	])	
	#Ag = 0.3
	Abond = 0.45


	if plotfigs == True:
		print("Plotting figures of the orbits: ON\n")
	elif plotfigs == False:
		print("Plotting figures of the orbits: OFF\n")
	else:
		print("Error in definition of plotfigs. Stopping run...\n")
		quit()


	###################################################
	#One can quickly compute the orbital evolution of a given planNone#'/Users/seschwaiger/Desktop/Communication_Master/Tables with METIS  Longrun/LOCATIS_run_summary_L_Band_1.3_Mag_12.csv'nd a given orbital configuration just calling the function orbit_evolution
	#Xcoord, Ycoord, t_tp, trueanom, angproj, Fp_Fstar, alpha, pl_star_distAU = orbit_evolution(aorbit, incl, ecc, longperiast, Rp, Ag, dist)
	#aorbit: semimajor axis [AU]
	#incl: orbital inclination [deg]
	#ecc: orbital eccentricity
	#longperiast: argument of periastron [deg]
	#Rp: planet radius [jupiter radii]
	#Ag: geometric albedo
	#dist: distance to the planetary system [pc]

	#Xcoord, Ycoord, t_tp, trueanom, angproj, Fp_Fstar, alpha, pl_star_distAU = orbit_evolution(4., 50., 0.3, 60., 1., 0.3, 3.)

	#Example for a given orbital realization of eps Eri b
	#Xcoord, Ycoord, t_tp, trueanom, angproj, Fp_Fstar, alpha, pl_star_distAU = orbit_evolution(3.39, 30.1, 0.7, 47., 1.175, Ag, 3.21)	#eps Eri b

	#quit()
	###################################################

	dictionary = read_table_exoplanets_NASAArchive(routeNASAarchive)
	dictionary_NASA_Compos = read_table_exoplanets_NASAArchive(routeNASAarchiveCOMPOS)
	print("Confirmed exoplanets: ", len(dictionary))

	#Important to correct the reported argument of periastron (see Appendix B of Carrion-Gonzalez et al (2021), A&A 651, A7)
	for namei in dictionary:
		if dictionary[namei]['pl_orblper'] != '':	#Here I assume that the reported pl_orblper is actually that of the star and, by default, I add the 180-degree shift
			if 180.+float(dictionary[namei]['pl_orblper']) < 360.:
				dictionary[namei]['pl_orblper'] = str(180.+float(dictionary[namei]['pl_orblper']))
			else:
				dictionary[namei]['pl_orblper'] = str(180.+float(dictionary[namei]['pl_orblper'])-360.)


	#Getting missing information from the Composite NASA Archive
	for key in dictionary:
		if dictionary[key]['st_spectype']=='' and dictionary_NASA_Compos[key]['st_spectype']!='':
			dictionary[key]['st_spectype']=dictionary_NASA_Compos[key]['st_spectype']	
		if dictionary[key]['sy_dist']=='' and dictionary_NASA_Compos[key]['sy_dist']!='':
			dictionary[key]['sy_dist']=dictionary_NASA_Compos[key]['sy_dist']			
			dictionary[key]['sy_disterr1']=dictionary_NASA_Compos[key]['sy_disterr1']
			dictionary[key]['sy_disterr2']=dictionary_NASA_Compos[key]['sy_disterr2']
		if dictionary[key]['st_rad']=='' and dictionary_NASA_Compos[key]['st_rad']!='':
			dictionary[key]['st_rad']=dictionary_NASA_Compos[key]['st_rad']
			dictionary[key]['st_raderr1']=dictionary_NASA_Compos[key]['st_raderr1']
			dictionary[key]['st_raderr2']=dictionary_NASA_Compos[key]['st_raderr2']
		if dictionary[key]['st_teff']=='' and dictionary_NASA_Compos[key]['st_teff']!='':
			dictionary[key]['st_teff']=dictionary_NASA_Compos[key]['st_teff']
			dictionary[key]['st_tefferr1']=dictionary_NASA_Compos[key]['st_tefferr1']
			dictionary[key]['st_tefferr2']=dictionary_NASA_Compos[key]['st_tefferr2']
		if dictionary[key]['st_mass']=='' and dictionary_NASA_Compos[key]['st_mass']!='':
			dictionary[key]['st_mass']=dictionary_NASA_Compos[key]['st_mass']
			dictionary[key]['st_masserr1']=dictionary_NASA_Compos[key]['st_masserr1']
			dictionary[key]['st_masserr2']=dictionary_NASA_Compos[key]['st_masserr2']
		if dictionary[key]['pl_bmassj']=='' and dictionary_NASA_Compos[key]['pl_bmassj']!='':
			dictionary[key]['pl_bmassj']=dictionary_NASA_Compos[key]['pl_bmassj']
			dictionary[key]['pl_bmassjerr1']=dictionary_NASA_Compos[key]['pl_bmassjerr1']
			dictionary[key]['pl_bmassjerr2']=dictionary_NASA_Compos[key]['pl_bmassjerr2']
		if dictionary[key]['sy_vmag']=='' and dictionary_NASA_Compos[key]['sy_vmag']!='':
			dictionary[key]['sy_vmag']=dictionary_NASA_Compos[key]['sy_vmag']
			dictionary[key]['sy_vmagerr1']=dictionary_NASA_Compos[key]['sy_vmagerr1']
			dictionary[key]['sy_vmagerr2']=dictionary_NASA_Compos[key]['sy_vmagerr2']
		if dictionary[key]['st_age']=='' and dictionary_NASA_Compos[key]['st_age']!='':
			dictionary[key]['st_age']=dictionary_NASA_Compos[key]['st_age']
			dictionary[key]['st_ageerr1']=dictionary_NASA_Compos[key]['st_ageerr1']
			dictionary[key]['st_ageerr2']=dictionary_NASA_Compos[key]['st_ageerr2']



		#ADDED SEBASTIAN for MATTHIEU PAPER> TIME OF PERIASTRON FOR CONTRAST +DETERMINATION IN MARCH 2027
		if dictionary[key]['pl_orbtper']=='' and dictionary_NASA_Compos[key]['pl_orbtper']!='':
			dictionary[key]['pl_orbtper']=dictionary_NASA_Compos[key]['pl_orbtper']
			dictionary[key]['pl_orbtpererr1']=dictionary_NASA_Compos[key]['pl_orbtpererr1']
			dictionary[key]['pl_orbtpererr2']=dictionary_NASA_Compos[key]['pl_orbtpererr2']


		# --- ADDED: Merge Infrared Magnitudes for METIS ---
		# Check if keys exist first to avoid errors if columns weren't downloaded
		if 'sy_kmag' in dictionary[key] and 'sy_kmag' in dictionary_NASA_Compos[key]:
			if dictionary[key]['sy_kmag']=='' and dictionary_NASA_Compos[key]['sy_kmag']!='':
				dictionary[key]['sy_kmag']=dictionary_NASA_Compos[key]['sy_kmag']

		if 'sy_w1mag' in dictionary[key] and 'sy_w1mag' in dictionary_NASA_Compos[key]:
			if dictionary[key]['sy_w1mag']=='' and dictionary_NASA_Compos[key]['sy_w1mag']!='':
				dictionary[key]['sy_w1mag']=dictionary_NASA_Compos[key]['sy_w1mag']

		if 'sy_w2mag' in dictionary[key] and 'sy_w2mag' in dictionary_NASA_Compos[key]:
			if dictionary[key]['sy_w2mag']=='' and dictionary_NASA_Compos[key]['sy_w2mag']!='':
				dictionary[key]['sy_w2mag']=dictionary_NASA_Compos[key]['sy_w2mag']

		if 'sy_w3mag' in dictionary[key] and 'sy_w3mag' in dictionary_NASA_Compos[key]:
			if dictionary[key]['sy_w3mag']=='' and dictionary_NASA_Compos[key]['sy_w3mag']!='':
				dictionary[key]['sy_w3mag']=dictionary_NASA_Compos[key]['sy_w3mag']
        # --------------------------------------------------


	# Apply manual parameter overrides if specified
	def apply_manual_overrides(planet_dict, key):
		"""Apply manual parameter overrides to the planet dictionary"""
		# Single value overrides
		if MANUAL_ST_AGE is not None:
			planet_dict[key]['st_age'] = str(MANUAL_ST_AGE)
		if MANUAL_ST_MASS is not None:
			planet_dict[key]['st_mass'] = str(MANUAL_ST_MASS)
		if MANUAL_ST_TEFF is not None:
			planet_dict[key]['st_teff'] = str(MANUAL_ST_TEFF)
		if MANUAL_ST_RAD is not None:
			planet_dict[key]['st_rad'] = str(MANUAL_ST_RAD)
		if MANUAL_ST_SPECTYPE is not None:
			planet_dict[key]['st_spectype'] = str(MANUAL_ST_SPECTYPE)
		if MANUAL_SY_DIST is not None:
			planet_dict[key]['sy_dist'] = str(MANUAL_SY_DIST)
		if MANUAL_SY_VMAG is not None:
			planet_dict[key]['sy_vmag'] = str(MANUAL_SY_VMAG)
		if MANUAL_SY_KMAG is not None:
			planet_dict[key]['sy_kmag'] = str(MANUAL_SY_KMAG)
		if MANUAL_SY_W1MAG is not None:
			planet_dict[key]['sy_w1mag'] = str(MANUAL_SY_W1MAG)
		if MANUAL_SY_W2MAG is not None:
			planet_dict[key]['sy_w2mag'] = str(MANUAL_SY_W2MAG)
		if MANUAL_SY_W3MAG is not None:
			planet_dict[key]['sy_w3mag'] = str(MANUAL_SY_W3MAG)
		if MANUAL_PL_BMASSJ is not None:
			planet_dict[key]['pl_bmassj'] = str(MANUAL_PL_BMASSJ)
		if MANUAL_PL_RADJ is not None:
			planet_dict[key]['pl_radj'] = str(MANUAL_PL_RADJ)
		if MANUAL_PL_ORBSMAX is not None:
			planet_dict[key]['pl_orbsmax'] = str(MANUAL_PL_ORBSMAX)
		if MANUAL_PL_ORBPER is not None:
			planet_dict[key]['pl_orbper'] = str(MANUAL_PL_ORBPER)
		if MANUAL_PL_ORBECCEN is not None:
			planet_dict[key]['pl_orbeccen'] = str(MANUAL_PL_ORBECCEN)
		if MANUAL_PL_ORBINCL is not None:
			planet_dict[key]['pl_orbincl'] = str(MANUAL_PL_ORBINCL)
		if MANUAL_PL_ORBLPER is not None:
			planet_dict[key]['pl_orblper'] = str(MANUAL_PL_ORBLPER)
		if MANUAL_PL_EQT is not None:
			planet_dict[key]['pl_eqt'] = str(MANUAL_PL_EQT)
		
		# Error bar overrides (upper and lower uncertainties)
		if MANUAL_ST_AGE_UPPER is not None:
			planet_dict[key]['st_ageerr1'] = str(MANUAL_ST_AGE_UPPER)
		if MANUAL_ST_AGE_LOWER is not None:
			planet_dict[key]['st_ageerr2'] = str(MANUAL_ST_AGE_LOWER)
		
		if MANUAL_ST_MASS_UPPER is not None:
			planet_dict[key]['st_masserr1'] = str(MANUAL_ST_MASS_UPPER)
		if MANUAL_ST_MASS_LOWER is not None:
			planet_dict[key]['st_masserr2'] = str(MANUAL_ST_MASS_LOWER)
		
		if MANUAL_ST_TEFF_UPPER is not None:
			planet_dict[key]['st_tefferr1'] = str(MANUAL_ST_TEFF_UPPER)
		if MANUAL_ST_TEFF_LOWER is not None:
			planet_dict[key]['st_tefferr2'] = str(MANUAL_ST_TEFF_LOWER)
		
		if MANUAL_ST_RAD_UPPER is not None:
			planet_dict[key]['st_raderr1'] = str(MANUAL_ST_RAD_UPPER)
		if MANUAL_ST_RAD_LOWER is not None:
			planet_dict[key]['st_raderr2'] = str(MANUAL_ST_RAD_LOWER)
		
		if MANUAL_SY_DIST_UPPER is not None:
			planet_dict[key]['sy_disterr1'] = str(MANUAL_SY_DIST_UPPER)
		if MANUAL_SY_DIST_LOWER is not None:
			planet_dict[key]['sy_disterr2'] = str(MANUAL_SY_DIST_LOWER)
		
		if MANUAL_SY_VMAG_UPPER is not None:
			planet_dict[key]['sy_vmagerr1'] = str(MANUAL_SY_VMAG_UPPER)
		if MANUAL_SY_VMAG_LOWER is not None:
			planet_dict[key]['sy_vmagerr2'] = str(MANUAL_SY_VMAG_LOWER)
		
		if MANUAL_SY_KMAG_UPPER is not None:
			planet_dict[key]['sy_kmagerr1'] = str(MANUAL_SY_KMAG_UPPER)
		if MANUAL_SY_KMAG_LOWER is not None:
			planet_dict[key]['sy_kmagerr2'] = str(MANUAL_SY_KMAG_LOWER)
		
		if MANUAL_SY_W1MAG_UPPER is not None:
			planet_dict[key]['sy_w1magerr1'] = str(MANUAL_SY_W1MAG_UPPER)
		if MANUAL_SY_W1MAG_LOWER is not None:
			planet_dict[key]['sy_w1magerr2'] = str(MANUAL_SY_W1MAG_LOWER)
		
		if MANUAL_SY_W2MAG_UPPER is not None:
			planet_dict[key]['sy_w2magerr1'] = str(MANUAL_SY_W2MAG_UPPER)
		if MANUAL_SY_W2MAG_LOWER is not None:
			planet_dict[key]['sy_w2magerr2'] = str(MANUAL_SY_W2MAG_LOWER)
		
		if MANUAL_SY_W3MAG_UPPER is not None:
			planet_dict[key]['sy_w3magerr1'] = str(MANUAL_SY_W3MAG_UPPER)
		if MANUAL_SY_W3MAG_LOWER is not None:
			planet_dict[key]['sy_w3magerr2'] = str(MANUAL_SY_W3MAG_LOWER)
		
		if MANUAL_PL_BMASSJ_UPPER is not None:
			planet_dict[key]['pl_bmassjerr1'] = str(MANUAL_PL_BMASSJ_UPPER)
		if MANUAL_PL_BMASSJ_LOWER is not None:
			planet_dict[key]['pl_bmassjerr2'] = str(MANUAL_PL_BMASSJ_LOWER)
		
		if MANUAL_PL_RADJ_UPPER is not None:
			planet_dict[key]['pl_radjerr1'] = str(MANUAL_PL_RADJ_UPPER)
		if MANUAL_PL_RADJ_LOWER is not None:
			planet_dict[key]['pl_radjerr2'] = str(MANUAL_PL_RADJ_LOWER)
		
		if MANUAL_PL_ORBSMAX_UPPER is not None:
			planet_dict[key]['pl_orbsmaxerr1'] = str(MANUAL_PL_ORBSMAX_UPPER)
		if MANUAL_PL_ORBSMAX_LOWER is not None:
			planet_dict[key]['pl_orbsmaxerr2'] = str(MANUAL_PL_ORBSMAX_LOWER)
		
		if MANUAL_PL_ORBPER_UPPER is not None:
			planet_dict[key]['pl_orbpererr1'] = str(MANUAL_PL_ORBPER_UPPER)
		if MANUAL_PL_ORBPER_LOWER is not None:
			planet_dict[key]['pl_orbpererr2'] = str(MANUAL_PL_ORBPER_LOWER)
		
		if MANUAL_PL_ORBECCEN_UPPER is not None:
			planet_dict[key]['pl_orbeccenerr1'] = str(MANUAL_PL_ORBECCEN_UPPER)
		if MANUAL_PL_ORBECCEN_LOWER is not None:
			planet_dict[key]['pl_orbeccenerr2'] = str(MANUAL_PL_ORBECCEN_LOWER)
		
		if MANUAL_PL_ORBINCL_UPPER is not None:
			planet_dict[key]['pl_orbinclerr1'] = str(MANUAL_PL_ORBINCL_UPPER)
		if MANUAL_PL_ORBINCL_LOWER is not None:
			planet_dict[key]['pl_orbinclerr2'] = str(MANUAL_PL_ORBINCL_LOWER)
		
		if MANUAL_PL_ORBLPER_UPPER is not None:
			planet_dict[key]['pl_orblpererr1'] = str(MANUAL_PL_ORBLPER_UPPER)
		if MANUAL_PL_ORBLPER_LOWER is not None:
			planet_dict[key]['pl_orblpererr2'] = str(MANUAL_PL_ORBLPER_LOWER)
		
		if MANUAL_PL_EQT_UPPER is not None:
			planet_dict[key]['pl_eqterr1'] = str(MANUAL_PL_EQT_UPPER)
		if MANUAL_PL_EQT_LOWER is not None:
			planet_dict[key]['pl_eqterr2'] = str(MANUAL_PL_EQT_LOWER)
		
		return planet_dict



	count_noplorbmax = 0
	count_noplorbper = 0
	new_dict = {}
	count_detect = 0
	dict_aux = copy.deepcopy(dictionary)
	countingplanets = 0



	#for key in dictionary:
	#for key in ['1RXS J160929.1-210524 b', '2M0437 b', '2MASS J01033563-5515561 AB b', '2MASS J01225093-2439505 b', '2MASS J02192210-3925225 b', '2MASS J22362452+4751425 b', 'AF Lep b', 'b Cen AB b', 'bet Pic b', 'bet Pic c', 'DH Tau b', 'GSC 06214-00210 b', 'HD 143811 AB b', 'HD 206893 b', 'HD 206893 c', 'HIP 21152 b', 'TYC 8998-760-1 b', 'TYC 8998-760-1 c', 'WISPIT 1 b', 'WISPIT 1 c', 'WISPIT 2 b', '51 Eri b', 'HD 113337 c', 'HIP 79098 AB b', 'HIP 39017 b', 'AB Pic b', 'HD 284149 AB b', 'HIP 65426 b', 'KOINTREAU-1 b', 'kap And b', 'CHXR 73 b', '2MASS J12073346-3932539 b', 'HD 62364 c', 'CD-35 2722 b', 'HIP 78530 b', 'PDS 70 b', 'HR 8799 e', 'HD 111232 c', 'HD 95086 b', 'HR 8799 d', 'HIP 99770 b', 'HR 8799 c', 'PZ Tel b', 'HR 8799 b', 'HD 106906 b', 'HD 128717 b', 'FU Tau b', 'HD 221420 b', 'HIP 54515 b', 'HR 2562 b', 'PDS 70 c', 'AB Aur b', '2MASS J22501512+2325342 b', 'HD 169142 b', 'HD 100546 b', 'eps Ind A b', 'ROXs 42 B b', 'HD 97048 b', 'CT Cha b', 'LP 261-75 b', 'GJ 504 b', 'ROXs 12 b', 'USco1621 b', 'GQ Lup b', 'USco1556 b', '2MASS J0249-0557 c', 'HIP 5158 c', 'HD 73256 c', 'iot Dra c', 'HD 11506 d', 'BD+60 1417 b', 'GU Psc b', 'HD 204313 e', 'HN Peg b', 'HD 28185 c', 'HD 165131 b', 'HD 68988 c']:
	#for key in ['GJ 832 b']: #
	#for key in ['HD 39091 b']:
	#for key in ['Gaia-1 b','Gaia-2 b','Gaia-4 b', 'Gaia-5 b']: #
	#for key in ['AB Pic b']: #This is for Pauline project about second hypothetical planet, b is discovered by Gael 
	#for key in ['51 Eri b','AF Lep b','bet Pic b','HD 95086 b','HIP 65426 b','HIP 99770 b','HR 8799 b','HR 8799 c','HR 8799 d','HR 8799 e','kap And b']: #Targets for ROMAN from GAEL and MATthieu ,'HIP 64892 B','HIP 54515 B', 'HR2562' brown dwarfs daher nicht in NASA database
	#for key in ['HIP 65426 b','HIP 99770 b','kap And b']: #
	#for key in ['Proxima Cen b','HD 39091 b']: #If we only want to compute the detectability of one planet, we can do it this way
	#for key in ['Proxima Cen b']: #If we only want to compute the detectability of one planet, we can do it this way
	#for key in ['bet Pic b']: #If we only want to compute the detectability of one planet, we can do it this way
	#for key in ['47 UMa b']: #If we only want to compute the detectability of one planet, we can do it this way
	#for key in ['HD 95086 b']: #If we only want to compute the detectability of one planet, we can do it this way
	#for key in ['HD 66428 c']:
	#for key in ['HD 145675 c']: #
	#for key in ['HIP 54515 B']: #
	#for key in ['51 Eri b']: #
	#for key in ['PDS 70 b']: #
	#for key in ['HD 10180 #']: #
	#for key in ['bet Pic b']: #
	#for key in ['AF Lep b']: #
	for key in ['HR 8799 e']: #
	#for key in ['51 Peg b']: #
	#for key in ['eps Ind A b', '14 Her b']: #
	#for key in ['HD 39091 b']: #
	#for key in ['HD 95086 b']: #
	#for key in ['HIP 65426 b']: #
	#for key in ['HIP 99770 b']: #
	#for key in ['HD 111232 b']: #
	#for key in ['AF Lep b']: #
	#for key in ['GJ 676 A c','HD 66428 c','HD 89839 b','HD 141937 b','HD 128311 c','HD 128311 b','HD 98649 b','HIP 5158 c']:#,'HD 13724 b', 'HD 30246 b', HD 209262 b, 'GJ 912 b' Brown dwarf 25-45 MJup
	#for key in ['HD 73256 c','HD 11506 d','HD 128311 c','HD 111232 b','HD 111232 c']: #no age ,'GJ 676 A c', This is the list of targets that Sasha and his previous MSc student sent me
	#for key in ['HR 8799 e', 'HR 8799 d']: #
	#for key in ['HD 62509 b']: #very bright mag =1.16, close 10.3pc, massive 2.3Mjup, large orbit 1.64AU N-band target


		print(countingplanets, dictionary[key]['pl_name'])
		countingplanets += 1


		if countingplanets < 0:
			countingplanets += 1
			continue



		# Apply manual parameter overrides if any are set
		dictionary = apply_manual_overrides(dictionary, key)

		print('Age of System', dictionary[key]['st_age'], dictionary[key]['st_ageerr1'], dictionary[key]['st_ageerr2'])
		if dictionary[key]['st_age'] == "":
			print('no age available, skipping t#o next planet')
			continue


		#if dictionary[key]['sy_kmag'] == '':
			#print('no k magnitude available, skipping to next planet')
			#continue
		if dictionary[key]['pl_orbsmax'] == '' and dictionary[key]['pl_orbper'] == '':
			print('no orbital parameters available, skipping to next planet')
			continue
		if dictionary[key]['sy_dist'] == '':
			print('no distance available, skipping to next planet')
			continue

		#here is the lower limmit of mass for BEX Model when mjup
		if dictionary[key]['pl_massj'] == '' and dictionary[key]['pl_msinij'] == '':
			print('no mass available, skipping to next planet')
			continue
		if dictionary[key]['pl_massj'] != '':
			if dictionary[key]['pl_massjerr1'] == '':
				pl_jup_mass_1sigma_up = float(dictionary[key]['pl_massj']) * 1.1
			else:
				pl_jup_mass_1sigma_up = float(dictionary[key]['pl_massj']) + float(dictionary[key]['pl_massjerr1'])
			if pl_jup_mass_1sigma_up < 0.02:
				print('planet mass + upper uncertainty smaller than 0.02, skipping to next planet')
				continue

		#here is the lower limmit of mass for BEX Model when msini
		if dictionary[key]['pl_msinij'] != '':
			if dictionary[key]['pl_msinijerr1'] == '':
				pl_jup_mass_1sigma_up = float(dictionary[key]['pl_msinij']) * 2
			else:
				pl_jup_mass_1sigma_up = float(dictionary[key]['pl_msinij']) + 1 * float(dictionary[key]['pl_msinijerr1'])
			if pl_jup_mass_1sigma_up < 0.02:
				print('planet mass sin i + 1 * upper uncertainty smaller than 0.02, skipping to next planet')
				continue


		#if dictionary[key]['sy_vmag'] == '':
			#print('no magnitude available, skipping to next planet')
			#continue


		#METIS declination range is -90 to +30 degrees
		if dictionary[key]['dec'] == '' or not (-90 <= float(dictionary[key]['dec']) <= 30):
			print('outside of METIS declination available, skipping to next planet')
			continue

			##MANUALL STELAR MAGNITUDE CUT OFF#######
		#if float(dictionary[key]['sy_kmag']) > 12: #(> 12 L-Band) (>10 M-Band) (>3 N-Band)
		#if float(dictionary[key]['sy_vmag']) > 7: #ROMAN
			#print('v/k magnitude too dimm, skipping to next planet')
			#continue


		#\if float(dictionary[key]['sy_kmag']) < 7:
			#print('k magnitude too bright, skipping to next planet')
			#continue		
		#elif dictionary[key]['pl_name'] == 'GJ 900 b': #overload ram
		#	continue




		#Standard orbital simulations with NASA Exoplanet Archive data
		(detectable, angproj_arr, Fp_Fstar_arr, observ_alpha, observ_X, observ_Y, NOTobserv_X, NOTobserv_Y, t_tp_arr, Teq_arr, observ_Teq_arr, NOTobserv_Teq_arr, Fp_Fstar_thermal_arr, detectable_thermal, observ_alpha_thermal, observ_X_thermal, observ_Y_thermal, NOTobserv_X_thermal, NOTobserv_Y_thermal, values_Mp, values_Temp, values_ecc, values_incl, values_orblper, values_abs_mag, values_age_gyr, Transmission_curve, dates_arr, observ_angproj, observ_Fp_Fstar, NOTobserv_angproj, NOTobserv_Fp_Fstar, values_orbper, values_orbtper, observ_angproj_thermal, NOTobserv_angproj_thermal, observ_Fp_Fstar_thermal, NOTobserv_Fp_Fstar_thermal, Fp_Fstar_total_arr, observ_Fp_Fstar_total, NOTobserv_Fp_Fstar_total) = run_multiple_orbital_simulations(dictionary, dict_aux, key, use_mass2mag, use_madys)
		#detectable_iconstrained, angproj_arr_iconstrained, Fp_Fstar_arr_iconstrained, observ_alpha_iconstrained, observ_X_iconstrained, observ_Y_iconstrained, NOTobserv_X_iconstrained, NOTobserv_Y_iconstrained, t_tp_arr_iconstrained = run_multiple_orbital_simulations(dictio_alternative, dict_aux, key)		



		#or sum(detectable_thermal) >= 1
		if sum(detectable)>=1 or sum(detectable_thermal) >= 1:	#If there is at least one detectable orbital position, in any of the 1000 orbital realizations, we write the detectability file of the planet
			count_detect += 1

		# -------------------------------------------------------------------------
		# CSV SUMMARY LOGGING
		# Appends one row per planet per run to LOCATIS_run_summary_<filter>_<model>.csv in route/
		# Toggle on/off with WRITE_SUMMARY_CSV at the top of the config section
		# -------------------------------------------------------------------------

		# Compute stellar magnitude and K-band magnitude unconditionally (used for CSV and summary plot)
		_st_mag_for_log = get_stellar_magnitude_for_band(dictionary[key], stellar_mag_priority)
		if _st_mag_for_log is None:
			_st_mag_for_log = float('nan')
		_kmag_for_log = float(dictionary[key]['sy_kmag']) if dictionary[key].get('sy_kmag', '') != '' else float('nan')

		if WRITE_SUMMARY_CSV:
			import csv as _csv
			import datetime as _datetime
			# Filename includes filter, model and run date to keep runs separated
			_safe_filter = FILTER_USED.replace(' ', '_')
			_safe_model  = Madys_Modell_selection.replace(' ', '_')
			_run_date    = _datetime.date.today().strftime('%Y-%m-%d')
			_csv_path = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter}_{_safe_model}_{_run_date}.csv')
			_csv_header = [
				'timestamp', 'planet_name', 'filter', 'model', 'nrun',
				'st_age_Gyr', 'st_ageerr1_Gyr', 'st_ageerr2_Gyr',
				'det_prob_thermal_%', 'det_prob_total_%',
				'median_contrast_thermal', 'p16_contrast_thermal', 'p84_contrast_thermal',
				'median_contrast_reflected', 'p16_contrast_reflected', 'p84_contrast_reflected',
				'median_contrast_total', 'p16_contrast_total', 'p84_contrast_total',
				'median_angsep_mas', 'p16_angsep_mas', 'p84_angsep_mas',
				'median_planet_temp_K', 'p16_planet_temp_K', 'p84_planet_temp_K',
				'st_mag', 'sy_kmag',
				'sy_dist_pc', 'pl_orbsmax_au',
				'sy_disterr1_pc', 'sy_disterr2_pc', 'pl_orbsmaxerr1_au', 'pl_orbsmaxerr2_au',
				'median_abs_mag', 'p16_abs_mag', 'p84_abs_mag',
				'median_mass_mjup', 'p16_mass_mjup', 'p84_mass_mjup',
				'discoverymethod',
				'rv_flag', 'pul_flag', 'ptv_flag', 'tran_flag', 'ast_flag',
				'obm_flag', 'micro_flag', 'etv_flag', 'ima_flag', 'dkin_flag',
			]
			_timestamp = _datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			_det_thermal  = (sum(detectable_thermal) / nrun) * 100.0
			_det_reflected = (sum(detectable) / nrun) * 100.0 if RUN_REFLECTED_LIGHT else float('nan')
			# Age of system with errors (Gyr) — strip whitespace/newlines from raw archive strings
			_st_age     = str(dictionary[key]['st_age']).strip()     if dictionary[key]['st_age']      != '' else 'N/A'
			_st_ageerr1 = str(dictionary[key]['st_ageerr1']).strip() if dictionary[key]['st_ageerr1']  != '' else 'N/A'
			_st_ageerr2 = str(dictionary[key]['st_ageerr2']).strip() if dictionary[key]['st_ageerr2']  != '' else 'N/A'
			# Contrast stats: median, 16th and 84th percentile over all runs and timepoints
			# Using median instead of mean: thermal contrast distributions are highly right-skewed
			# (young-age tail samples can give orders-of-magnitude higher contrasts), causing
			# mean > p84. The median is always guaranteed to lie between p16 and p84.
			_th_vals = np.array([v for _run in Fp_Fstar_thermal_arr for v in _run if not np.isnan(v)])
			if len(_th_vals) > 0:
				_median_th = float(np.median(_th_vals))
				_p16_th    = float(np.percentile(_th_vals, 16))
				_p84_th    = float(np.percentile(_th_vals, 84))
			else:
				_median_th = _p16_th = _p84_th = float('nan')
			if RUN_REFLECTED_LIGHT and len(Fp_Fstar_arr) > 0:
				_ref_vals = np.array([v for _run in Fp_Fstar_arr for v in _run if not np.isnan(v)])
				if len(_ref_vals) > 0:
					_median_ref = float(np.median(_ref_vals))
					_p16_ref    = float(np.percentile(_ref_vals, 16))
					_p84_ref    = float(np.percentile(_ref_vals, 84))
				else:
					_median_ref = _p16_ref = _p84_ref = float('nan')
			else:
				_median_ref = _p16_ref = _p84_ref = float('nan')
			if RUN_REFLECTED_LIGHT and len(Fp_Fstar_total_arr) > 0:
				_tot_vals = np.array([v for _run in Fp_Fstar_total_arr for v in _run if not np.isnan(v)])
				if len(_tot_vals) > 0:
					_median_tot = float(np.median(_tot_vals))
					_p16_tot    = float(np.percentile(_tot_vals, 16))
					_p84_tot    = float(np.percentile(_tot_vals, 84))
				else:
					_median_tot = _p16_tot = _p84_tot = float('nan')
			else:
				_median_tot = _p16_tot = _p84_tot = float('nan')
			# Angular separation stats: median, 16th and 84th percentile over all runs and timepoints
			_angsep_vals = np.array([v for _run in angproj_arr for v in _run if not np.isnan(v)])
			if len(_angsep_vals) > 0:
				_median_angsep = float(np.median(_angsep_vals))
				_p16_angsep    = float(np.percentile(_angsep_vals, 16))
				_p84_angsep    = float(np.percentile(_angsep_vals, 84))
			else:
				_median_angsep = _p16_angsep = _p84_angsep = float('nan')
			# Planet effective temperature stats over all runs
			_temp_vals = np.array([v for v in values_Temp if not np.isnan(v)]) if len(values_Temp) > 0 else np.array([])
			if len(_temp_vals) > 0:
				_median_temp = float(np.median(_temp_vals))
				_p16_temp    = float(np.percentile(_temp_vals, 16))
				_p84_temp    = float(np.percentile(_temp_vals, 84))
			else:
				_median_temp = _p16_temp = _p84_temp = float('nan')
			# Discovery method and detection flags
			_pdict = dictionary[key]
			_disc_method = _pdict.get('discoverymethod', 'N/A').strip()
			_flag_cols = ['rv_flag', 'pul_flag', 'ptv_flag', 'tran_flag', 'ast_flag',
			              'obm_flag', 'micro_flag', 'etv_flag', 'ima_flag', 'dkin_flag']
			_flag_vals = [_pdict.get(fc, '').strip() if _pdict.get(fc, '').strip() != '' else 'N/A'
			              for fc in _flag_cols]
			# Distance [pc] and semi-major axis [AU] — direct from NASA archive
			_sy_dist_pc      = float(_pdict['sy_dist'])       if _pdict.get('sy_dist', '')       != '' else float('nan')
			_pl_orbsmax      = float(_pdict['pl_orbsmax'])    if _pdict.get('pl_orbsmax', '')    != '' else float('nan')
			_sy_disterr1_pc  = float(_pdict['sy_disterr1'])   if _pdict.get('sy_disterr1', '')   != '' else float('nan')
			_sy_disterr2_pc  = float(_pdict['sy_disterr2'])   if _pdict.get('sy_disterr2', '')   != '' else float('nan')
			_pl_orbsmaxerr1  = float(_pdict['pl_orbsmaxerr1']) if _pdict.get('pl_orbsmaxerr1', '') != '' else float('nan')
			_pl_orbsmaxerr2  = float(_pdict['pl_orbsmaxerr2']) if _pdict.get('pl_orbsmaxerr2', '') != '' else float('nan')
			# Absolute magnitude stats over all runs
			_absmag_vals = np.array([v for v in values_abs_mag if not np.isnan(v)]) if len(values_abs_mag) > 0 else np.array([])
			if len(_absmag_vals) > 0:
				_median_absmag = float(np.median(_absmag_vals))
				_p16_absmag    = float(np.percentile(_absmag_vals, 16))
				_p84_absmag    = float(np.percentile(_absmag_vals, 84))
			else:
				_median_absmag = _p16_absmag = _p84_absmag = float('nan')
			# Planet mass stats over all runs [MJup]
			_mass_vals = np.array([v for v in values_Mp if not np.isnan(v)]) if len(values_Mp) > 0 else np.array([])
			if len(_mass_vals) > 0:
				_median_mass = float(np.median(_mass_vals))
				_p16_mass    = float(np.percentile(_mass_vals, 16))
				_p84_mass    = float(np.percentile(_mass_vals, 84))
			else:
				_median_mass = _p16_mass = _p84_mass = float('nan')
			_write_header = not os.path.exists(_csv_path)
			with open(_csv_path, 'a', newline='') as _f:
				_writer = _csv.writer(_f)
				if _write_header:
					_writer.writerow(_csv_header)
				_writer.writerow([
					_timestamp,
					dictionary[key]['pl_name'],
					FILTER_USED,
					Madys_Modell_selection,
					nrun,
					_st_age, _st_ageerr1, _st_ageerr2,
					f'{_det_thermal:.2f}',
					f'{_det_reflected:.2f}' if RUN_REFLECTED_LIGHT else 'N/A',
					f'{_median_th:.4e}',  f'{_p16_th:.4e}',  f'{_p84_th:.4e}',
					f'{_median_ref:.4e}' if RUN_REFLECTED_LIGHT else 'N/A',
					f'{_p16_ref:.4e}'    if RUN_REFLECTED_LIGHT else 'N/A',
					f'{_p84_ref:.4e}'    if RUN_REFLECTED_LIGHT else 'N/A',
					f'{_median_tot:.4e}' if RUN_REFLECTED_LIGHT else 'N/A',
					f'{_p16_tot:.4e}'    if RUN_REFLECTED_LIGHT else 'N/A',
					f'{_p84_tot:.4e}'    if RUN_REFLECTED_LIGHT else 'N/A',
					f'{_median_angsep:.2f}', f'{_p16_angsep:.2f}', f'{_p84_angsep:.2f}',
					f'{_median_temp:.2f}' if not np.isnan(_median_temp) else 'N/A',
					f'{_p16_temp:.2f}'    if not np.isnan(_p16_temp)    else 'N/A',
					f'{_p84_temp:.2f}'    if not np.isnan(_p84_temp)    else 'N/A',
					f'{_st_mag_for_log:.4f}' if not np.isnan(_st_mag_for_log) else 'N/A',
					f'{_kmag_for_log:.4f}'   if not np.isnan(_kmag_for_log)   else 'N/A',
					f'{_sy_dist_pc:.4f}'       if not np.isnan(_sy_dist_pc)      else 'N/A',
					f'{_pl_orbsmax:.4f}'       if not np.isnan(_pl_orbsmax)      else 'N/A',
					f'{_sy_disterr1_pc:.4f}'   if not np.isnan(_sy_disterr1_pc)  else 'N/A',
					f'{_sy_disterr2_pc:.4f}'   if not np.isnan(_sy_disterr2_pc)  else 'N/A',
					f'{_pl_orbsmaxerr1:.4f}'   if not np.isnan(_pl_orbsmaxerr1)  else 'N/A',
					f'{_pl_orbsmaxerr2:.4f}'   if not np.isnan(_pl_orbsmaxerr2)  else 'N/A',
					f'{_median_absmag:.4f}'  if not np.isnan(_median_absmag)  else 'N/A',
					f'{_p16_absmag:.4f}'     if not np.isnan(_p16_absmag)     else 'N/A',
					f'{_p84_absmag:.4f}'     if not np.isnan(_p84_absmag)     else 'N/A',
					f'{_median_mass:.4f}'    if not np.isnan(_median_mass)    else 'N/A',
					f'{_p16_mass:.4f}'       if not np.isnan(_p16_mass)       else 'N/A',
					f'{_p84_mass:.4f}'       if not np.isnan(_p84_mass)       else 'N/A',
					_disc_method,
					*_flag_vals,
				])
			print(f'  CSV logged: {dictionary[key]["pl_name"]} -> {_csv_path}')

		# -------------------------------------------------------------------------
		# INPUT TABLE LOGGING (NASA archive values, one row per simulated planet)
		# -------------------------------------------------------------------------
		if WRITE_INPUT_TABLE_CSV:
			import csv as _csv_inp
			import datetime as _datetime_inp
			_safe_filter_inp = FILTER_USED.replace(' ', '_')
			_safe_model_inp  = Madys_Modell_selection.replace(' ', '_')
			_run_date_inp    = _datetime_inp.date.today().strftime('%Y-%m-%d')
			_inp_csv_path    = os.path.join(route, f'LOCATIS_input_table_{_safe_filter_inp}_{_safe_model_inp}_{_run_date_inp}.csv')
			_inp_header = [
				'planet_name',
				'sy_dist_pc', 'sy_disterr1_pc', 'sy_disterr2_pc',
				'pl_orbper_days', 'pl_orbpererr1_days', 'pl_orbpererr2_days',
				'pl_orbsmax_au', 'pl_orbsmaxerr1_au', 'pl_orbsmaxerr2_au',
				'pl_mass_mjup', 'pl_masserr1_mjup', 'pl_masserr2_mjup',
				'pl_orbincl_deg', 'pl_orbinclerr1_deg', 'pl_orbinclerr2_deg',
				'pl_orbeccen', 'pl_orbeccenerr1', 'pl_orbeccenerr2',
				'pl_orblper_deg', 'pl_orblpererr1_deg', 'pl_orblpererr2_deg',
				'st_spectype',
				'st_mass_msun', 'st_masserr1_msun', 'st_masserr2_msun',
				'st_age_gyr', 'st_ageerr1_gyr', 'st_ageerr2_gyr',
				'st_lband_mag', 'sy_kmag',
			]
			_inp_pd = dictionary[key]
			def _fi(val):
				try:
					v = float(str(val).strip()) if str(val).strip() not in ('', 'N/A') else float('nan')
				except (ValueError, TypeError):
					v = float('nan')
				return f'{v:.4f}' if not np.isnan(v) else 'N/A'
			_inp_mass     = _inp_pd.get('pl_bmassj', '')     or _inp_pd.get('pl_massj', '')     or _inp_pd.get('pl_msinij', '')
			_inp_masserr1 = _inp_pd.get('pl_bmassjerr1', '') or _inp_pd.get('pl_massjerr1', '') or _inp_pd.get('pl_msinijerr1', '')
			_inp_masserr2 = _inp_pd.get('pl_bmassjerr2', '') or _inp_pd.get('pl_massjerr2', '') or _inp_pd.get('pl_msinijerr2', '')
			_inp_write_header = not os.path.exists(_inp_csv_path)
			with open(_inp_csv_path, 'a', newline='') as _f_inp:
				_writer_inp = _csv_inp.writer(_f_inp)
				if _inp_write_header:
					_writer_inp.writerow(_inp_header)
				_writer_inp.writerow([
					_inp_pd.get('pl_name', key),
					_fi(_inp_pd.get('sy_dist', '')),        _fi(_inp_pd.get('sy_disterr1', '')),     _fi(_inp_pd.get('sy_disterr2', '')),
					_fi(_inp_pd.get('pl_orbper', '')),      _fi(_inp_pd.get('pl_orbpererr1', '')),   _fi(_inp_pd.get('pl_orbpererr2', '')),
					_fi(_inp_pd.get('pl_orbsmax', '')),     _fi(_inp_pd.get('pl_orbsmaxerr1', '')),  _fi(_inp_pd.get('pl_orbsmaxerr2', '')),
					_fi(_inp_mass), _fi(_inp_masserr1), _fi(_inp_masserr2),
					_fi(_inp_pd.get('pl_orbincl', '')),     _fi(_inp_pd.get('pl_orbinclerr1', '')),  _fi(_inp_pd.get('pl_orbinclerr2', '')),
					_fi(_inp_pd.get('pl_orbeccen', '')),    _fi(_inp_pd.get('pl_orbeccenerr1', '')), _fi(_inp_pd.get('pl_orbeccenerr2', '')),
					_fi(_inp_pd.get('pl_orblper', '')),     _fi(_inp_pd.get('pl_orblpererr1', '')),  _fi(_inp_pd.get('pl_orblpererr2', '')),
					_inp_pd.get('st_spectype', 'N/A').strip() or 'N/A',
					_fi(_inp_pd.get('st_mass', '')),        _fi(_inp_pd.get('st_masserr1', '')),     _fi(_inp_pd.get('st_masserr2', '')),
					_fi(_inp_pd.get('st_age', '')),         _fi(_inp_pd.get('st_ageerr1', '')),      _fi(_inp_pd.get('st_ageerr2', '')),
					f'{_st_mag_for_log:.4f}' if not np.isnan(_st_mag_for_log) else 'N/A',
					f'{_kmag_for_log:.4f}'   if not np.isnan(_kmag_for_log)   else 'N/A',
				])
			print(f'  Input table logged: {_inp_pd.get("pl_name", key)} -> {_inp_csv_path}')

		if plotfigs==True:
			###############################################################################
			#The 2-subplot figure with the contrast-angproj tracks + the observable alphas
			
			
			#Manuell Plot planet namen ändern nur für Plot Beschriftung
			#dictionary[key]['pl_name'] = 'AB Pic c'
			#print('manuall planet name enabled')

			textsize = 16

    
			# Create appropriate range for plotting
			# Extended to match the configured OWA values
			sep_min = 50  # mas (below typical IWA but good for plotting)
			if owa_extension is None:
				sep_max = 500
			else:
				sep_max = owa_extension  # Use OWA from config
			sep_array = np.linspace(sep_min, sep_max, 3000)
			#print(sep_array)
			
			# Get stellar magnitude for the selected filter
			st_mag = get_stellar_magnitude_for_band(dictionary[key], stellar_mag_priority)
			if st_mag is None:
				print("Stellar magnitude not found, using default value of 5.0")
				st_mag = 5.0

			# K-band magnitude for METIS contrast curve interpolation (curves are labelled by K-mag)
			kmag_val = float(dictionary[key]['sy_kmag']) if dictionary[key].get('sy_kmag', '') != '' else None
				
			# Determine band letter for contrast curve
			band_letter = FILTER_USED
			
			dist_str = "Normal" if SPLIT_GAUSSIAN else "Uniform"
				
			min_contrast_arr = minimum_contrast_METIS(sep_array, band_letter, st_mag, kmag=kmag_val)

			if RUN_REFLECTED_LIGHT == True and PLOT_CONTRAST_ALPHAS_SUBPLOTS:
				fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(6, 3))
				figname = '%s/%s_nrun%s_contrast_angsep_AND_alphas_subplots_%s_%s'%(route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)
				if 'dictio_alternative' in locals():
					figname = '%s/%s_nrun%s_contrast_angsep_AND_alphas_subplots_%s_WITH-i-constraints_%s'%(route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)
				fig.suptitle(dictionary[key]['pl_name'], fontsize=textsize)

				# Plot contrast limit for the selected filter
				if min_contrast_arr is not None:
					ax1.plot(sep_array, min_contrast_arr, '--g', label=f'{FILTER_USED} Limit')
					ax1.fill_between(sep_array, min_contrast_arr, 1., color='g', alpha=0.05)
				
				ax1.legend(fontsize=textsize-8, loc='upper right')
		

				for l in range(len(Fp_Fstar_arr)):
					ax1.plot(angproj_arr[l], Fp_Fstar_arr[l], linestyle='-', color='k', alpha=0.05)
				if 'dictio_alternative' in locals():
					for l in range(len(Fp_Fstar_arr_iconstrained)):
						ax1.plot(angproj_arr_iconstrained[l], Fp_Fstar_arr_iconstrained[l], linestyle='-', color='r', alpha=0.05)
				xmaxi = ax1.get_xlim()[1]
				ymaxi = ax1.get_ylim()[1]

				ax1.set_ylabel('$F_p / F_*$ (reflected light)', fontsize=textsize)
				ax1.set_xlabel(r"$\Delta \theta$ [mas]", fontsize=textsize)
				ax1.set_ylim(ymin=1.E-11, ymax=1E-2)
				ax1.set_xlim(xmin=0, xmax=100)
				#ax1.set_xlim(xmin=5, xmax=5000)
				#ax1.set_xscale('log')	
				def magnitude(value):
					if (value == 0): return 0
					return int(math.floor(math.log10(abs(value))))
				minvaluex = np.min(angproj_arr)
				maxvaluex = np.max(angproj_arr)
				minvaluey = np.min(Fp_Fstar_arr)
				maxvaluey = np.max(Fp_Fstar_arr)
				minvaluexmag = magnitude(minvaluex)
				maxvaluexmag = magnitude(maxvaluex)
				ax1.set_xlim(xmin=minvaluex*0.8, xmax=maxvaluex*1.4)
				ax1.set_ylim(ymin=1.E-11, ymax=0.001)
				if np.abs(minvaluexmag)+np.abs(maxvaluexmag) > 4:
					ax1.set_xscale('log')
				ax1.set_yscale('log')
				ax1.minorticks_on()
				ax1.tick_params(labelsize=textsize-2)
				

				ax2.hist(np.asarray(observ_alpha).flatten()*180./np.pi, bins=36, range=(0.,180.), color="forestgreen", alpha=0.3, zorder=0)
				ax2.hist(np.asarray(observ_alpha).flatten()*180./np.pi, bins=36, range=(0.,180.), histtype='step', color="forestgreen", zorder=5)
				if 'dictio_alternative' in locals():
					ax2.hist(np.asarray(alphas_arr).flatten()*180./np.pi, bins=36, range=(0.,180.), color="r", alpha=0.3, zorder=0)
					ax2.hist(np.asarray(alphas_arr).flatten()*180./np.pi, bins=36, range=(0.,180.), histtype='step', color="r", zorder=5)
				ax2.tick_params(labelsize=textsize-2)
				ax2.set_xlabel(r"$\alpha_{obs}$ [deg]", fontsize=textsize)
				ax2.set_xlim(0., 180.)
				ax2.set_xticks([0., 90., 180.])
				plt.gca().axes.get_yaxis().set_visible(False)
				fig.savefig(figname+'.png', bbox_inches='tight')
				#fig.savefig(figname+'.pdf', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.pdf"%figname)







			###############################################################################
			#Plotting the detectability windows vs. time
			#print(dictionary[key]['pl_orbtper'])
			if PLOT_DETECTABILITY_WINDOWS:
				if dictionary[key]['pl_orbtper'] != '':
					if RUN_REFLECTED_LIGHT:
						detectability_windows(t_tp_arr, dates_arr, angproj_arr, Fp_Fstar_total_arr, observ_angproj, observ_Fp_Fstar_total, NOTobserv_angproj, NOTobserv_Fp_Fstar_total, values_orbper, values_orbtper)
					else:
						detectability_windows(t_tp_arr, dates_arr, angproj_arr, Fp_Fstar_thermal_arr, observ_angproj_thermal, observ_Fp_Fstar_thermal, NOTobserv_angproj_thermal, NOTobserv_Fp_Fstar_thermal, values_orbper, values_orbtper)
				else:
					print("No time of periastron passage available for ", dictionary[key]['pl_name'])



			###############################################################################
			#Plotting the angular separation in a two-dimension plot for thermal emission 
			textsize = 16
			
			# Use the single thermal band data
			if PLOT_2D_ANGSEP_MAP and len(observ_X_thermal) > 0:
				# Define IWA and OWA from the contrast curve
				if min_contrast_arr is not None:
					valid_indices = np.where(min_contrast_arr < 1E90)[0]
					if len(valid_indices) > 0:
						IWA_band = sep_array[valid_indices[0]]
						OWA_band = sep_array[valid_indices[-1]]
					else:
						# Fallback if no valid contrast values
						IWA_band = 80
						OWA_band = owa_extension

						
				else:
					# No contrast curve available
					IWA_band = 80
					OWA_band = owa_extension
				
				# Calculate max extent of observed and non-observed positions to set plot limits
				max_extent = 0
				for l in range(len(observ_X_thermal)):
					if len(observ_X_thermal[l]) > 0:
						x_vals = np.asarray(observ_X_thermal[l])/(float(dictionary[key]['sy_dist']))*1000.
						y_vals = np.asarray(observ_Y_thermal[l])/(float(dictionary[key]['sy_dist']))*1000.
						max_extent = max(max_extent, np.max(np.abs(x_vals)), np.max(np.abs(y_vals)))
				for l in range(len(NOTobserv_X_thermal)):
					if len(NOTobserv_X_thermal[l]) > 0:
						x_vals = np.asarray(NOTobserv_X_thermal[l])/(float(dictionary[key]['sy_dist']))*1000.
						y_vals = np.asarray(NOTobserv_Y_thermal[l])/(float(dictionary[key]['sy_dist']))*1000.
						max_extent = max(max_extent, np.max(np.abs(x_vals)), np.max(np.abs(y_vals)))
				
				# Set plot limits to max of 800 or 1.5 times the max extent
				agseplim = int(max(750, max_extent * 1.5)) if max_extent > 0 else 800
				angsepstep = np.arange(-agseplim, agseplim+1, 1)
				xx, yy = np.meshgrid(angsepstep, angsepstep)
				angsepradial = np.sqrt(np.power(xx,2.)+np.power(yy,2.))
				
				# Create masks for IWA and OWA regions
				mask_inside_IWA = angsepradial < IWA_band
				mask_outside_OWA = angsepradial > OWA_band
				
				fig, ax1 = plt.subplots(figsize=(6,6))
				figname = '%s/%s_nrun%s_2Dangsep-map_thermal_%s_%s'%(route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)
				plt.title(dictionary[key]['pl_name'] + f' - Thermal {FILTER_USED}', fontsize=textsize)
				
				# Set background to white initially
				ax1.set_facecolor('white')
				
				# Fill regions with black: inside IWA and outside OWA
				ax1.contourf(xx, yy, mask_inside_IWA.astype(int), levels=[0.5, 1.5], colors=['black'], alpha=1.0, zorder=0)
				ax1.contourf(xx, yy, mask_outside_OWA.astype(int), levels=[0.5, 1.5], colors=['black'], alpha=1.0, zorder=0)
				
				for l in range(0, len(observ_X_thermal), 1):
					ax1.plot(np.asarray(observ_X_thermal[l])/(float(dictionary[key]['sy_dist']))*1000., np.asarray(observ_Y_thermal[l])/(float(dictionary[key]['sy_dist']))*1000., linestyle='-', color='limegreen', alpha=0.3, zorder=2)
					ax1.plot(np.asarray(NOTobserv_X_thermal[l])/(float(dictionary[key]['sy_dist']))*1000., np.asarray(NOTobserv_Y_thermal[l])/(float(dictionary[key]['sy_dist']))*1000., linestyle='-', color='goldenrod', alpha=0.5, zorder=2)
				ax1.plot(0, 0, 'w*')
				ax1.set_xlabel("$\Delta$x [mas]", fontsize=textsize)
				ax1.set_ylabel("$\Delta$y [mas]", fontsize=textsize)
				ax1.plot(IWA_band*np.cos(np.arange(-180., 181., 1.)*np.pi/180.), IWA_band*np.sin(np.arange(-180., 181., 1.)*np.pi/180.), '--r', linewidth=2)
				ax1.plot(OWA_band*np.cos(np.arange(-180., 181., 1.)*np.pi/180.), OWA_band*np.sin(np.arange(-180., 181., 1.)*np.pi/180.), '--r', linewidth=2)
				ax1.set_xlim(xmin=-agseplim, xmax=agseplim)
				ax1.set_ylim(ymin=-agseplim, ymax=agseplim)
				ax1.minorticks_on()
				ax1.tick_params(labelsize=textsize-2)
				#plt.show()
				fig.savefig(figname+'.png', bbox_inches='tight')
				fig.savefig(figname+'.pdf', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.pdf"%figname)


			##############Plots Sebastian#################
			if len(values_Temp) > 0:
				mean_temp = np.nanmean(values_Temp)
				print(f"Average Effective Temperature: {mean_temp:.2f} K")
				median_temp = np.nanmedian(values_Temp)
				print(f"Median Effective Temperature: {median_temp:.2f} K")
			else:
				median_temp = -1 # Or None
				mean_temp = -1 # Or None

			# Thermal flux vs angular separation plot
			if PLOT_THERMAL_FLUX_VS_ANGSEP and len(Fp_Fstar_thermal_arr) > 0:
				fig, ax1 = plt.subplots(figsize=(6, 4))
				figname = '%s/%s_nrun%s_Fp_Fstar_thermal_%s_vs_angsep_%s' % (route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)

				if median_temp > 1:
					ax1.set_title(f"{dictionary[key]['pl_name']} ($T_{{eff}}$={median_temp:.0f} K)", fontsize=textsize)
				else:
					ax1.set_title(f"{dictionary[key]['pl_name']}", fontsize=textsize)

				if min_contrast_arr is not None:
					ax1.plot(sep_array, min_contrast_arr, '--g')
					ax1.fill_between(sep_array, min_contrast_arr, 1., color='g', alpha=0.1)

				pct_detect = (sum(detectable_thermal) / nrun) * 100
				pct_detect_reflected = (sum(detectable) / nrun) * 100 if RUN_REFLECTED_LIGHT else 0
				
				# Plot a subset of orbital realizations (every 10th) to avoid clutter
				for l in range(0, len(Fp_Fstar_thermal_arr), 10):
					label_text = f'{FILTER_USED} ({pct_detect:.0f}% detectable)' if l == 0 else ''
					ax1.plot(angproj_arr[l], Fp_Fstar_thermal_arr[l], linestyle='-', color='black', alpha=0.5, label=label_text)

				ax1.set_xlabel(r"$\Delta \theta$ [mas]", fontsize=textsize)
				ax1.set_ylabel('$F_p / F_*$ (thermal)', fontsize=textsize)
				ax1.set_yscale('log')
				ax1.legend()
				
				# Dynamic limits
				axmax_val = np.max(angproj_arr) * 1.3 if len(angproj_arr) > 0 else None
				if axmax_val:
					ax1.set_xlim(xmin=0, xmax=axmax_val)
				else:
					ax1.set_xlim(xmin=0)
				ax1.set_ylim(ymax=1E-1)

				ax1.minorticks_on()
				ax1.tick_params(labelsize=textsize - 2)

				fig.savefig(figname + '.png', bbox_inches='tight')
				#fig.savefig(figname + '.pdf', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.png" % figname)






			#Reflected plus thermal emission plot 
			if PLOT_REFLECTED_PLUS_THERMAL and RUN_REFLECTED_LIGHT and len(Fp_Fstar_thermal_arr) > 0 and len(Fp_Fstar_arr) > 0:
				fig, ax1 = plt.subplots(figsize=(6, 4))
				figname = '%s/%s_nrun%s_Fp_Fstar_reflected_plus_thermal_%s_vs_angsep_%s' % (route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)

				ax1.set_title(f"{dictionary[key]['pl_name']}", fontsize=textsize)

				if min_contrast_arr is not None:
					ax1.plot(sep_array, min_contrast_arr, '--g')
					ax1.fill_between(sep_array, min_contrast_arr, 1., color='g', alpha=0.1)

				pct_detect = (sum(detectable_thermal) / nrun) * 100
				pct_detect_reflected = (sum(detectable) / nrun) * 100
				
				# Plot a subset of orbital realizations (every 10th) to avoid clutter 
				for l in range(0, len(Fp_Fstar_thermal_arr), 10):
					label_thermal = f'{FILTER_USED} thermal($P_{{det}}$={pct_detect:.0f}%)' if l == 0 else ''
					label_reflected = f'{FILTER_USED} reflected($P_{{det}}$={pct_detect_reflected:.0f}%)' if l == 0 else ''
					label_combined = f'{FILTER_USED} total' if l == 0 else ''
					ax1.plot(angproj_arr[l], Fp_Fstar_thermal_arr[l], linestyle='-', color='r', alpha=0.5, label=label_thermal)#this is for thermal 
					ax1.plot(angproj_arr[l], Fp_Fstar_arr[l], linestyle='-', color='b', alpha=0.5, label=label_reflected)#this is for reflected
					ax1.plot(angproj_arr[l], np.array(Fp_Fstar_thermal_arr[l])+np.array(Fp_Fstar_arr[l]), linestyle='-', color='k', alpha=0.5, label=label_combined)#this is for thermal+reflected  						
				ax1.set_xlabel(r"$\Delta \theta$ [mas]", fontsize=textsize)
				ax1.set_ylabel('$F_p / F_*$', fontsize=textsize)
				ax1.set_yscale('log')
				
				ax1.legend(loc='best')
				
				# Dynamic limits
				axmax_val = np.max(angproj_arr) * 1.3 if len(angproj_arr) > 0 else None
				if axmax_val:
					ax1.set_xlim(xmin=0, xmax=axmax_val)
				else:
					ax1.set_xlim(xmin=0)
				ax1.set_ylim(ymax=1E-1, ymin=1E-13)

				ax1.minorticks_on()
				ax1.tick_params(labelsize=textsize - 2)

				fig.savefig(figname + '.png', bbox_inches='tight')
				#fig.savefig(figname + '.pdf', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.png" % figname)

			# Combined Summary Plot: Flux Ratio (left) and Orbit (right)
			if PLOT_COMBINED_SUMMARY and len(Fp_Fstar_thermal_arr) > 0:
				fig, (ax_flux, ax_orbit) = plt.subplots(1, 2, figsize=(15, 6))
				figname = '%s/%s_nrun%s_Summary_of_orbits_and_total_contrast_%s_%s_%s' % (route, dictionary[key]['pl_name'], str(nrun), namerun, Madys_Modell_selection, dist_str)
				
				# --- Left Plot: Flux Ratio vs Separation ---
				if median_temp > 1:
					ax_flux.set_title(f"{dictionary[key]['pl_name']} Contrast ($T_{{eff}}$={median_temp:.0f} K)", fontsize=textsize)
				else:
					ax_flux.set_title(f"{dictionary[key]['pl_name']} Contrast", fontsize=textsize)

				# Plot Sensitivity Background (Contrast Curve)
				if min_contrast_arr is not None:
					ax_flux.plot(sep_array, min_contrast_arr, '--g')
					ax_flux.fill_between(sep_array, min_contrast_arr, 1., color='g', alpha=0.1)

				pct_detect = (sum(detectable_thermal) / nrun) * 100
				pct_detect_total = (sum(detectable) / nrun) * 100 if RUN_REFLECTED_LIGHT else 0

				# Plot curves
				for l in range(0, len(Fp_Fstar_thermal_arr), 10):
					# Determine Total Flux
					if RUN_REFLECTED_LIGHT and len(Fp_Fstar_arr) > l:
						flux_thermal = np.asarray(Fp_Fstar_thermal_arr[l])
						flux_reflected = np.asarray(Fp_Fstar_arr[l])
						flux_total = flux_thermal + flux_reflected
						
						# Labels only for the first iteration
						label_thermal = f'thermal($P_{{det}}$={pct_detect:.0f}%)' if l == 0 else ''
						label_reflected = f'reflected' if l == 0 else ''
						label_total = f'total($P_{{det}}$={pct_detect_total:.0f}%)' if l == 0 else ''
						
						ax_flux.plot(angproj_arr[l], flux_thermal, linestyle='-', color='r', alpha=0.5, label=label_thermal)
						ax_flux.plot(angproj_arr[l], flux_reflected, linestyle='-', color='b', alpha=0.5, label=label_reflected)
						ax_flux.plot(angproj_arr[l], flux_total, linestyle='-', color='k', alpha=0.5, label=label_total)

					else:
						# Thermal Only Case
						flux_total = np.asarray(Fp_Fstar_thermal_arr[l])
						label_text = f'thermal ({pct_detect:.0f}% detectable)' if l == 0 else ''
						ax_flux.plot(angproj_arr[l], flux_total, linestyle='-', color='k', alpha=0.5, label=label_text)

				ax_flux.set_xlabel(r"$\Delta \theta$ [mas]", fontsize=textsize)
				ax_flux.set_ylabel('$F_p / F_*$', fontsize=textsize)
				ax_flux.set_yscale('log')
				ax_flux.legend(loc='best')
				
				# Dynamic limits for Flux plot
				axmax_val = np.max(angproj_arr) * 1.3 if len(angproj_arr) > 0 else None
				if axmax_val:
					ax_flux.set_xlim(xmin=0, xmax=axmax_val)
				else:
					ax_flux.set_xlim(xmin=0)
				ax_flux.set_ylim(ymax=1E-1, ymin=1E-13)
				ax_flux.minorticks_on()
				ax_flux.tick_params(labelsize=textsize - 2)

				# --- Right Plot: Orbit (Delta RA vs Delta Dec) ---
				ax_orbit.set_title("Orbital Geometry", fontsize=textsize)
				ax_orbit.set_aspect('equal')
				ax_orbit.set_facecolor('white')

				# IWA/OWA Masks
				# Re-create grid for contourf if needed, or reuse xx, yy, mask_inside_IWA, mask_outside_OWA from previous scope
				# Assuming xx, yy, mask_inside_IWA, mask_outside_OWA are still available from the Orbit Plot section
				if 'mask_inside_IWA' in locals() and 'mask_outside_OWA' in locals():
					ax_orbit.contourf(xx, yy, mask_inside_IWA.astype(int), levels=[0.5, 1.5], colors=['black'], alpha=1.0, zorder=0)
					ax_orbit.contourf(xx, yy, mask_outside_OWA.astype(int), levels=[0.5, 1.5], colors=['black'], alpha=1.0, zorder=0)

					# Plot Orbits
					for l in range(0, len(observ_X_thermal), 1):
						# Detectable parts in limegreen
						ax_orbit.plot(np.asarray(observ_X_thermal[l])/(float(dictionary[key]['sy_dist']))*1000., 
									  np.asarray(observ_Y_thermal[l])/(float(dictionary[key]['sy_dist']))*1000., 
									  linestyle='-', color='limegreen', alpha=0.3, zorder=2)
						# Non-detectable parts in yellow
						ax_orbit.plot(np.asarray(NOTobserv_X_thermal[l])/(float(dictionary[key]['sy_dist']))*1000., 
									  np.asarray(NOTobserv_Y_thermal[l])/(float(dictionary[key]['sy_dist']))*1000., 
									  linestyle='-', color='goldenrod', alpha=0.2, zorder=2)
				
				ax_orbit.plot(0, 0, 'w*') # Star
				ax_orbit.set_xlabel("$\Delta$x [mas]", fontsize=textsize)
				ax_orbit.set_ylabel("$\Delta$y [mas]", fontsize=textsize)
				
				# IWA/OWA Red Dashed Circles
				ax_orbit.plot(IWA_band*np.cos(np.arange(-180., 181., 1.)*np.pi/180.), IWA_band*np.sin(np.arange(-180., 181., 1.)*np.pi/180.), '--r', linewidth=2)
				ax_orbit.plot(OWA_band*np.cos(np.arange(-180., 181., 1.)*np.pi/180.), OWA_band*np.sin(np.arange(-180., 181., 1.)*np.pi/180.), '--r', linewidth=2)
				
				ax_orbit.set_xlim(xmin=-agseplim, xmax=agseplim)
				ax_orbit.set_ylim(ymin=-agseplim, ymax=agseplim)
				ax_orbit.minorticks_on()
				ax_orbit.tick_params(labelsize=textsize-2)
				
				# Invert X axis for RA convention (East is left)
				ax_orbit.invert_xaxis()
				# Set aspect to equal box to enforce squareness
				ax_orbit.set_box_aspect(1)

				plt.tight_layout()
				fig.savefig(figname + '.pdf', bbox_inches='tight')
				#fig.savefig(figname + '.png', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.pdf" % figname)









				#HISTOGRAM PLOTS FOR ALL VARIABLES

			if PLOT_MASS_HISTOGRAM and len(values_Mp) > 0:
				fig, ax = plt.subplots(figsize=(8, 6))
				figname = '%s/%s_nrun%s_Mass_Distribution_%s_%s' % (route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)
				
				# Plot Histogram
				# bins='auto' lets matplotlib decide the best bin width
				# alpha=0.7 makes it slightly transparent
				# rwidth=0.85 gives a little gap between bars
				n_hist, bins, patches = ax.hist(values_Mp, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85, edgecolor='black')

				# Add labels and title
				ax.set_xlabel(r'Planet Mass [$M_J$]', fontsize=textsize)
				ax.set_ylabel('Frequency (Count)', fontsize=textsize)
				ax.set_title(f'Mass Distribution for {dictionary[key]["pl_name"]} ({nrun} runs)', fontsize=textsize)
				
				# Optional: Add a vertical line for the median
				median_mass = np.median(values_Mp)
				ax.axvline(median_mass, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_mass:.2f} $M_J$')
				ax.legend()

				# Grid and Ticks
				ax.grid(axis='y', alpha=0.5)
				ax.minorticks_on()
				ax.tick_params(labelsize=textsize - 2)

				# Save the plot
				fig.savefig(figname + '.png', bbox_inches='tight')
				# fig.savefig(figname + '.pdf', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.png" % figname)

			if PLOT_ECC_HISTOGRAM and len(values_ecc) > 0:
				fig, ax = plt.subplots(figsize=(8, 6))
				figname = '%s/%s_nrun%s_Eccentricity_Distribution_%s_%s' % (route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)
				
				n_hist, bins, patches = ax.hist(values_ecc, bins='auto', color='lightgreen', alpha=0.7, rwidth=0.85, edgecolor='black')

				ax.set_xlabel('Eccentricity', fontsize=textsize)
				ax.set_ylabel('Frequency (Count)', fontsize=textsize)
				ax.set_title(f'Eccentricity Distribution for {dictionary[key]["pl_name"]} ({nrun} runs)', fontsize=textsize)
				
				median_ecc = np.median(values_ecc)
				ax.axvline(median_ecc, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_ecc:.2f}')
				ax.legend()

				ax.grid(axis='y', alpha=0.5)
				ax.minorticks_on()
				ax.tick_params(labelsize=textsize - 2)

				fig.savefig(figname + '.png', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.png" % figname)

			if PLOT_INCL_HISTOGRAM and len(values_incl) > 0:
				fig, ax = plt.subplots(figsize=(8, 6))
				figname = '%s/%s_nrun%s_Inclination_Distribution_%s_%s' % (route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)
				
				n_hist, bins, patches = ax.hist(values_incl, bins='auto', color='salmon', alpha=0.7, rwidth=0.85, edgecolor='black')

				ax.set_xlabel('Inclination [deg]', fontsize=textsize)
				ax.set_ylabel('Frequency (Count)', fontsize=textsize)
				ax.set_title(f'Inclination Distribution for {dictionary[key]["pl_name"]} ({nrun} runs)', fontsize=textsize)
				
				median_incl = np.median(values_incl)
				ax.axvline(median_incl, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_incl:.2f} deg')
				ax.legend()

				ax.grid(axis='y', alpha=0.5)
				ax.minorticks_on()
				ax.tick_params(labelsize=textsize - 2)

				fig.savefig(figname + '.png', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.png" % figname)

			if PLOT_TEMP_HISTOGRAM and len(values_Temp) > 0:
				fig, ax = plt.subplots(figsize=(8, 6))
				figname = '%s/%s_nrun%s_Temperature_Distribution_%s_%s' % (route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)
				
				n_hist, bins, patches = ax.hist(values_Temp, bins='auto', color='orange', alpha=0.7, rwidth=0.85, edgecolor='black')

				ax.set_xlabel('Effective Temperature [K]', fontsize=textsize)
				ax.set_ylabel('Frequency (Count)', fontsize=textsize)
				ax.set_title(f'Temperature Distribution for {dictionary[key]["pl_name"]} ({nrun} runs)', fontsize=textsize)
				
				median_temp = np.median(values_Temp)
				ax.axvline(median_temp, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_temp:.2f} K')
				ax.legend()

				ax.grid(axis='y', alpha=0.5)
				ax.minorticks_on()
				ax.tick_params(labelsize=textsize - 2)

				fig.savefig(figname + '.png', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.png" % figname)

			if PLOT_ABS_MAG_HISTOGRAM and len(values_abs_mag) > 0:
				fig, ax = plt.subplots(figsize=(8, 6))
				figname = '%s/%s_nrun%s_AbsMag_Distribution_%s_%s' % (route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)
				
				n_hist, bins, patches = ax.hist(values_abs_mag, bins='auto', color='orchid', alpha=0.7, rwidth=0.85, edgecolor='black')

				ax.set_xlabel('Absolute Magnitude', fontsize=textsize)
				ax.set_ylabel('Frequency (Count)', fontsize=textsize)
				ax.set_title(f'Absolute Magnitude Distribution for {dictionary[key]["pl_name"]} ({nrun} runs)', fontsize=textsize)
				
				median_mag = np.median(values_abs_mag)
				ax.axvline(median_mag, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_mag:.2f}')
				ax.legend()

				ax.grid(axis='y', alpha=0.5)
				ax.minorticks_on()
				ax.tick_params(labelsize=textsize - 2)

				fig.savefig(figname + '.png', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.png" % figname)


			if PLOT_AGE_HISTOGRAM and len(values_age_gyr) > 0:
				fig, ax = plt.subplots(figsize=(8, 6))
				figname = '%s/%s_nrun%s_Age_Distribution_%s_%s' % (route, dictionary[key]['pl_name'], str(nrun), namerun, dist_str)
				
				n_hist, bins, patches = ax.hist(values_age_gyr, bins=1000, color='gold', alpha=0.7, rwidth=0.85, edgecolor='black')

				ax.set_xlabel('Age [Gyr]', fontsize=textsize)
				ax.set_ylabel('Frequency (Count)', fontsize=textsize)
				ax.set_title(f'Age Distribution for {dictionary[key]["pl_name"]} ({nrun} runs)', fontsize=textsize)
				
				median_age = np.median(values_age_gyr)
				ax.axvline(median_age, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_age:.2f} Gyr')
				ax.legend()

				ax.grid(axis='y', alpha=0.5)
				ax.minorticks_on()
				ax.tick_params(labelsize=textsize - 2)

				fig.savefig(figname + '.png', bbox_inches='tight')
				plt.close(fig)
				print("SAVED: %s.png" % figname)

		analyze_march_2027(dictionary[key]['pl_name'], dates_arr, 
						   Fp_Fstar_thermal_arr, Fp_Fstar_arr, Fp_Fstar_total_arr, 
						   observ_Fp_Fstar_thermal, observ_Fp_Fstar, observ_Fp_Fstar_total,
						   csv_filename=os.path.join(route, 'March2027_Analysis.csv'),
						   run_reflected=RUN_REFLECTED_LIGHT,
						   split_gaussian=SPLIT_GAUSSIAN,
						   filter_name=FILTER_USED,
						   entropy_model=Madys_Modell_selection)


	# =========================================================================
	# SUMMARY CONTRAST vs. ANGULAR SEPARATION PLOT
	# Runs after the planet loop. Reads the CSV (either just written or an
	# existing one given by SUMMARY_CSV_RERUN_PATH) and plots all planets
	# whose det_prob_total_% exceeds SUMMARY_CONTRAST_DET_THRESHOLD.
	# Points are colour-coded by stellar age (red = young, blue = old) with
	# error bars for both contrast (p16/p84) and angular separation (p16/p84).
	# =========================================================================
	if PLOT_SUMMARY_CONTRAST_SEP:
		import csv as _csv_sum
		import datetime as _datetime

		# Decide which CSV to read
		_safe_filter_sum = FILTER_USED.replace(' ', '_')
		_safe_model_sum  = Madys_Modell_selection.replace(' ', '_')
		_run_date_sum    = _datetime.date.today().strftime('%Y-%m-%d')
		_default_csv     = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_sum}_{_safe_model_sum}_{_run_date_sum}.csv')
		_sum_csv_path    = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv

		if not os.path.exists(_sum_csv_path):
			print(f'[Summary plot] CSV not found: {_sum_csv_path}  – skipping plot.')
		else:
			print(f'[Summary plot] Reading {_sum_csv_path} …')

			# -----------------------------------------------------------------
			# Read CSV into rows
			# -----------------------------------------------------------------
			_sum_rows = []
			with open(_sum_csv_path, newline='') as _fsum:
				_reader = _csv_sum.DictReader(_fsum)
				for _row in _reader:
					_sum_rows.append(_row)

			if len(_sum_rows) == 0:
				print('[Summary plot] CSV is empty – skipping plot.')
			else:
				# Helper: safe float conversion
				def _sf(val, fallback=float('nan')):
					try:
						v = float(val)
						return v if np.isfinite(v) else fallback
					except (ValueError, TypeError):
						return fallback

				# Decide which contrast columns to use (total if reflected, thermal if not)
				if RUN_REFLECTED_LIGHT:
					_c_median_col = 'median_contrast_total'
					_c_p16_col    = 'p16_contrast_total'
					_c_p84_col    = 'p84_contrast_total'
				else:
					_c_median_col = 'median_contrast_thermal'
					_c_p16_col    = 'p16_contrast_thermal'
					_c_p84_col    = 'p84_contrast_thermal'

				# Filter rows above detection threshold
				_plot_names    = []
				_plot_contrast = []
				_plot_c_lo     = []   # lower error bar size (mean – p16)
				_plot_c_hi     = []   # upper error bar size (p84 – mean)
				_plot_sep      = []
				_plot_s_lo     = []
				_plot_s_hi     = []
				_plot_age      = []
				_plot_st_mag   = []
				_plot_kmag     = []

				for _r in _sum_rows:
					_det = _sf(_r.get('det_prob_total_%', 'nan'))
					if np.isnan(_det) or _det < SUMMARY_CONTRAST_DET_THRESHOLD:
						continue

					_c_median = _sf(_r.get(_c_median_col, 'nan'))
					_c_p16  = _sf(_r.get(_c_p16_col,  'nan'))
					_c_p84  = _sf(_r.get(_c_p84_col,  'nan'))
					_s_median = _sf(_r.get('median_angsep_mas', 'nan'))
					_s_p16  = _sf(_r.get('p16_angsep_mas',  'nan'))
					_s_p84  = _sf(_r.get('p84_angsep_mas',  'nan'))
					_age    = _sf(_r.get('st_age_Gyr', 'nan'))
					_st_mag_r = _sf(_r.get('st_mag',   'nan'))
					_kmag_r   = _sf(_r.get('sy_kmag',  'nan'))

					# Skip if essential values are missing
					if any(np.isnan(v) for v in [_c_median, _s_median]):
						continue

					_plot_names.append(_r.get('planet_name', '?'))
					_plot_contrast.append(_c_median)
					_plot_c_lo.append(max(0.0, _c_median - _c_p16) if not np.isnan(_c_p16) else 0.0)
					_plot_c_hi.append(max(0.0, _c_p84 - _c_median) if not np.isnan(_c_p84) else 0.0)
					_plot_sep.append(_s_median)
					_plot_s_lo.append(max(0.0, _s_median - _s_p16) if not np.isnan(_s_p16) else 0.0)
					_plot_s_hi.append(max(0.0, _s_p84 - _s_median) if not np.isnan(_s_p84) else 0.0)
					_plot_age.append(_age)
					_plot_st_mag.append(_st_mag_r)
					_plot_kmag.append(_kmag_r)

				if len(_plot_names) == 0:
					print(f'[Summary plot] No planets above {SUMMARY_CONTRAST_DET_THRESHOLD}% threshold – skipping plot.')
				else:
					_plot_contrast = np.array(_plot_contrast)
					_plot_sep      = np.array(_plot_sep)
					_plot_age      = np.array(_plot_age)
					_plot_st_mag   = np.array(_plot_st_mag)
					_plot_kmag     = np.array(_plot_kmag)

					# Age colourmap: finite ages drive the scale; NaN ages shown in grey
					_finite_age_mask = np.isfinite(_plot_age)
					_age_min = float(np.nanmin(_plot_age[_finite_age_mask])) if _finite_age_mask.any() else 0.0
					_age_max = float(np.nanmax(_plot_age[_finite_age_mask])) if _finite_age_mask.any() else 1.0
					if _age_max == _age_min:
						_age_max = _age_min + 1.0

					_cmap = plt.cm.plasma
					_norm = matplotlib.colors.Normalize(vmin=_age_min, vmax=_age_max)

					# ----------------------------------------------------------
					# Build 3 contrast curves: brightest (min mag), median, faintest (max mag)
					# For METIS filters use sy_kmag; otherwise use st_mag
					# ----------------------------------------------------------
					_sep_min_sum = 50
					_sep_max_sum = owa_extension if owa_extension is not None else 2000
					_sep_arr_sum = np.linspace(_sep_min_sum, _sep_max_sum, 3000)
					_band_sum    = FILTER_USED

					_is_metis   = FILTER_USED.startswith('METIS')
					_curve_mags = _plot_kmag if _is_metis else _plot_st_mag
					_finite_mag_mask = np.isfinite(_curve_mags)

					if _finite_mag_mask.any():
						_mag_finite    = _curve_mags[_finite_mag_mask]
						_st_mag_finite = _plot_st_mag[_finite_mag_mask]

						_mag_bright = float(np.min(_mag_finite))
						_mag_mid    = float(np.median(_mag_finite))
						_mag_faint  = float(np.max(_mag_finite))

						_idx_bright = int(np.argmin(_curve_mags[_finite_mag_mask]))
						_idx_faint  = int(np.argmax(_curve_mags[_finite_mag_mask]))
						_idx_mid    = int(np.argmin(np.abs(_curve_mags[_finite_mag_mask] - _mag_mid)))

						_st_bright = float(_st_mag_finite[_idx_bright])
						_st_mid    = float(_st_mag_finite[_idx_mid])
						_st_faint  = float(_st_mag_finite[_idx_faint])

						if _is_metis:
							_curve_bright = minimum_contrast_METIS(_sep_arr_sum, _band_sum, _st_bright, kmag=_mag_bright)
							_curve_mid    = minimum_contrast_METIS(_sep_arr_sum, _band_sum, _st_mid,    kmag=_mag_mid)
							_curve_faint  = minimum_contrast_METIS(_sep_arr_sum, _band_sum, _st_faint,  kmag=_mag_faint)
						else:
							_curve_bright = minimum_contrast_METIS(_sep_arr_sum, _band_sum, _mag_bright)
							_curve_mid    = minimum_contrast_METIS(_sep_arr_sum, _band_sum, _mag_mid)
							_curve_faint  = minimum_contrast_METIS(_sep_arr_sum, _band_sum, _mag_faint)

						_valid_bright = _curve_bright < 1e10
						_valid_mid    = _curve_mid    < 1e10
						_valid_faint  = _curve_faint  < 1e10
						_have_curves  = True
					else:
						_have_curves = False

					fig_sum, ax_sum = plt.subplots(figsize=(10, 7))

					# ----------------------------------------------------------
					# Draw contrast curves + shading FIRST (background layer)
					# Region ABOVE a curve is detectable (matching per-planet plots).
					# Stacked layers with increasing alpha going downward:
					#   above faintest curve        → alpha 0.15
					#   between faintest and median  → alpha 0.10
					#   between median and brightest → alpha 0.05
					# ----------------------------------------------------------
					_curve_color = 'dimgray'
					if _have_curves:
						_ylim_top = 1.0

						ax_sum.fill_between(
							_sep_arr_sum[_valid_faint],
							_curve_faint[_valid_faint], _ylim_top,
							color=_curve_color, alpha=0.15, zorder=1
						)
						ax_sum.fill_between(
							_sep_arr_sum[_valid_mid],
							_curve_mid[_valid_mid],
							np.interp(_sep_arr_sum[_valid_mid],
							          _sep_arr_sum[_valid_faint], _curve_faint[_valid_faint],
							          left=_ylim_top, right=_ylim_top),
							color=_curve_color, alpha=0.10, zorder=1
						)
						ax_sum.fill_between(
							_sep_arr_sum[_valid_bright],
							_curve_bright[_valid_bright],
							np.interp(_sep_arr_sum[_valid_bright],
							          _sep_arr_sum[_valid_mid], _curve_mid[_valid_mid],
							          left=_ylim_top, right=_ylim_top),
							color=_curve_color, alpha=0.05, zorder=1
						)

						_mag_label = 'K' if _is_metis else 'st'
						ax_sum.plot(_sep_arr_sum[_valid_faint],  _curve_faint[_valid_faint],
						            '--', color=_curve_color, linewidth=1.2, zorder=2,
						            label=f'Contrast limit ({_mag_label}={_mag_faint:.1f}, faintest)')
						ax_sum.plot(_sep_arr_sum[_valid_mid],    _curve_mid[_valid_mid],
						            '--', color=_curve_color, linewidth=1.2, zorder=2, alpha=0.7,
						            label=f'Contrast limit ({_mag_label}={_mag_mid:.1f}, median)')
						ax_sum.plot(_sep_arr_sum[_valid_bright], _curve_bright[_valid_bright],
						            '--', color=_curve_color, linewidth=1.2, zorder=2, alpha=0.4,
						            label=f'Contrast limit ({_mag_label}={_mag_bright:.1f}, brightest)')

					# Plot planet data points on top
					for _i, _name in enumerate(_plot_names):
						_age_i = _plot_age[_i]
						_color = _cmap(_norm(_age_i)) if np.isfinite(_age_i) else (0.55, 0.55, 0.55, 1.0)
						ax_sum.errorbar(
							_plot_sep[_i], _plot_contrast[_i],
							xerr=[[_plot_s_lo[_i]], [_plot_s_hi[_i]]],
							yerr=[[_plot_c_lo[_i]], [_plot_c_hi[_i]]],
							fmt='o', color=_color, ecolor=_color,
							elinewidth=1.2, capsize=3, markersize=7, zorder=3
						)
						"""ax_sum.annotate(
							_name,
							xy=(_plot_sep[_i], _plot_contrast[_i]),
							xytext=(4, 4), textcoords='offset points',
							fontsize=7, color=_color, zorder=4
						)"""

					ax_sum.set_yscale('log')
					ax_sum.set_xlabel('Median Angular Separation [mas]', fontsize=13)
					_contrast_label = 'Total (thermal + reflected)' if RUN_REFLECTED_LIGHT else 'Thermal'
					ax_sum.set_ylabel(f'Median Contrast $F_p/F_*$ ({_contrast_label})', fontsize=13)
					ax_sum.set_title(
						f'Detectable planets  (detection prob > {SUMMARY_CONTRAST_DET_THRESHOLD}%)\n'
						f'Filter: {FILTER_USED}   Model: {Madys_Modell_selection}',
						fontsize=12
					)
					ax_sum.grid(alpha=0.3)
					if _have_curves:
						ax_sum.legend(fontsize=9, loc='upper right')

					_sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
					_sm.set_array([])
					_cbar = fig_sum.colorbar(_sm, ax=ax_sum, pad=0.02)
					_cbar.set_label('Stellar Age [Gyr]', fontsize=11)

					#ax_sum.set_xlim(0, 2500)
					ax_sum.set_xscale('log')

					_sum_figname = os.path.join(
						route,
						f'LOCATIS_summary_contrast_sep_{_safe_filter_sum}_{_safe_model_sum}.pdf'
					)
					fig_sum.savefig(_sum_figname, bbox_inches='tight')
					fig_sum.savefig(_sum_figname.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
					plt.close(fig_sum)
					print(f'[Summary plot] Saved: {_sum_figname}')

	# =========================================================================
	# SUMMARY FLAG-CONTRAST PLOT
	# Bar chart: planet name on x-axis, total (or thermal) contrast on y-axis.
	# Colour encodes detection flags:
	#   yellow  – ima_flag == 1  (direct imaging, takes priority)
	#   blue    – tran_flag == 1 (transit, next priority)
	#   violet  – ast_flag == 1 AND rv_flag == 1, all others 0
	#   green   – rv_flag == 1 only (all others 0)
	#   red     – dkin_flag == 1 only (disk perturbation)
	#   black   – none of the above
	# Planets sorted by contrast descending (brightest left).
	# Same det_prob threshold and CSV path as the contrast-sep plot.
	# =========================================================================
	if PLOT_SUMMARY_FLAG_CONTRAST:
		import csv as _csv_flag
		import datetime as _datetime

		_safe_filter_flag = FILTER_USED.replace(' ', '_')
		_safe_model_flag  = Madys_Modell_selection.replace(' ', '_')
		_run_date_flag    = _datetime.date.today().strftime('%Y-%m-%d')
		_default_csv_flag = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_flag}_{_safe_model_flag}_{_run_date_flag}.csv')
		_flag_csv_path    = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv_flag

		if not os.path.exists(_flag_csv_path):
			print(f'[Flag-contrast plot] CSV not found: {_flag_csv_path} – skipping plot.')
		else:
			print(f'[Flag-contrast plot] Reading {_flag_csv_path} …')

			# Decide which contrast column to use
			_fc_median_col = 'median_contrast_total'   if RUN_REFLECTED_LIGHT else 'median_contrast_thermal'
			_fc_p16_col    = 'p16_contrast_total'      if RUN_REFLECTED_LIGHT else 'p16_contrast_thermal'
			_fc_p84_col    = 'p84_contrast_total'      if RUN_REFLECTED_LIGHT else 'p84_contrast_thermal'

			_fc_names    = []
			_fc_contrast = []
			_fc_c_lo     = []
			_fc_c_hi     = []
			_fc_colors   = []

			with open(_flag_csv_path, newline='') as _ff:
				_freader = _csv_flag.DictReader(_ff)
				for _frow in _freader:
					# Detection probability filter
					try:
						_fdet = float(_frow.get('det_prob_total_%', 'nan'))
					except ValueError:
						_fdet = float('nan')
					if np.isnan(_fdet) or _fdet < SUMMARY_CONTRAST_DET_THRESHOLD:
						continue

					# Contrast values
					try:
						_fc_median = float(_frow.get(_fc_median_col, 'nan'))
						_fc_p16    = float(_frow.get(_fc_p16_col,    'nan'))
						_fc_p84    = float(_frow.get(_fc_p84_col,    'nan'))
					except ValueError:
						continue
					if np.isnan(_fc_median):
						continue

					# Flag values (treat missing/N/A as 0)
					def _fl(col):
						v = _frow.get(col, '0').strip()
						try: return int(float(v))
						except ValueError: return 0

					_ima   = _fl('ima_flag')
					_tran  = _fl('tran_flag')
					_rv    = _fl('rv_flag')
					_ast   = _fl('ast_flag')
					_pul   = _fl('pul_flag')
					_ptv   = _fl('ptv_flag')
					_obm   = _fl('obm_flag')
					_micro = _fl('micro_flag')
					_etv   = _fl('etv_flag')
					_dkin  = _fl('dkin_flag')

					# Colour priority rules
					if _ima == 1:
						_fcolor = 'gold'
					elif _tran == 1:
						_fcolor = 'royalblue'
					elif _rv == 1 and _ast == 1 and _pul == 0 and _ptv == 0 and _tran == 0 and _obm == 0 and _micro == 0 and _etv == 0 and _ima == 0 and _dkin == 0:
						_fcolor = 'mediumpurple'
					elif _rv == 1 and _pul == 0 and _ptv == 0 and _tran == 0 and _ast == 0 and _obm == 0 and _micro == 0 and _etv == 0 and _ima == 0 and _dkin == 0:
						_fcolor = 'mediumseagreen'
					elif _dkin == 1 and _rv == 0 and _pul == 0 and _ptv == 0 and _tran == 0 and _ast == 0 and _obm == 0 and _micro == 0 and _etv == 0 and _ima == 0:
						_fcolor = 'tomato'
					else:
						_fcolor = 'black'

					_fc_names.append(_frow.get('planet_name', '?'))
					_fc_contrast.append(_fc_median)
					_fc_c_lo.append(max(0.0, _fc_median - _fc_p16) if not np.isnan(_fc_p16) else 0.0)
					_fc_c_hi.append(max(0.0, _fc_p84 - _fc_median) if not np.isnan(_fc_p84) else 0.0)
					_fc_colors.append(_fcolor)

			if len(_fc_names) == 0:
				print(f'[Flag-contrast plot] No planets above {SUMMARY_CONTRAST_DET_THRESHOLD}% threshold – skipping plot.')
			else:
				# Sort ascending so highest contrast ends up at the top of the horizontal plot
				_fc_order = np.argsort(_fc_contrast)
				_fc_names    = [_fc_names[i]    for i in _fc_order]
				_fc_contrast = [_fc_contrast[i] for i in _fc_order]
				_fc_c_lo     = [_fc_c_lo[i]     for i in _fc_order]
				_fc_c_hi     = [_fc_c_hi[i]     for i in _fc_order]
				_fc_colors   = [_fc_colors[i]   for i in _fc_order]

				_n_fc = len(_fc_names)
				_fig_fc_h = max(6, _n_fc * 0.4)
				fig_fc, ax_fc = plt.subplots(figsize=(14, _fig_fc_h))

				for _fi in range(_n_fc):
					ax_fc.errorbar(
						_fc_contrast[_fi], _fi,
						xerr=[[_fc_c_lo[_fi]], [_fc_c_hi[_fi]]],
						fmt='o', color=_fc_colors[_fi],
						ecolor=_fc_colors[_fi], elinewidth=1.5,
						capsize=4, markersize=8, zorder=3
					)

				ax_fc.set_xscale('log')
				ax_fc.set_yticks(range(_n_fc))
				ax_fc.set_yticklabels(_fc_names, fontsize=16)
				ax_fc.set_ylim(-0.8, _n_fc - 0.2)
				_fc_contrast_label = 'Total (thermal + reflected)' if RUN_REFLECTED_LIGHT else 'Thermal'
				ax_fc.set_xlabel(f'Mean Contrast $F_p/F_*$ ({_fc_contrast_label})', fontsize=18)
				ax_fc.tick_params(axis='x', labelsize=18)
				ax_fc.set_title(
					f'Detectable planets  (det_prob_total > {SUMMARY_CONTRAST_DET_THRESHOLD}%) — sorted by contrast\n'
					f'Filter: {FILTER_USED}   Model: {Madys_Modell_selection}',
					fontsize=11
				)
				ax_fc.grid(axis='x', alpha=0.3)

				# Legend
				_leg_handles = [
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',           markersize=12, label='Direct imaging (ima)'),
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',      markersize=12, label='Transit (tran)'),
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumpurple',   markersize=12, label='RV + Astrometry'),
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumseagreen', markersize=12, label='RV only'),
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',         markersize=12, label='Disk perturbation only (dkin)'),
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',          markersize=12, label='Other / unclassified'),
				]
				ax_fc.legend(handles=_leg_handles, fontsize=12, loc='upper left')

				plt.tight_layout()
				_fc_figname = os.path.join(
					route,
					f'LOCATIS_summary_flag_contrast_{_safe_filter_flag}_{_safe_model_flag}_{_run_date_flag}.pdf'
				)
				fig_fc.savefig(_fc_figname, bbox_inches='tight')
				fig_fc.savefig(_fc_figname.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
				plt.close(fig_fc)
				print(f'[Flag-contrast plot] Saved: {_fc_figname}')


	# =========================================================================
	# SUMMARY DETPROB-NAMES PLOT
	# Horizontal dot plot: planet name on y-axis, det_prob_total_% on x-axis.
	# Colour encodes detection flags (same rules as PLOT_SUMMARY_FLAG_CONTRAST).
	# Planets sorted by det_prob descending (highest at top).
	# Same det_prob threshold and CSV path as the contrast-sep plot.
	# =========================================================================
	if PLOT_SUMMARY_DETPROB_NAMES:
		import csv as _csv_dp
		import datetime as _datetime

		_safe_filter_dp = FILTER_USED.replace(' ', '_')
		_safe_model_dp  = Madys_Modell_selection.replace(' ', '_')
		_run_date_dp    = _datetime.date.today().strftime('%Y-%m-%d')
		_default_csv_dp = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_dp}_{_safe_model_dp}_{_run_date_dp}.csv')
		_dp_csv_path    = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv_dp

		if not os.path.exists(_dp_csv_path):
			print(f'[DetProb-names plot] CSV not found: {_dp_csv_path} – skipping plot.')
		else:
			print(f'[DetProb-names plot] Reading {_dp_csv_path} …')

			_dp_names    = []
			_dp_detprob  = []
			_dp_colors   = []

			with open(_dp_csv_path, newline='') as _fdp:
				_dpreader = _csv_dp.DictReader(_fdp)
				for _dprow in _dpreader:
					# Detection probability filter
					try:
						_dpdet = float(_dprow.get('det_prob_total_%', 'nan'))
					except ValueError:
						_dpdet = float('nan')
					if np.isnan(_dpdet) or _dpdet < SUMMARY_CONTRAST_DET_THRESHOLD:
						continue

					# Flag values (treat missing/N/A as 0)
					def _dp_fl(col):
						v = _dprow.get(col, '0').strip()
						try: return int(float(v))
						except ValueError: return 0

					_dp_ima   = _dp_fl('ima_flag')
					_dp_tran  = _dp_fl('tran_flag')
					_dp_rv    = _dp_fl('rv_flag')
					_dp_ast   = _dp_fl('ast_flag')
					_dp_pul   = _dp_fl('pul_flag')
					_dp_ptv   = _dp_fl('ptv_flag')
					_dp_obm   = _dp_fl('obm_flag')
					_dp_micro = _dp_fl('micro_flag')
					_dp_etv   = _dp_fl('etv_flag')
					_dp_dkin  = _dp_fl('dkin_flag')

					# Colour priority rules (same as flag-contrast plot)
					if _dp_ima == 1:
						_dpcolor = 'gold'
					elif _dp_tran == 1:
						_dpcolor = 'royalblue'
					elif _dp_rv == 1 and _dp_ast == 1 and _dp_pul == 0 and _dp_ptv == 0 and _dp_tran == 0 and _dp_obm == 0 and _dp_micro == 0 and _dp_etv == 0 and _dp_ima == 0 and _dp_dkin == 0:
						_dpcolor = 'mediumpurple'
					elif _dp_rv == 1 and _dp_pul == 0 and _dp_ptv == 0 and _dp_tran == 0 and _dp_ast == 0 and _dp_obm == 0 and _dp_micro == 0 and _dp_etv == 0 and _dp_ima == 0 and _dp_dkin == 0:
						_dpcolor = 'mediumseagreen'
					elif _dp_dkin == 1 and _dp_rv == 0 and _dp_pul == 0 and _dp_ptv == 0 and _dp_tran == 0 and _dp_ast == 0 and _dp_obm == 0 and _dp_micro == 0 and _dp_etv == 0 and _dp_ima == 0:
						_dpcolor = 'tomato'
					else:
						_dpcolor = 'black'

					_dp_names.append(_dprow.get('planet_name', '?'))
					_dp_detprob.append(_dpdet)
					_dp_colors.append(_dpcolor)

			if len(_dp_names) == 0:
				print(f'[DetProb-names plot] No planets above {SUMMARY_CONTRAST_DET_THRESHOLD}% threshold – skipping plot.')
			else:
				# Sort ascending so highest det_prob ends up at the top of the horizontal plot
				_dp_order   = np.argsort(_dp_detprob)
				_dp_names   = [_dp_names[i]   for i in _dp_order]
				_dp_detprob = [_dp_detprob[i] for i in _dp_order]
				_dp_colors  = [_dp_colors[i]  for i in _dp_order]

				_n_dp = len(_dp_names)
				_fig_dp_h = max(6, _n_dp * 0.4)
				fig_dp, ax_dp = plt.subplots(figsize=(14, _fig_dp_h))

				for _dpi in range(_n_dp):
					ax_dp.plot(
						_dp_detprob[_dpi], _dpi,
						'o', color=_dp_colors[_dpi],
						markersize=8, zorder=3
					)

				ax_dp.axvline(SUMMARY_CONTRAST_DET_THRESHOLD, color='grey', linestyle='--', linewidth=1, label=f'Threshold: {SUMMARY_CONTRAST_DET_THRESHOLD}%')
				ax_dp.set_xlim(0, 105)
				ax_dp.set_yticks(range(_n_dp))
				ax_dp.set_yticklabels(_dp_names, fontsize=16)
				ax_dp.set_ylim(-0.8, _n_dp - 0.2)
				ax_dp.set_xlabel('Detection Probability (%)', fontsize=18)
				ax_dp.tick_params(axis='x', labelsize=18)
				ax_dp.set_title(
					f'Detectable planets  (det_prob_total > {SUMMARY_CONTRAST_DET_THRESHOLD}%) — sorted by detection probability\n'
					f'Filter: {FILTER_USED}   Model: {Madys_Modell_selection}',
					fontsize=11
				)
				ax_dp.grid(axis='x', alpha=0.3)

				# Legend
				_leg_dp_handles = [
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',           markersize=12, label='Direct imaging (ima)'),
					#plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',      markersize=12, label='Transit (tran)'),
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumpurple',   markersize=12, label='RV + Astrometry'),
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumseagreen', markersize=12, label='RV only'),
					plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',         markersize=12, label='Disk perturbation only (dkin)'),
					#plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',          markersize=12, label='Other / unclassified'),
				]
				ax_dp.legend(handles=_leg_dp_handles, fontsize=12, loc='upper left')

				plt.tight_layout()
				_dp_figname = os.path.join(
					route,
					f'LOCATIS_summary_detprob_names_{_safe_filter_dp}_{_safe_model_dp}_{_run_date_dp}.pdf'
				)
				fig_dp.savefig(_dp_figname, bbox_inches='tight')
				fig_dp.savefig(_dp_figname.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
				plt.close(fig_dp)
				print(f'[DetProb-names plot] Saved: {_dp_figname}')


	# =========================================================================
	# SUMMARY: DISTANCE vs SEMI-MAJOR AXIS
	# =========================================================================
	if PLOT_SUMMARY_DIST_VS_SMA:
		import csv as _csv_dsma
		import datetime as _datetime

		_safe_filter_dsma = FILTER_USED.replace(' ', '_')
		_safe_model_dsma  = Madys_Modell_selection.replace(' ', '_')
		_run_date_dsma    = _datetime.date.today().strftime('%Y-%m-%d')
		_default_csv_dsma = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_dsma}_{_safe_model_dsma}_{_run_date_dsma}.csv')
		_dsma_csv_path    = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv_dsma

		if not os.path.exists(_dsma_csv_path):
			print(f'[Dist vs SMA plot] CSV not found: {_dsma_csv_path} – skipping plot.')
		else:
			print(f'[Dist vs SMA plot] Reading {_dsma_csv_path} …')
			_dsma_dist = []; _dsma_sma = []; _dsma_names = []; _dsma_colors = []
			_dsma_d_lo = []; _dsma_d_hi = []; _dsma_s_lo = []; _dsma_s_hi = []
			with open(_dsma_csv_path, newline='') as _fdsma:
				_dsma_reader = _csv_dsma.DictReader(_fdsma)
				for _dsma_row in _dsma_reader:
					try:
						_dsma_det = float(_dsma_row.get('det_prob_total_%', 'nan'))
					except ValueError:
						_dsma_det = float('nan')
					if np.isnan(_dsma_det) or _dsma_det < SUMMARY_CONTRAST_DET_THRESHOLD:
						continue
					try:
						_d = float(_dsma_row.get('sy_dist_pc', 'nan'))
						_s = float(_dsma_row.get('pl_orbsmax_au', 'nan'))
					except ValueError:
						continue
					if np.isnan(_d) or np.isnan(_s):
						continue
					# Distance error bars (err1=upper, err2=lower; both stored as signed in archive)
					try: _d1 = abs(float(_dsma_row.get('sy_disterr1_pc', 'nan')))
					except ValueError: _d1 = 0.0
					if np.isnan(_d1): _d1 = 0.0
					try: _d2 = abs(float(_dsma_row.get('sy_disterr2_pc', 'nan')))
					except ValueError: _d2 = 0.0
					if np.isnan(_d2): _d2 = 0.0
					# SMA error bars
					try: _s1 = abs(float(_dsma_row.get('pl_orbsmaxerr1_au', 'nan')))
					except ValueError: _s1 = 0.0
					if np.isnan(_s1): _s1 = 0.0
					try: _s2 = abs(float(_dsma_row.get('pl_orbsmaxerr2_au', 'nan')))
					except ValueError: _s2 = 0.0
					if np.isnan(_s2): _s2 = 0.0
					_dsma_dist.append(_d)
					_dsma_sma.append(_s)
					_dsma_names.append(_dsma_row.get('planet_name', '?'))
					_dsma_colors.append(_dsma_det)
					_dsma_d_lo.append(_d2)  # lower uncertainty (|err2|)
					_dsma_d_hi.append(_d1)  # upper uncertainty (err1)
					_dsma_s_lo.append(_s2)
					_dsma_s_hi.append(_s1)
				if len(_dsma_names) == 0:
					print(f'[Dist vs SMA plot] No planets above threshold – skipping plot.')
				else:
					fig_dsma, ax_dsma = plt.subplots(figsize=(9, 7))
					_sc_dsma = ax_dsma.scatter(_dsma_dist, _dsma_sma, c=_dsma_colors,
						                          cmap='viridis', s=60, zorder=3)
					for _i in range(len(_dsma_names)):
						ax_dsma.errorbar(
							_dsma_dist[_i], _dsma_sma[_i],
							xerr=[[_dsma_d_lo[_i]], [_dsma_d_hi[_i]]],
							yerr=[[_dsma_s_lo[_i]], [_dsma_s_hi[_i]]],
							fmt='none', ecolor='grey', elinewidth=1, capsize=3, zorder=2
						)
					plt.colorbar(_sc_dsma, ax=ax_dsma, label='Detection Probability (%)') 
					ax_dsma.set_yscale('log')
					ax_dsma.set_xlabel('Distance [pc]', fontsize=12)
					ax_dsma.set_ylabel('Semi-major Axis [AU] (log)', fontsize=12)
					ax_dsma.tick_params(axis='both', labelsize=18)
					ax_dsma.set_title(
						f'Distance vs Semi-major Axis\nFilter: {FILTER_USED}   Model: {Madys_Modell_selection}',
						fontsize=11
					)
					ax_dsma.grid(alpha=0.3)
					plt.tight_layout()
					_dsma_figname = os.path.join(route, f'LOCATIS_summary_dist_vs_sma_{_safe_filter_dsma}_{_safe_model_dsma}_{_run_date_dsma}.pdf')
					fig_dsma.savefig(_dsma_figname, bbox_inches='tight')
					fig_dsma.savefig(_dsma_figname.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
					plt.close(fig_dsma)
					print(f'[Dist vs SMA plot] Saved: {_dsma_figname}')


	# =========================================================================
	# SUMMARY: ABSOLUTE MAGNITUDE vs AGE
	# =========================================================================
	if PLOT_SUMMARY_ABSMAG_VS_AGE:
		import csv as _csv_ama
		import datetime as _datetime

		_safe_filter_ama = FILTER_USED.replace(' ', '_')
		_safe_model_ama  = Madys_Modell_selection.replace(' ', '_')
		_run_date_ama    = _datetime.date.today().strftime('%Y-%m-%d')
		_default_csv_ama = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_ama}_{_safe_model_ama}_{_run_date_ama}.csv')
		_ama_csv_path    = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv_ama

		if not os.path.exists(_ama_csv_path):
			print(f'[AbsMag vs Age plot] CSV not found: {_ama_csv_path} – skipping plot.')
		else:
			print(f'[AbsMag vs Age plot] Reading {_ama_csv_path} …')
			_ama_age = []; _ama_mag = []; _ama_mag_lo = []; _ama_mag_hi = []
			_ama_age_lo = []; _ama_age_hi = []
			_ama_names = []; _ama_colors = []
			with open(_ama_csv_path, newline='') as _fama:
				_ama_reader = _csv_ama.DictReader(_fama)
				for _ama_row in _ama_reader:
					try:
						_ama_det = float(_ama_row.get('det_prob_total_%', 'nan'))
					except ValueError:
						_ama_det = float('nan')
					if np.isnan(_ama_det) or _ama_det < SUMMARY_CONTRAST_DET_THRESHOLD:
						continue
					try:
						_ag  = float(_ama_row.get('st_age_Gyr', 'nan'))
						_am  = float(_ama_row.get('median_abs_mag', 'nan'))
						_am16 = float(_ama_row.get('p16_abs_mag', 'nan'))
						_am84 = float(_ama_row.get('p84_abs_mag', 'nan'))
					except ValueError:
						continue
					if np.isnan(_ag) or np.isnan(_am):
						continue
					# Age error bars (err1 = upper, err2 = lower in NASA archive)
					try: _ag_hi = abs(float(_ama_row.get('st_ageerr1_Gyr', 'nan')))
					except ValueError: _ag_hi = 0.0
					if np.isnan(_ag_hi): _ag_hi = 0.0
					try: _ag_lo = abs(float(_ama_row.get('st_ageerr2_Gyr', 'nan')))
					except ValueError: _ag_lo = 0.0
					if np.isnan(_ag_lo): _ag_lo = 0.0
					_ama_age.append(_ag)
					_ama_mag.append(_am)
					_ama_mag_lo.append(max(0.0, _am - _am16) if not np.isnan(_am16) else 0.0)
					_ama_mag_hi.append(max(0.0, _am84 - _am) if not np.isnan(_am84) else 0.0)
					_ama_age_lo.append(_ag_lo)
					_ama_age_hi.append(_ag_hi)
					_ama_names.append(_ama_row.get('planet_name', '?'))
					_ama_colors.append(_ama_det)
				if len(_ama_names) == 0:
					print(f'[AbsMag vs Age plot] No planets above threshold – skipping plot.')
				else:
					fig_ama, ax_ama = plt.subplots(figsize=(9, 7))
					_sc_ama = ax_ama.scatter(_ama_age, _ama_mag, c=_ama_colors,
						                        cmap='viridis', s=60, zorder=3)
					for _i in range(len(_ama_names)):
						ax_ama.errorbar(_ama_age[_i], _ama_mag[_i],
							            xerr=[[_ama_age_lo[_i]], [_ama_age_hi[_i]]],
							            yerr=[[_ama_mag_lo[_i]], [_ama_mag_hi[_i]]],
							            fmt='none', ecolor='grey', elinewidth=1, capsize=3, zorder=2)
					plt.colorbar(_sc_ama, ax=ax_ama, label='Detection Probability (%)')
					ax_ama.invert_yaxis()  # brighter = lower mag number = top of plot
					ax_ama.set_xscale('log')
					ax_ama.set_xlabel('Stellar Age [Gyr] (log)', fontsize=12)
					ax_ama.set_ylabel('Absolute Magnitude', fontsize=12)
					ax_ama.tick_params(axis='both', labelsize=18)
					ax_ama.set_title(
						f'Absolute Magnitude vs Age\nFilter: {FILTER_USED}   Model: {Madys_Modell_selection}',
						fontsize=11
					)
					ax_ama.grid(alpha=0.3)
					plt.tight_layout()
					_ama_figname = os.path.join(route, f'LOCATIS_summary_absmag_vs_age_{_safe_filter_ama}_{_safe_model_ama}_{_run_date_ama}.pdf')
					fig_ama.savefig(_ama_figname, bbox_inches='tight')
					fig_ama.savefig(_ama_figname.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
					plt.close(fig_ama)
					print(f'[AbsMag vs Age plot] Saved: {_ama_figname}')


	# =========================================================================
	# SUMMARY: EFFECTIVE TEMPERATURE vs AGE
	# =========================================================================
	if PLOT_SUMMARY_TEMP_VS_AGE:
		import csv as _csv_tva
		import datetime as _datetime

		_safe_filter_tva = FILTER_USED.replace(' ', '_')
		_safe_model_tva  = Madys_Modell_selection.replace(' ', '_')
		_run_date_tva    = _datetime.date.today().strftime('%Y-%m-%d')
		_default_csv_tva = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_tva}_{_safe_model_tva}_{_run_date_tva}.csv')
		_tva_csv_path    = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv_tva

		if not os.path.exists(_tva_csv_path):
			print(f'[Temp vs Age plot] CSV not found: {_tva_csv_path} – skipping plot.')
		else:
			print(f'[Temp vs Age plot] Reading {_tva_csv_path} …')
			_tva_age = []; _tva_temp = []; _tva_t_lo = []; _tva_t_hi = []
			_tva_age_lo = []; _tva_age_hi = []
			_tva_names = []; _tva_colors = []
			with open(_tva_csv_path, newline='') as _ftva:
				_tva_reader = _csv_tva.DictReader(_ftva)
				for _tva_row in _tva_reader:
					try:
						_tva_det = float(_tva_row.get('det_prob_total_%', 'nan'))
					except ValueError:
						_tva_det = float('nan')
					if np.isnan(_tva_det) or _tva_det < SUMMARY_CONTRAST_DET_THRESHOLD:
						continue
					try:
						_ta_ag   = float(_tva_row.get('st_age_Gyr', 'nan'))
						_ta_t    = float(_tva_row.get('median_planet_temp_K', 'nan'))
						_ta_t16  = float(_tva_row.get('p16_planet_temp_K', 'nan'))
						_ta_t84  = float(_tva_row.get('p84_planet_temp_K', 'nan'))
					except ValueError:
						continue
					if np.isnan(_ta_ag) or np.isnan(_ta_t):
						continue
					# Age error bars (err1 = upper, err2 = lower in NASA archive)
					try: _ta_ag_hi = abs(float(_tva_row.get('st_ageerr1_Gyr', 'nan')))
					except ValueError: _ta_ag_hi = 0.0
					if np.isnan(_ta_ag_hi): _ta_ag_hi = 0.0
					try: _ta_ag_lo = abs(float(_tva_row.get('st_ageerr2_Gyr', 'nan')))
					except ValueError: _ta_ag_lo = 0.0
					if np.isnan(_ta_ag_lo): _ta_ag_lo = 0.0
					_tva_age.append(_ta_ag)
					_tva_temp.append(_ta_t)
					_tva_t_lo.append(max(0.0, _ta_t - _ta_t16) if not np.isnan(_ta_t16) else 0.0)
					_tva_t_hi.append(max(0.0, _ta_t84 - _ta_t) if not np.isnan(_ta_t84) else 0.0)
					_tva_age_lo.append(_ta_ag_lo)
					_tva_age_hi.append(_ta_ag_hi)
					_tva_names.append(_tva_row.get('planet_name', '?'))
					_tva_colors.append(_tva_det)
				if len(_tva_names) == 0:
					print(f'[Temp vs Age plot] No planets above threshold – skipping plot.')
				else:
					fig_tva, ax_tva = plt.subplots(figsize=(9, 7))
					_sc_tva = ax_tva.scatter(_tva_age, _tva_temp, c=_tva_colors,
						                        cmap='viridis', s=60, zorder=3)
					for _i in range(len(_tva_names)):
						ax_tva.errorbar(_tva_age[_i], _tva_temp[_i],
							            xerr=[[_tva_age_lo[_i]], [_tva_age_hi[_i]]],
							            yerr=[[_tva_t_lo[_i]], [_tva_t_hi[_i]]],
							            fmt='none', ecolor='grey', elinewidth=1, capsize=3, zorder=2)
					plt.colorbar(_sc_tva, ax=ax_tva, label='Detection Probability (%)')
					ax_tva.set_xscale('log')
					ax_tva.set_xlabel('Stellar Age [Gyr] (log)', fontsize=12)
					ax_tva.set_ylabel('Effective Temperature [K]', fontsize=12)
					ax_tva.tick_params(axis='both', labelsize=18)
					ax_tva.set_title(
						f'Effective Temperature vs Age\nFilter: {FILTER_USED}   Model: {Madys_Modell_selection}',
						fontsize=11
					)
					ax_tva.grid(alpha=0.3)
					plt.tight_layout()
					_tva_figname = os.path.join(route, f'LOCATIS_summary_temp_vs_age_{_safe_filter_tva}_{_safe_model_tva}_{_run_date_tva}.pdf')
					fig_tva.savefig(_tva_figname, bbox_inches='tight')
					fig_tva.savefig(_tva_figname.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
					plt.close(fig_tva)
					print(f'[Temp vs Age plot] Saved: {_tva_figname}')


	# =========================================================================
	# SUMMARY: EFFECTIVE TEMPERATURE vs MASS
	# =========================================================================
	if PLOT_SUMMARY_TEMP_VS_MASS:
		import csv as _csv_tvm
		import datetime as _datetime

		_safe_filter_tvm = FILTER_USED.replace(' ', '_')
		_safe_model_tvm  = Madys_Modell_selection.replace(' ', '_')
		_run_date_tvm    = _datetime.date.today().strftime('%Y-%m-%d')
		_default_csv_tvm = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_tvm}_{_safe_model_tvm}_{_run_date_tvm}.csv')
		_tvm_csv_path    = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv_tvm

		if not os.path.exists(_tvm_csv_path):
			print(f'[Temp vs Mass plot] CSV not found: {_tvm_csv_path} – skipping plot.')
		else:
			print(f'[Temp vs Mass plot] Reading {_tvm_csv_path} …')
			_tvm_mass = []; _tvm_temp = []; _tvm_t_lo = []; _tvm_t_hi = []
			_tvm_m_lo = []; _tvm_m_hi = []
			_tvm_names = []; _tvm_colors = []
			with open(_tvm_csv_path, newline='') as _ftvm:
				_tvm_reader = _csv_tvm.DictReader(_ftvm)
				for _tvm_row in _tvm_reader:
					try:
						_tvm_det = float(_tvm_row.get('det_prob_total_%', 'nan'))
					except ValueError:
						_tvm_det = float('nan')
					if np.isnan(_tvm_det) or _tvm_det < SUMMARY_CONTRAST_DET_THRESHOLD:
						continue
					try:
						_tv_m    = float(_tvm_row.get('median_mass_mjup', 'nan'))
						_tv_m16  = float(_tvm_row.get('p16_mass_mjup', 'nan'))
						_tv_m84  = float(_tvm_row.get('p84_mass_mjup', 'nan'))
						_tv_t    = float(_tvm_row.get('median_planet_temp_K', 'nan'))
						_tv_t16  = float(_tvm_row.get('p16_planet_temp_K', 'nan'))
						_tv_t84  = float(_tvm_row.get('p84_planet_temp_K', 'nan'))
					except ValueError:
						continue
					if np.isnan(_tv_m) or np.isnan(_tv_t):
						continue
					# Color by stellar age
					try:
						_tv_age = float(_tvm_row.get('st_age_Gyr', 'nan'))
					except ValueError:
						_tv_age = float('nan')
					_tvm_mass.append(_tv_m)
					_tvm_temp.append(_tv_t)
					_tvm_m_lo.append(max(0.0, _tv_m - _tv_m16) if not np.isnan(_tv_m16) else 0.0)
					_tvm_m_hi.append(max(0.0, _tv_m84 - _tv_m) if not np.isnan(_tv_m84) else 0.0)
					_tvm_t_lo.append(max(0.0, _tv_t - _tv_t16) if not np.isnan(_tv_t16) else 0.0)
					_tvm_t_hi.append(max(0.0, _tv_t84 - _tv_t) if not np.isnan(_tv_t84) else 0.0)
					_tvm_names.append(_tvm_row.get('planet_name', '?'))
					_tvm_colors.append(_tv_age)
			if len(_tvm_names) == 0:
				print(f'[Temp vs Mass plot] No planets above threshold – skipping plot.')
			else:
					fig_tvm, ax_tvm = plt.subplots(figsize=(9, 7))
					_sc_tvm = ax_tvm.scatter(_tvm_mass, _tvm_temp, c=_tvm_colors,
						                        cmap='plasma', s=60, zorder=3)
					for _i in range(len(_tvm_names)):
						ax_tvm.errorbar(_tvm_mass[_i], _tvm_temp[_i],
							            xerr=[[_tvm_m_lo[_i]], [_tvm_m_hi[_i]]],
							            yerr=[[_tvm_t_lo[_i]], [_tvm_t_hi[_i]]],
							            fmt='none', ecolor='grey', elinewidth=1, capsize=3, zorder=2)
					plt.colorbar(_sc_tvm, ax=ax_tvm, label='Stellar Age [Gyr]')
					ax_tvm.set_xscale('log')
					ax_tvm.set_xlabel('Planet Mass [$M_{Jup}$] (log)', fontsize=12)
					ax_tvm.set_ylabel('Effective Temperature [K]', fontsize=12)
					ax_tvm.tick_params(axis='both', labelsize=18)
					ax_tvm.set_title(
						f'Effective Temperature vs Mass\nFilter: {FILTER_USED}   Model: {Madys_Modell_selection}',
						fontsize=11
					)
					ax_tvm.grid(alpha=0.3)
					plt.tight_layout()
					_tvm_figname = os.path.join(route, f'LOCATIS_summary_temp_vs_mass_{_safe_filter_tvm}_{_safe_model_tvm}_{_run_date_tvm}.pdf')
					fig_tvm.savefig(_tvm_figname, bbox_inches='tight')
					fig_tvm.savefig(_tvm_figname.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
					plt.close(fig_tvm)
					print(f'[Temp vs Mass plot] Saved: {_tvm_figname}')


	# =========================================================================
	# SUMMARY: SEMI-MAJOR AXIS vs. DISTANCE  (a vs d diagram)
	# All confirmed NASA archive planets shown as hollow markers (background),
	# coloured by discovery method. Detectable planets (det_prob > threshold)
	# shown as filled markers on top, same colour scheme.
	# =========================================================================
	if PLOT_SUMMARY_A_VS_D:
		import csv as _csv_avd
		import datetime as _datetime_avd

		_safe_filter_avd = FILTER_USED.replace(' ', '_')
		_safe_model_avd  = Madys_Modell_selection.replace(' ', '_')
		_run_date_avd    = _datetime_avd.date.today().strftime('%Y-%m-%d')
		_default_csv_avd = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_avd}_{_safe_model_avd}_{_run_date_avd}.csv')
		_avd_csv_path    = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv_avd

		if not os.path.exists(_avd_csv_path):
			print(f'[a vs d plot] CSV not found: {_avd_csv_path}  – skipping plot.')
		else:
			print(f'[a vs d plot] Reading {_avd_csv_path} …')

			# Collect names of detectable planets from CSV
			_avd_det_names = set()
			with open(_avd_csv_path, newline='') as _f_avd:
				_reader_avd = _csv_avd.DictReader(_f_avd)
				for _row_avd in _reader_avd:
					try:
						_det_avd = float(_row_avd.get('det_prob_total_%', 'nan'))
					except (ValueError, TypeError):
						_det_avd = float('nan')
					if np.isfinite(_det_avd) and _det_avd >= SUMMARY_CONTRAST_DET_THRESHOLD:
						_avd_det_names.add(_row_avd.get('planet_name', ''))

			_avd_labels  = ['Transit', 'Radial Velocity', 'Imaging', 'Microlensing', 'Others']
			_avd_colours = ['r', 'b', 'magenta', 'lime', 'k']

			_avd_orbsmax_all = [[] for _ in range(len(_avd_labels))]
			_avd_dist_all    = [[] for _ in range(len(_avd_labels))]
			_avd_orbsmax_obs = [[] for _ in range(len(_avd_labels))]
			_avd_dist_obs    = [[] for _ in range(len(_avd_labels))]

			for _avd_key in dictionary:
				_avd_entry  = dictionary[_avd_key]
				_avd_a = float('nan') if _avd_entry.get('pl_orbsmax', '') == '' else float(_avd_entry['pl_orbsmax'])
				_avd_d = float('nan') if _avd_entry.get('sy_dist',    '') == '' else float(_avd_entry['sy_dist'])
				_avd_method = _avd_entry.get('discoverymethod', '')

				if _avd_method == 'Transit':
					_avd_idx = 0
				elif _avd_method == 'Radial Velocity':
					_avd_idx = 1
				elif _avd_method == 'Imaging':
					_avd_idx = 2
				elif _avd_method == 'Microlensing':
					_avd_idx = 3
				else:
					_avd_idx = 4

				_avd_orbsmax_all[_avd_idx].append(_avd_a)
				_avd_dist_all[_avd_idx].append(_avd_d)

				# Check if this planet is detectable
				_avd_pl_name = _avd_entry.get('pl_name', _avd_key)
				if _avd_pl_name in _avd_det_names or _avd_key in _avd_det_names:
					_avd_orbsmax_obs[_avd_idx].append(_avd_a)
					_avd_dist_obs[_avd_idx].append(_avd_d)

			_avd_lettersize = 17
			fig_avd, ax_avd = plt.subplots(figsize=(9, 9))

			# Background: all archive planets (hollow markers)
			for _avd_i in range(len(_avd_labels)):
				ax_avd.scatter(_avd_orbsmax_all[_avd_i], _avd_dist_all[_avd_i],
				               marker='.', facecolors='none', edgecolors=_avd_colours[_avd_i],
				               alpha=0.3, s=20)

			# Foreground: detectable planets (filled markers, with legend)
			for _avd_i in range(len(_avd_labels)):
				ax_avd.scatter(_avd_orbsmax_obs[_avd_i], _avd_dist_obs[_avd_i],
				               marker='o', facecolors=_avd_colours[_avd_i],
				               edgecolors=_avd_colours[_avd_i],
				               label=_avd_labels[_avd_i], s=50, zorder=3)

			ax_avd.legend(fontsize=_avd_lettersize - 4)
			ax_avd.set_yscale('log')
			ax_avd.set_xscale('log')
			ax_avd.set_xlabel('$a$ / AU', fontsize=_avd_lettersize)
			ax_avd.set_ylabel('$d$ / pc', fontsize=_avd_lettersize)
			ax_avd.set_ylim(ymin=1.0)
			ax_avd.grid(alpha=0.3)
			ax_avd.tick_params(axis='both', labelsize=_avd_lettersize - 4)
			ax_avd.set_title(
				f'Semi-major axis vs. distance  (detectable: det_prob > {SUMMARY_CONTRAST_DET_THRESHOLD}%)\n'
				f'Filter: {FILTER_USED}   Model: {Madys_Modell_selection}',
				fontsize=12
			)

			_avd_figname = os.path.join(route, f'LOCATIS_summary_a_vs_d_{_safe_filter_avd}_{_safe_model_avd}.pdf')
			fig_avd.savefig(_avd_figname, bbox_inches='tight')
			fig_avd.savefig(_avd_figname.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
			plt.close(fig_avd)
			print(f'[a vs d plot] Saved: {_avd_figname}')


	# =========================================================================
	# OUTPUT TABLE (LOCATIS RESULTS CSV)
	# Reads the summary CSV, filters to det_prob > SUMMARY_CONTRAST_DET_THRESHOLD,
	# writes a compact results table with median / p16 / p84 for planet temp,
	# angular separation, Fp/Fstar and detection probability.
	# If COMBINE_OUTPUT_TABLES_PATH is set, also writes a merged L+M table
	# sorted by the sum of both bands' detection probabilities.
	# =========================================================================
	if WRITE_OUTPUT_TABLE_CSV:
		import csv as _csv_out
		import datetime as _datetime_out
		_safe_filter_out = FILTER_USED.replace(' ', '_')
		_safe_model_out  = Madys_Modell_selection.replace(' ', '_')
		_run_date_out    = _datetime_out.date.today().strftime('%Y-%m-%d')
		_default_csv_out = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_out}_{_safe_model_out}_{_run_date_out}.csv')
		_out_src_csv     = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv_out

		if not os.path.exists(_out_src_csv):
			print(f'[Output table] Source CSV not found: {_out_src_csv} – skipping.')
		else:
			_out_rows = []
			with open(_out_src_csv, newline='') as _f_out_src:
				_reader_out = _csv_out.DictReader(_f_out_src)
				for _row_out in _reader_out:
					try:
						_dp_out = float(_row_out.get('det_prob_total_%', 'nan'))
					except (ValueError, TypeError):
						_dp_out = float('nan')
					if np.isfinite(_dp_out) and _dp_out >= SUMMARY_CONTRAST_DET_THRESHOLD:
						_out_rows.append(_row_out)

			if len(_out_rows) == 0:
				print(f'[Output table] No planets above {SUMMARY_CONTRAST_DET_THRESHOLD}% – skipping.')
			else:
				# Sort by det_prob descending (single-band)
				_out_rows.sort(key=lambda r: float(r.get('det_prob_total_%', 0) or 0), reverse=True)

				_out_csv_path = os.path.join(route, f'LOCATIS_output_table_{_safe_filter_out}_{_safe_model_out}_{_run_date_out}.csv')
				_out_header = [
					'planet_name',
					'median_planet_temp_K', 'p16_planet_temp_K', 'p84_planet_temp_K',
					'median_angsep_mas', 'p16_angsep_mas', 'p84_angsep_mas',
					f'median_Fp_Fstar_total_{_safe_filter_out}',
					f'p16_Fp_Fstar_total_{_safe_filter_out}',
					f'p84_Fp_Fstar_total_{_safe_filter_out}',
					f'det_prob_total_%_{_safe_filter_out}',
				]
				with open(_out_csv_path, 'w', newline='') as _f_out:
					_writer_out = _csv_out.writer(_f_out)
					_writer_out.writerow(_out_header)
					for _r in _out_rows:
						_writer_out.writerow([
							_r.get('planet_name', 'N/A'),
							_r.get('median_planet_temp_K', 'N/A'), _r.get('p16_planet_temp_K', 'N/A'), _r.get('p84_planet_temp_K', 'N/A'),
							_r.get('median_angsep_mas', 'N/A'),    _r.get('p16_angsep_mas', 'N/A'),    _r.get('p84_angsep_mas', 'N/A'),
							_r.get('median_contrast_total', 'N/A'), _r.get('p16_contrast_total', 'N/A'), _r.get('p84_contrast_total', 'N/A'),
							_r.get('det_prob_total_%', 'N/A'),
						])
				print(f'[Output table] Saved: {_out_csv_path}')

				# ----------------------------------------------------------
				# MERGE with second-band output table (if path provided)
				# ----------------------------------------------------------
				if COMBINE_OUTPUT_TABLES_PATH is not None and os.path.exists(COMBINE_OUTPUT_TABLES_PATH):
					_other_rows_dict = {}
					_other_header_cols = []
					with open(COMBINE_OUTPUT_TABLES_PATH, newline='') as _f_other:
						_reader_other = _csv_out.DictReader(_f_other)
						_other_header_cols = list(_reader_other.fieldnames or [])
						for _ro in _reader_other:
							_other_rows_dict[_ro.get('planet_name', '')] = _ro

					# Identify the other-band filter tag from its column names
					_other_det_col  = next((c for c in _other_header_cols if c.startswith('det_prob_total_%_')),         None)
					_other_cont_med = next((c for c in _other_header_cols if c.startswith('median_Fp_Fstar_total_')),    None)
					_other_cont_p16 = next((c for c in _other_header_cols if c.startswith('p16_Fp_Fstar_total_')),       None)
					_other_cont_p84 = next((c for c in _other_header_cols if c.startswith('p84_Fp_Fstar_total_')),       None)
					_other_filter_tag = _other_det_col.replace('det_prob_total_%_', '') if _other_det_col else 'other'

					# Union of all planet names across both bands, sorted by combined det_prob
					_cur_dict  = {r.get('planet_name', ''): r for r in _out_rows}
					_all_names = list(dict.fromkeys(list(_cur_dict.keys()) + list(_other_rows_dict.keys())))

					def _comb_dp(name):
						_v1 = float(_cur_dict.get(name, {}).get('det_prob_total_%', 0) or 0)
						_v2 = float(_other_rows_dict.get(name, {}).get(_other_det_col, 0) or 0) if _other_det_col else 0.0
						return _v1 + _v2

					_all_names.sort(key=_comb_dp, reverse=True)

					_comb_header = [
						'planet_name',
						'median_planet_temp_K', 'p16_planet_temp_K', 'p84_planet_temp_K',
						'median_angsep_mas', 'p16_angsep_mas', 'p84_angsep_mas',
						f'median_Fp_Fstar_total_{_safe_filter_out}',
						f'p16_Fp_Fstar_total_{_safe_filter_out}',
						f'p84_Fp_Fstar_total_{_safe_filter_out}',
						f'det_prob_total_%_{_safe_filter_out}',
						f'median_Fp_Fstar_total_{_other_filter_tag}',
						f'p16_Fp_Fstar_total_{_other_filter_tag}',
						f'p84_Fp_Fstar_total_{_other_filter_tag}',
						f'det_prob_total_%_{_other_filter_tag}',
					]
					_comb_csv_path = os.path.join(route, f'LOCATIS_combined_output_table_{_run_date_out}.csv')
					with open(_comb_csv_path, 'w', newline='') as _f_comb:
						_writer_comb = _csv_out.writer(_f_comb)
						_writer_comb.writerow(_comb_header)
						for _pname in _all_names:
							_cr = _cur_dict.get(_pname, {})
							_or = _other_rows_dict.get(_pname, {})
							_writer_comb.writerow([
								_pname,
								_cr.get('median_planet_temp_K', 'N/A'), _cr.get('p16_planet_temp_K', 'N/A'), _cr.get('p84_planet_temp_K', 'N/A'),
								_cr.get('median_angsep_mas', 'N/A'),    _cr.get('p16_angsep_mas', 'N/A'),    _cr.get('p84_angsep_mas', 'N/A'),
								_cr.get('median_contrast_total', 'N/A'), _cr.get('p16_contrast_total', 'N/A'), _cr.get('p84_contrast_total', 'N/A'),
								_cr.get('det_prob_total_%', 'N/A'),
								_or.get(_other_cont_med, 'N/A') if _other_cont_med else 'N/A',
								_or.get(_other_cont_p16, 'N/A') if _other_cont_p16 else 'N/A',
								_or.get(_other_cont_p84, 'N/A') if _other_cont_p84 else 'N/A',
								_or.get(_other_det_col,  'N/A') if _other_det_col  else 'N/A',
							])
					print(f'[Output table] Combined L+M table saved: {_comb_csv_path}')
				elif COMBINE_OUTPUT_TABLES_PATH is not None:
					print(f'[Output table] COMBINE_OUTPUT_TABLES_PATH set but file not found: {COMBINE_OUTPUT_TABLES_PATH}')


	# =========================================================================
	# STELLAR PROPERTIES HISTOGRAM
	# Three-panel figure: spectral type bar chart, age histogram, stellar mass
	# histogram. Black hatched = all confirmed exoplanets; green dotted =
	# detectable planets (det_prob_total_% > SUMMARY_CONTRAST_DET_THRESHOLD).
	# Reads detectable planet names from CSV, looks up stellar data in
	# dictionary. Saves to route/Population_study/.
	# =========================================================================
	if PLOT_STELLAR_PROPERTIES_HISTOGRAM:
		import csv as _csv_sph
		import datetime as _datetime

		_safe_filter_sph = FILTER_USED.replace(' ', '_')
		_safe_model_sph  = Madys_Modell_selection.replace(' ', '_')
		_run_date_sph    = _datetime.date.today().strftime('%Y-%m-%d')
		_default_csv_sph = os.path.join(route, f'LOCATIS_run_summary_{_safe_filter_sph}_{_safe_model_sph}_{_run_date_sph}.csv')
		_sph_csv_path    = SUMMARY_CSV_RERUN_PATH if SUMMARY_CSV_RERUN_PATH is not None else _default_csv_sph

		if not os.path.exists(_sph_csv_path):
			print(f'[Stellar histogram] CSV not found: {_sph_csv_path} – skipping plot.')
		else:
			print(f'[Stellar histogram] Reading {_sph_csv_path} …')

			# ----------------------------------------------------------
			# Build set of detectable planet names from CSV
			# ----------------------------------------------------------
			_sph_det_planets = set()
			with open(_sph_csv_path, newline='') as _fsph:
				_sph_reader = _csv_sph.DictReader(_fsph)
				for _sph_row in _sph_reader:
					try:
						_sph_det = float(_sph_row.get('det_prob_total_%', 'nan'))
					except ValueError:
						_sph_det = float('nan')
					if not (np.isnan(_sph_det) or _sph_det < SUMMARY_CONTRAST_DET_THRESHOLD):
						_sph_det_planets.add(_sph_row.get('planet_name', '').strip())

			if len(_sph_det_planets) == 0:
				print(f'[Stellar histogram] No planets above threshold – skipping plot.')
			else:
				st_types = ['W', 'O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T']
				hist_all_spec = [0] * len(st_types)
				hist_obs_spec = [0] * len(st_types)
				hist_all_age, hist_obs_age   = [], []
				hist_all_mass, hist_obs_mass = [], []

				done_obs, done_all = [], []

				# --- Detectable planets: loop over planet names from CSV ---
				for _pname in _sph_det_planets:
					if _pname not in dictionary:
						continue
					_pd = dictionary[_pname]
					_hn = _pd.get('hostname', '').strip()
					if _hn == '':
						_hn = _pname  # fallback: use planet name to avoid merging unknowns
					if _hn not in done_obs:
						done_obs.append(_hn)
						# Spectral type
						_sp = _pd.get('st_spectype', '').strip()
						if _sp != '':
							for _stt in st_types:
								if _sp[0].upper() == _stt:
									hist_obs_spec[st_types.index(_stt)] += 1
						# Age
						if _pd.get('st_age', '').strip() != '':
							try: hist_obs_age.append(float(_pd['st_age']))
							except ValueError: pass
						# Stellar mass
						if _pd.get('st_mass', '').strip() != '':
							try: hist_obs_mass.append(float(_pd['st_mass']))
							except ValueError: pass

				print(f'[Stellar histogram] Stars hosting detectable exoplanets: {len(done_obs)}')
				print(f'[Stellar histogram] Stars with spectral type info: {sum(hist_obs_spec)}')
				print(f'[Stellar histogram] Stars with age info: {len(hist_obs_age)}')

				# --- All confirmed exoplanets ---
				for _key_all in dictionary:
					_pa = dictionary[_key_all]
					_hn_all = _pa.get('hostname', '').strip()
					if _hn_all == '':
						_hn_all = _key_all
					if _hn_all not in done_all:
						done_all.append(_hn_all)
						# Spectral type
						_sp_all = _pa.get('st_spectype', '').strip()
						if _sp_all != '':
							for _stt in st_types:
								if _sp_all[0].upper() == _stt:
									hist_all_spec[st_types.index(_stt)] += 1
						# Age
						if _pa.get('st_age', '').strip() != '':
							try: hist_all_age.append(float(_pa['st_age']))
							except ValueError: pass
						# Stellar mass
						if _pa.get('st_mass', '').strip() != '':
							try: hist_all_mass.append(float(_pa['st_mass']))
							except ValueError: pass
						# Transit layer (commented out — kept for reference)
						# if _pa.get('tran_flag', '').strip() == '1': ...

				# ----------------------------------------------------------
				# Build figure
				# ----------------------------------------------------------
				_lettersize = 15
				fig_sph, (ax_sp, ax_age, ax_mass) = plt.subplots(3, 1, figsize=(6, 7))

				# --- Spectral type (dual y-axis) ---
				ax_sp_twin = ax_sp.twinx()
				ax_sp.bar(st_types, hist_all_spec, width=1, color='k', alpha=0.3)
				ax_sp.bar(st_types, hist_all_spec, width=1, facecolor='none', hatch='/', edgecolor='k', linewidth=0.01)
				ax_sp_twin.bar(st_types, hist_obs_spec, width=1, color='lightgreen', alpha=0.3)
				ax_sp_twin.bar(st_types, hist_obs_spec, width=1, facecolor='none', hatch='.', edgecolor='lightgreen', linewidth=0.01)
				_ymini, _ymaxi = ax_sp_twin.get_ylim()
				ax_sp_twin.set_ylim(ymax=_ymaxi + 0.2 * _ymaxi if _ymaxi > 0 else 1)
				_ymini, _ymaxi = ax_sp.get_ylim()
				ax_sp.set_ylim(ymax=_ymaxi + 0.2 * _ymaxi if _ymaxi > 0 else 1)
				ax_sp_twin.tick_params(axis='y', labelsize=_lettersize - 2, colors='g')
				ax_sp.tick_params(axis='y', labelsize=_lettersize - 2, colors='k')
				ax_sp.set_xlabel('Spectral type', fontsize=_lettersize)
				ax_sp.tick_params(axis='x', labelsize=_lettersize - 2)
				ax_sp_twin.tick_params(axis='x', labelsize=_lettersize - 2)

				# --- Age ---
				if len(hist_all_age) > 0:
					ax_age.hist(hist_all_age, histtype='step', density=True, fill=True, color='k', alpha=0.3)
					ax_age.hist(hist_all_age, histtype='stepfilled', density=True, fill=False, hatch='/', color='k', linewidth=0.01)
				if len(hist_obs_age) > 0:
					ax_age.hist(hist_obs_age, histtype='step', density=True, fill=True, color='lightgreen', alpha=0.3)
					ax_age.hist(hist_obs_age, histtype='stepfilled', density=True, fill=False, hatch='.', edgecolor='lightgreen', linewidth=0.01)
				ax_age.set_xlabel('Age [Gyr]', fontsize=_lettersize)
				ax_age.set_yscale('log')
				ax_age.tick_params(axis='both', labelsize=_lettersize - 2)

				# --- Stellar mass ---
				if len(hist_all_mass) > 0:
					ax_mass.hist(hist_all_mass, histtype='step', bins=50, density=True, fill=True, color='k', alpha=0.3)
					ax_mass.hist(hist_all_mass, histtype='stepfilled', bins=50, density=True, fill=False, hatch='/', color='k', linewidth=0.01)
				if len(hist_obs_mass) > 0:
					ax_mass.hist(hist_obs_mass, histtype='step', density=True, fill=True, color='lightgreen', alpha=0.3)
					ax_mass.hist(hist_obs_mass, histtype='stepfilled', density=True, fill=False, hatch='.', edgecolor='lightgreen', linewidth=0.01)
				ax_mass.set_xlabel(r'$M_\star$ [$M_\odot$]', fontsize=_lettersize)
				ax_mass.set_yscale('log')
				ax_mass.tick_params(axis='both', labelsize=_lettersize - 2)

				# Metallicity panel (commented out)
				# ax_met.hist(hist_all_FeH, density=True, color='k', alpha=0.3)
				# ax_met.hist(hist_obs_FeH, density=True, color='lightgreen', alpha=0.3)
				# ax_met.set_xlabel('Fe/H [dex]', fontsize=_lettersize)

				plt.tight_layout()

				_sph_figname = os.path.join(route, f'histogram_stellarproperties_{_safe_filter_sph}_{_safe_model_sph}_{_run_date_sph}')
				fig_sph.savefig(_sph_figname + '.png', bbox_inches='tight')
				fig_sph.savefig(_sph_figname + '.pdf', bbox_inches='tight')
				plt.close(fig_sph)
				print(f'[Stellar histogram] Saved: {_sph_figname}.pdf')




