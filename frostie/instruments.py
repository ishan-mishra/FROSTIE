'''Functions utilized to simulate data and noise from various isntruments'''

import numpy as np
import pandas as pd
import os

def get_si(start_wav, end_wav, file='kurucz'):
    """Function to read in the solar irradiance data

    Parameters
    ----------

    start_wav: float
        start value of the desired wavelength range
    end_wav: float
        end valye of the desired wavelength range
    file: string
        SI data file to load. Options are 'kurucz' and 'astm'

    Returns:
    --------
    si: 1D numpy array
        solar irradiance in W/m^2/micron
    si_wav: 1D numpy array
        corresponsing wavelength axis in microns
    """

    # read in the solar irradiance data

    if file not in ['kurucz','astm']:
        raise ValueError("Invalid file string. Valid options are 'kurucz' and 'astm'")

    if file=='astm':

        filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','instruments',
        'solar_irradiance','e490_00a_amo.xls'))

        df = pd.read_excel(filename)

        if start_wav < df['Wavelength, microns'].min():
            raise ValueError('start wavelength outside the range of SI data')

        if end_wav > df['Wavelength, microns'].max():
            raise ValueError('send wavelength outside the range of SI data')

        si_wav = df['Wavelength, microns'][(df['Wavelength, microns'] > start_wav) \
        & (df['Wavelength, microns'] < end_wav)]
        si = df['E-490 W/m2/micron'][(df['Wavelength, microns'] > start_wav) \
        & (df['Wavelength, microns'] < end_wav)]

        si_wav = np.array(si_wav)
        si = np.array(si)

        return si, si_wav

    if file=='kurucz':

        filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','instruments',
        'solar_irradiance','kurucz_si.txt'))

        f=open(filename, 'r')

        lines = f.read().splitlines()

        lines = [lines[i].split() for i in range(len(lines))]

        f.close()

        si_kurucz = []
        wav_kurucz = []

        for i in range(2,len(lines)):
            si_kurucz.append(float(lines[i][1]))
            wav_kurucz.append(float(lines[i][0]))
    
        si_kurucz = np.array(si_kurucz)
        wav_kurucz = np.array(wav_kurucz)/1000.

        return si_kurucz,wav_kurucz
    

### INCOMPLETE FUNCTION
def rad_to_IF(rad,wav_rad, SI, wav_SI, D):
    """Function to convert radiance to (I/F) spectrum.
    SNR = radiance/nesr, where radiance is calculated from the (I/F)
    spectrum.
     
    Parameters
    ----------
    
    rad: 1D numpy array
        the I/F spectrum
    wav_data: 1D numpy array
        corresponding wavelength array in microns
        
    Returns:
    --------
    
    snr: 1D numpy array
    wav_snr: 1D numpy array
        corresponding wavelength array in microns

    """
    
    radiance, wav_radiance = get_radiance(data,wav_data)
    
    nesr,wav_nesr = io.get_jiram_nesr()
    
    rad_nesr_match = spectra_match(radiance,wav_radiance,nesr,wav_nesr)


    radiance,nesr, wav_snr = rad_nesr_match['new_data_1'],rad_nesr_match['new_data_2'],rad_nesr_match['wav_common']
    
    snr = radiance/nesr
    
    return snr, wav_snr


def get_NIMS_wav():
    """Function to load Galileo/NIMS wavelength channels
    """
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','instruments',
        'Galileo_NIMS_wav.txt'))
    
    wav = np.loadtxt(filename)
    
    return wav