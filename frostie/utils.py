# Utility functions for FROSTIE

import numpy as np
import os
import warnings
from scipy.special import erfcinv
from scipy.special import lambertw as W

def load_water_op_cons(wav_low=None, wav_high=None):
    """
    Load water ice optical constants included in FROSTIE.

    Returns
    -------
    wav : ndarray
        Wavelength array in microns.
    n : ndarray
        Real part of the refractive index.
    k : ndarray
        Imaginary part of the refractive index.
    """

    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'data',
        'h2o.dat'))
    
    # load columns from file

    data = np.loadtxt(filename)
    wav = data[0,:]
    n = data[1,:]
    k = data[2,:]

    # slice arrays within the wavelength limits input by user
    
    lower_cutoff = wav.min()
    upper_cutoff = wav.max()

    if wav_low != None:
        if wav_low < wav.min():
            warnings.warn('Lower wavelength value out of bounds with data. Data is in the %f - %f microns range.'%(wav.min(), wav.max()))
        else:
            lower_cutoff = wav_low
    if wav_high != None:
        if wav_high > wav.max():
            warnings.warn('Upper wavelength value out of bounds with data. Data is in the %f - %f microns range.'%(wav.min(), wav.max())) 
        else:
            upper_cutoff = wav_high


    ind = np.where((wav > lower_cutoff) & (wav < upper_cutoff))
    
    return wav[ind], n[ind], k[ind]


def load_co2_op_cons(wav_low=None, wav_high=None):
    """Load carbon dioxide ice optical constants included in FROSTIE.

    Returns
    -------
    wav : ndarray
        Wavelength array in microns.
    n : ndarray
        Real part of the refractive index.
    k : ndarray
        Imaginary part of the refractive index.
    """

    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'data',
        'co2.dat'))
    
    # load columns from file

    data = np.loadtxt(filename)
    wav = data[0,:]
    n = data[1,:]
    k = data[2,:]

    # slice arrays within the wavelength limits input by user
    
    lower_cutoff = wav.min()
    upper_cutoff = wav.max()

    if wav_low != None:
        if wav_low < wav.min():
            warnings.warn('Lower wavelength value out of bounds with data. Data is in the %f - %f microns range.'%(wav.min(), wav.max()))
        else:
            lower_cutoff = wav_low
    if wav_high != None:
        if wav_high > wav.max():
            warnings.warn('Upper wavelength value out of bounds with data. Data is in the %f - %f microns range.'%(wav.min(), wav.max())) 
        else:
            upper_cutoff = wav_high


    ind = np.where((wav > lower_cutoff) & (wav < upper_cutoff))
    
    return wav[ind], n[ind], k[ind]


def spectra_list_match(data_list,wav_list):
    """
    Resample all input spectra to a common wavelength axis.

    The common axis is chosen as the lowest-resolution wavelength array in the list.

    Parameters
    ----------
    data_list : list of ndarray
        List of 1D arrays representing spectral data to be resampled.
    wav_list : list of ndarray
        Corresponding list of 1D wavelength arrays.

    Returns
    -------
    data_matched_list : list of ndarray
        Spectra resampled to the common wavelength grid.
    wav_common : ndarray
        Common wavelength axis used for resampling.
    """

    
    min_res_list = []
    
    for i in range(len(data_list)):
        res = np.mean(wav_bins(data_list[i]))
        min_res_list.append(res)
    i_max= np.argmax(np.array(min_res_list))
        
    wav_common = np.copy(wav_list[i_max])
    
    # trim wav_common's lower end to be close to the highest lower end value among all 
    # wavelength arrays
    
    wav_low = max([wav_list[i][0] for i in range(len(wav_list))])
    
    idx_low = find_nearest(wav_common, wav_low)
    
    wav_common = wav_common[idx_low:]
    
    # trim wav_common's upper end to be close to the lowest upper end value among all 
    # wavelength arrays
    
    wav_high = min([wav_list[i][-1] for i in range(len(wav_list))])
    
    idx_high = find_nearest(wav_common, wav_high)
    
    wav_common = wav_common[:idx_high]
        
    # match all data arrays to this modified wav_common axis
        
    data_matched_list = []
        
    for i in range(len(data_list)):
        data_mod = []
        for j in range(wav_common.size):
            idx = find_nearest(wav_list[i],wav_common[j])
            data_mod.append(data_list[i][idx])
            
        data_matched_list.append(np.array(data_mod))
        
    return data_matched_list, wav_common


def wav_bins(wav):
    """
    Compute the wavelength bin edges for a given wavelength array.

    Parameters
    ----------
    wav : ndarray
        Wavelength array.

    Returns
    -------
    bins : ndarray
        Bin widths between consecutive wavelengths.
    """

    wav_bins = np.empty(wav.size-1)

    for i in range(wav.size-1):
        wav_bins[i] = wav[i+1] - wav[i]
        
    return wav_bins

def find_nearest(array,value):
    """
    Find the index of the array element closest to the supplied value.

    If multiple elements are equally close, the first match is returned.

    Parameters
    ----------
    array : ndarray
        Array to search.
    value : float
        Target value.

    Returns
    -------
    index : int
        Index of the closest array element.
    """


    idx = np.argwhere(np.abs(array-value) == np.abs(array-value).min())[0][0]
    return idx


# def instrument_convolution(wav, spec, instrument="NIMS"):
#     """
#     Convolve a spectrum with an instrument-specific response function.

#     Currently supports:
#     - Galileo NIMS (default: boxcar of 0.03 μm FWHM)

#     Parameters
#     ----------
#     spectrum : ndarray
#         Input reflectance spectrum.
#     wavelengths : ndarray
#         Wavelength array corresponding to the spectrum.
#     instrument : str, optional
#         Name of the instrument to simulate (default is 'nims').

#     Returns
#     -------
#     spectrum_convolved : ndarray
#         Convolved spectrum.
#     """


#     return wav, spec

#     # if instrument is not NIMS, print an error


#     # define NIMS convolution function


#     # convolve with input arrays

def Z_to_sigma(ln_Z1, ln_Z2):
    """
    Convert log-evidences of two models to a sigma confidence level.

    Parameters
    ----------
    ln_Z1 : float
        Log-evidence of full model.
    ln_Z2 : float
        Log-evidence of reduced model.

    Returns
    -------
    B : float
        Bayes factor.
    sigma : float
        Sigma confidence level.
    """
    delta_logZ = ln_Z1 - ln_Z2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        B = np.exp(delta_logZ)

    # if not np.isfinite(B) or B < 1.0:
    #     warnings.warn("Bayes factor is invalid (infinite or < 1); sigma significance may be meaningless.")
    #     return np.inf, np.inf

    p = np.real(np.exp(W((-1.0 / (B * np.exp(1))), -1)))
    sigma = np.sqrt(2) * erfcinv(p)

    return B, sigma
