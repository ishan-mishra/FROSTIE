# Utility functions for FROSTIE

import numpy as np
import os
import warnings

def load_water_op_cons(wav_low=None, wav_high=None):

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
    
    '''
    Modifies the input spectra so that their wavelength axes are the same. The common wavelength
    axis is the wavelength array from wav_list with the lowest resolution.
 
    Parameters
    ----------
    data_list: list of 1D numpy arrays
        list of data arrays
    wav_list: list of 1D numpy arrays
        list of wavelength arrays

    Returns
    -------

    data_matched_list: list of 1D numpy arrays
        list of modified data arrays
    wav_common: 1D numpy array
        common wavelength axis
    '''
    
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
    """Gives the wavelength bin array for a given wavelength array"""

    wav_bins = np.empty(wav.size-1)

    for i in range(wav.size-1):
        wav_bins[i] = wav[i+1] - wav[i]
        
    return wav_bins

def find_nearest(array,value):
    """Finds the index of the array element with value closest to the supplied value.
    If multiple array elements are 'nearest' to the input value, then the lowest index is
    selected.
    """

    idx = np.argwhere(np.abs(array-value) == np.abs(array-value).min())[0][0]
    return idx
