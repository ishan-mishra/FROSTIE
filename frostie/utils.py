'''
Library of general utility functions helpful in Europa spectroscopic data analysis work.
'''

import numpy as np
from scipy.interpolate import interp1d
from pandas import DataFrame, read_csv
import pandas as pd
import warnings
from pandexo.engine import bintools
import os
from jdap import io

# define global variable to store this file's absolute path
dirname = os.path.dirname(os.path.abspath(__file__))


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs in a 1D numpy array.

    Parameters
    ----------
    y: numpy array
        array with possible NaNs
        
    Returns
    -------
    nans: numpy array 
        logical indices of NaNs
    index: lambda function
        a function, with signature indices= index(nans),
        to convert the logical indices of NaNs to 'equivalent' indices
        
    Example
    -------
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    nans = np.isnan(y)
    index = lambda z: z.nonzero()[0]
    
    return nans, index


def despike_data(y,x,window_len=21,iqr_factor=2.5,x_start=2.0,x_end=5.0):
    """Flags points that are beyond a caculated threshold within the moving window. 
    The threshold is calculated as iqr_factor*IQR(y[data_in_window]), where IQR is the Inter Quartile Range. 
    
    Parameters
    ----------
    y: 1D numpy float array
        data that needs to be despiked
    x: 1D numpy float array
        the abscissa/x-axis correspoding to y (usually wavelength)   
    window_len: int
        size of the moving window
    iqr_factor: float
        multiplicative factor used in calculating the threshold
    x_start: float
        starting value of x array if only a limited range needs to be despiked.
        Default value is in microns as x is usually a wavelength array.
    x_end: float
        end value of x array if only a limited range needs to be despiked.
        Default value is in microns as x is usually a wavelength array.
        
    Returns
    -------
    
    flagged_list: list of 1D numpy float arrays
        list of all the flagged indices
    y_cleaned: 1D numpy float array
        final cleaned y array
    x_cleaned: 1D numpy float array
        x-axis for y_cleaned
        
    Example
    -------
    
    >>> y = np.array([ 0.,  1.,  2.,  3., 30.,  5., 40.,  7., 20.,  9., 10., 11., 12.,
                      13., 14., 15.])
    >>> x = np.linspace(15,30,num=16)  
    >>> flagged_list,y_cleaned,x_cleaned = despike_data(y,x,5)
    >>> flagged_list
    [4, 6, 8]
    >>> y_cleaned
    array([ 0.,  1.,  2.,  3.,  5.,  7.,  9., 10., 11., 12., 13., 14., 15.])
    >>> x_cleaned
    array([15., 16., 17., 18., 20., 22., 24., 25., 26., 27., 28., 29., 30.])
    
    """
    
    if np.sum(np.isnan(y)) > 0:
        raise ValueError("Data should not have any nans")

    # make sure the window len is int

    window_len = int(window_len)

    # slice the y array in the range x_start:x_end
    ind_start = find_nearest(x,x_start)
    ind_end = find_nearest(x,x_end)

    y_slice = np.copy(y[ind_start:ind_end+1])

    # extend the sliced array at the upper end by mirroring points

    y_slice = np.r_[y_slice,y_slice[-2:-window_len-1:-1]]

    
    # slide window across the array and flag points that are outliers

    #print(ind_start,ind_end)

    flagged_list = []

    for i in range(0,y_slice.size - window_len + 1,window_len):

        in_window = y_slice[i:i+window_len]

        #Find Q1 and Q3
        q25, q75 = np.percentile(in_window, 25), np.percentile(in_window, 75)
        iqr = q75 - q25

        # calculate the outlier cutoff
        cut_off = iqr*iqr_factor
        lower, upper = q25 - cut_off, q75 + cut_off

        # identify outliers
        outliers = [j + i for j in range(in_window.size) if in_window[j] < lower or in_window[j] > upper]

        #add to list of flagged indices

        for outlier in outliers:
            if outlier not in flagged_list:
                flagged_list.append(outlier)

    flagged_list = np.array(flagged_list) + ind_start

    y_cleaned = np.array([y[i] for i in range(y.size) if i not in flagged_list])
    x_cleaned = np.array([x[i] for i in range(y.size) if i not in flagged_list])

    
    flagged_list = [i for i in range(y.size) if y[i] not in y_cleaned]
   
    return flagged_list,y_cleaned,x_cleaned


def find_nearest(array,value):
    """Finds the index of the array element with value closest to the supplied value.
    If multiple array elements are 'nearest' to the input value, then the lowest index is
    selected.
    """

    idx = np.argwhere(np.abs(array-value) == np.abs(array-value).min())[0][0]
    return idx

def resolution(wav):
    """Gives the resolution array for a given wavelength array"""

    resolution = np.empty(wav.size-1)

    for i in range(wav.size-1):
        resolution[i] = wav[i+1] - wav[i]
        
    return resolution

def eq_wav_axes(wav_1,wav_2):
    
    """Trims the two wavelength arrays so that their min and max values closely match

    Parameters
    ----------
    wav_1: 1D numpy array
        first wavelength axis
    wav_2: 1D numpy array
        second wavelength axis

    Returns
    -------
    wav_1_l: int
        lower index for the trimmed wav_1
    wav_1_u: int
        upper index for the trimmed wav_1
    wav_2_l: int
        lower index for the trimmed wav_2
    wav_2_u: int
        upper index for the trimmed wav_2
    
    """
    #Adjust at the lower end first
    
    if wav_1.min() >= wav_2.min():
        
        idx = find_nearest(wav_2,wav_1.min())
        wav_1_l = 0
        wav_2_l = idx
        
    elif wav_2.min() > wav_1.min():
        idx = find_nearest(wav_1,wav_2.min())
        wav_2_l = 0
        wav_1_l = idx
        
    #Adjust the upper end
    
    if wav_1.max() <= wav_2.max():
        
        idx = find_nearest(wav_2,wav_1.max())
        wav_1_u = wav_1.size - 1     #+1 because [:i] trims upto index i-1
        wav_2_u = idx
        
    elif wav_2.max() < wav_1.max():
        idx = find_nearest(wav_1,wav_2.max())
        wav_2_u = wav_2.size - 1
        wav_1_u = idx
    
    return wav_1_l,wav_1_u,wav_2_l,wav_2_u


''' OLD VERSION:

def spectra_match(wav_1,wav_2):
    
    
    Modifies the two input spectra so that their wavelength axis are the same
    The idea is to trim down the spectra with higher resolution to the lower resolution
    spectra's wavelength axis.

    Parameters
    ----------
    wav_1: 1D numpy array
        first wavelength axis
    wav_2: 1D numpy array
        second wavelength axis

    Returns
    -------
    wav_common_axis: 1D numpy array
        modified common wavelength axis
    indices_1: list of int
        surviving indices of wav_1
    indices_2: list of int
        surviving indices of wav_2
    
    
    #trim the low resolution array to match the wavelength range with the high res array
    l_1,u_1,l_2,u_2 = eq_wav_axes(wav_1,wav_2)
    wav_1 = wav_1[l_1:u_1+1]
    wav_2 = wav_2[l_2:u_2+1]




        
    if resolution(wav_1).mean() >= resolution(wav_2).mean():
        
        #Now trim out the high_res array to exactly match the low_res array 
        
        indices_2 = np.zeros(wav_1.size,dtype=int)
    
        for i in range(wav_1.size):
            
            idx = find_nearest(wav_2,wav_1[i])
            indices_2[i] = idx + l_2
        
        
        wav_2 = np.copy(wav_1)          #APPROXIMATION
        #val_2 = val_2[indices_2]
        indices_1 = np.array([i for i in range(l_1,u_1+1)])
               
    
    elif resolution(wav_2).mean() > resolution(wav_1).mean():
    
        
        #Now trim out the high_res array to exactly match the low_res array 
        
        indices_1 = np.zeros(wav_2.size,dtype=int)
    
        for i in range(wav_2.size):
            
            idx = find_nearest(wav_1,wav_2[i])
            indices_1[i] = idx + l_1
        
         

        wav_1 = np.copy(wav_2)          #APPROXIMATION
        #val_1 = val_1[indices_1] 
        indices_2 = np.array([i for i in range(l_2,u_2+1)])
        #k_obs = k_obs[indices]
        
    wav_common_axis = np.copy(wav_1)    #Wav_2 is the same 
    
    #print('test')
    return wav_common_axis,indices_1,indices_2

'''

def spectra_match(data_1,wav_1,data_2,wav_2):
    
    '''
    Modifies the two input spectra so that their wavelength axes are the same.
    The idea is to reduce the spectra with higher resolution to the lower resolution
    spectra's wavelength axis.

    Parameters
    ----------
    data_1: 1D numpy array
        first data array
    wav_1: 1D numpy array
        first wavelength axis in microns
    data_2: 1D numpy array
        second data array
    wav_2: 1D numpy array
        second wavelength axis in microns

    Returns
    -------

    binned: dictionary with the follwing keys:

        'new_data_1': 1D numpy array
            modified first data array
        'new_data_2': 1D numpy array
            modified second data array
        'wav_common': 1D numpy array
            common wavelength axis
    
    '''
    # trim the edges of the arrays so that they span roughly the same wavelength range

    l_1,u_1,l_2,u_2 = eq_wav_axes(wav_1,wav_2)

    wav_1 = wav_1[l_1:u_1+1]
    data_1 = data_1[l_1:u_1+1]

    wav_2 = wav_2[l_2:u_2+1]
    data_2 = data_2[l_2:u_2+1]


    if resolution(wav_1).mean() >= resolution(wav_2).mean():
        
        # wav_2 has a higher resolution or more closely spaced wavelength axis

        binned = bintools.binning(wav_2,data_2,newx=wav_1)

        # binned['bin_x'] can be different from wav_1 because binning() drops any nans that may 
        # be produced, so modify data_1 accordingly
    
        new_data_1 = []

        for i in range(wav_1.size):
            wav = wav_1[i]
            if wav in binned['bin_x']:
                new_data_1.append(data_1[i])

        new_data_1 = np.array(new_data_1)

        wav_common = binned['bin_x']
        new_data_2 = binned['bin_y']

        binned = {'new_data_1':new_data_1,'new_data_2':new_data_2,'wav_common':wav_common}

        return binned       
    
    elif resolution(wav_2).mean() > resolution(wav_1).mean():
    
        
        # wav_1 has a higher resolution or more closely spaced wavelength axis

        binned = bintools.binning(wav_1,data_1,newx=wav_2)

        # binned['bin_x'] can be different from wav_2 because binning() drops any nans that may 
        # be produced, so modify data_2 accordingly
    
        new_data_2 = []

        for i in range(wav_2.size):
            wav = wav_2[i]
            if wav in binned['bin_x']:
                new_data_2.append(data_2[i])

        new_data_2 = np.array(new_data_2)

        wav_common = binned['bin_x']
        new_data_1 = binned['bin_y']

        binned = {'new_data_1':new_data_1,'new_data_2':new_data_2,'wav_common':wav_common}

        return binned




def bin_model(wav_data, model_wav, model_ref):
    """Function to bin a model into the data's wavelength bins values.

    Parameters
    ----------

    wav_data: numpy float array
        wavelength values for the data, usually in microns. 
    model_wav: numpy float array
        wavelength values for the model, usually in microns. 
    model_ref: numpy float array
        reflectance values for the model.
    

    Returns
    -------

    model_ref_binned: numpy float array
        model reflectance values binned to the data's resolution.

    
    """
    
    wav_data_err = get_wav_bins(wav_data)

    if resolution(model_wav).mean() > resolution(wav_data).mean():
        warnings.warn('The model is of a lower resolution than the data, which can create nans during binning')

    model_ref_binned = np.full(wav_data.size,np.nan)

    len_data = wav_data.size
    

    for i in range(len_data-1):
        i=i+1
        loc=np.where((model_wav > wav_data[i]-wav_data_err[i]) & (model_wav < wav_data[i]+wav_data_err[i]))
        model_ref_binned[i] = np.mean(model_ref[loc])
    
    loc=np.where((model_wav > wav_data[0]-wav_data_err[0]) & (model_wav < wav_data[0]+wav_data_err[0]))
    model_ref_binned[0] = np.mean(model_ref[loc])

    return model_ref_binned



def data_median(data_list, wav_data_list):
    """Function to take median of given set of JIRAM data.

    Parameters
    ----------

    data_list: list of 1D numpy arrays
        list of spectra which are products of io.get_jiram_data() function.
    wav_data_list: list of 1D numpy arrays
        list of corresponding wavelength axes in microns.

    Returns
    -------

    data_median: 1D numpy array
        the median spectrum
    wav_median: 1D numpy array
        the original wavelength array in microns


    """

    for wav_data in wav_data_list:
        for wav in wav_data:
            if wav >3.7 and wav <3.9:
                raise ValueError('Please remove 3.7-3.9 microns region from the data as its too noisy.')
    
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Data','wavelength.txt'))

    wav_data = np.loadtxt(filename)

    wav_data = wav_data/1000.

    # remove 3.7-3.9 microns region just like the input data
    ind=(wav_data < 3.7) | (wav_data > 3.9)

    wav_data = wav_data[ind]

    data_median = []

    for wav in wav_data:
        val_list = []
        for i in range(len(wav_data_list)):
            if wav in wav_data_list[i]:
                val_list.append(data_list[i][np.where(wav_data_list[i] == wav)[0]])
        if len(val_list) == 0:
            data_median.append(np.nan)
        else:
            data_median.append(np.median(np.array([val_list])))
    
    data_median = np.array(data_median)
    
    nans,index = nan_helper(data_median)
    
    data_median = data_median[~nans]
    wav_median = wav_data[~nans]    

    return data_median, wav_median


def data_mean(data_list, wav_data_list):
    """Function to take median of given set of JIRAM data.

    Parameters
    ----------

    data_list: list of 1D numpy arrays
        list of spectra which are products of io.get_jiram_data() function.
    wav_data_list: list of 1D numpy arrays
        list of corresponding wavelength axes in microns.

    Returns
    -------

    data_mean: 1D numpy array
        the median spectrum
    wav_data: 1D numpy array
        the original wavelength array in microns


    """

    for wav_data in wav_data_list:
        for wav in wav_data:
            if wav >3.7 and wav <3.9:
                raise ValueError('Please remove 3.7-3.9 microns region from the data as its too noisy.')
    
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Data','wavelength.txt'))

    wav_data = np.loadtxt(filename)

    wav_data = wav_data/1000.

    # remove 3.7-3.9 microns region just like the input data
    ind=(wav_data < 3.7) | (wav_data > 3.9)

    wav_data = wav_data[ind]


    data_mean = []

    for wav in wav_data:
        val_list = []
        for i in range(len(wav_data_list)):
            if wav in wav_data_list[i]:
                val_list.append(data_list[i][np.where(wav_data_list[i] == wav)[0]])
        if len(val_list) == 0:
            data_mean.append(np.nan)
        else:
            data_mean.append(np.mean(np.array([val_list])))
    
    data_mean = np.array(data_mean)
    
    nans,index = nan_helper(data_mean)
    data_mean = data_mean[~nans]
    wav_data = wav_data[~nans]    

    return data_mean, wav_data


def data_std(data_list, wav_data_list):
    """Function to take standard deviation of given set of JIRAM data.

    Parameters
    ----------

    data_list: list of 1D numpy arrays
        list of spectra which are products of io.get_jiram_data() function.
    wav_data_list: list of 1D numpy arrays
        list of corresponding wavelength axes in microns.

    Returns
    -------

    data_std: 1D numpy array
        the median spectrum
    wav_data: 1D numpy array
        the original wavelength array in microns


    """

    for wav_data in wav_data_list:
        for wav in wav_data:
            if wav >3.7 and wav <3.9:
                raise ValueError('Please remove 3.7-3.9 microns region from the data as its too noisy.')
    
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Data','wavelength.txt'))

    wav_data = np.loadtxt(filename)

    wav_data = wav_data/1000.

    # remove 3.7-3.9 microns region just like the input data
    ind=(wav_data < 3.7) | (wav_data > 3.9)

    wav_data = wav_data[ind]


    data_std = []

    for wav in wav_data:
        val_list = []
        for i in range(len(wav_data_list)):
            if wav in wav_data_list[i]:
                val_list.append(data_list[i][np.where(wav_data_list[i] == wav)[0]])
        if len(val_list) == 0:
            data_std.append(np.nan)
        else:
            data_std.append(np.std(np.array([val_list])))
    
    data_std = np.array(data_std)
    
    nans,index = nan_helper(data_std)
    data_std = data_std[~nans]
    wav_data = wav_data[~nans]    

    return data_std, wav_data


def get_snr(data,wav_data):
    """Function to calculate the snr for a given (I/F) spectrum.
    SNR = radiance/nesr, where radiance is calculated from the (I/F)
    spectrum.
     
    Parameters
    ----------
    
    data: 1D numpy array
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

def get_noise(data,wav_data):
    """Function to calculate the noise for a given (I/F) spectrum.
    
    Parameters
    ----------
    
    data: 1D numpy array
        the I/F spectrum
    wav_data: 1D numpy array
        corresponding wavelength array in microns
        
    Returns:
    --------
    
    data: 1D numpy array
    noise: 1D numpy array
    wav: 1D numpy array
        common wavelength array for data_new and noise in microns
    """
    snr, wav_snr = get_snr(data,wav_data)
    
    data_snr_match = spectra_match(data,wav_data,
                                                 snr,wav_snr)

    data,snr,wav_common = data_snr_match['new_data_1'],data_snr_match['new_data_2'],data_snr_match['wav_common']
    
    noise = data/snr
    
    # remove NANs 
    
    nans,index = nan_helper(noise)
    noise = noise[~nans]
    wav_common = wav_common[~nans]  
    data = data[~nans]
    
    return data,noise, wav_common

def get_avg_snr(data_list,wav_data_list):
    """Function to calculate the average snr for a given set of (I/F) spectra. 
    
    Parameters
    ----------
    
    data_list: list of 1D numpy array
        I/F spectra
    wav_data_list: list of 1D numpy array
        corresponding wavelength array in microns
    
    Returns
    -------
    
    snr_avg: 1D numpy array
        average snr of all the input spectra
    wav_avg: 1D numpy array
        corresponding wavelength array in microns
    """

    for wav_data in wav_data_list:
        for wav in wav_data:
            if wav >3.7 and wav <3.9:
                raise ValueError('Please remove 3.7-3.9 microns region from the data as its too noisy.')
    
    snr_list = []
    wav_snr_list = []

    for i in range(len(data_list)):
        snr,wav_snr = get_snr(data_list[i],wav_data_list[i])
        snr_list.append(snr)
        wav_snr_list.append(wav_snr)

        
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Data','wavelength.txt'))

    wav_data = np.loadtxt(filename)

    wav_data = wav_data/1000.
        
    snr_avg = []
    
    for wav in wav_data:
        val_list = []
        for i in range(len(wav_snr_list)):
            if wav in wav_snr_list[i]:
                val_list.append(snr_list[i][np.where(wav_snr_list[i] == wav)[0]])
        if len(val_list) == 0:
            snr_avg.append(np.nan)
        else:
            snr_avg.append(np.mean(np.array([val_list])))
    
    snr_avg = np.array(snr_avg)
    
    nans,index = nan_helper(snr_avg)
    snr_avg = snr_avg[~nans]
    wav_avg = wav_data[~nans]    

    return snr_avg, wav_avg


def get_rms_noise(data_list,wav_data_list):
    """Function to calculate the rms noise for a given set of (I/F) spectra. 
    
    Parameters
    ----------
    
    data_list: list of 1D numpy array
        I/F spectra
    wav_data_list: list of 1D numpy array
        corresponding wavelength array in microns
    
    Returns
    -------
    
    noise_rms: 1D numpy array
        rms noise of all the input spectra
    wav_noise_rms: 1D numpy array
        corresponding wavelength array in microns
    """
    noise_list = []
    wav_noise_list = []
    
    for i in range(len(data_list)):
        
        junk,noise, wav_noise = get_noise(data_list[i],wav_data_list[i])

        noise_list.append(noise)
        wav_noise_list.append(wav_noise)

    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','Data','wavelength.txt'))

    wav_data = np.loadtxt(filename)

    wav_data = wav_data/1000.
        
    noise_rms = []
    
    for wav in wav_data:
        val_list = []
        for i in range(len(wav_noise_list)):
            if wav in wav_noise_list[i]:
                val_list.append(noise_list[i][np.where(wav_noise_list[i] == wav)[0]])
        if len(val_list) == 0:
            noise_rms.append(np.nan)
        else:
            noise_rms.append(np.sqrt(np.sum(np.array(val_list)**2)))
    
    noise_rms = np.array(noise_rms)

    nans,index = nan_helper(noise_rms)
    noise_rms = noise_rms[~nans]
    wav_noise_rms = wav_data[~nans]    

    return noise_rms, wav_noise_rms
    
    
    
''' How I was calculating noise originally. Refer to the notebook for the new methodology
followed by Filacchione et al. (2019)
    
def jiram_noise(data,wav_data):
    """Function to calculate noise on JIRAM's data.

    Parameters
    ----------

    data: 1D numpy array
        spectrum to calculate error for
    wav_data: 1D numpy array
        corresponding wavelength array

    Returns
    -------

    data_noise: 1D numpy array
        spectrum noise
    data_new: 1D numpy array
        data mapped to a common wavelength axis
    wav_common: 1D numpy array
        common wavelength axis for data_noise and data_new

    """

    D = 5.43    # in AU, calculated using JIRAM 


    # read in the solar irradiance data

    si, si_wav = io.get_si(start_wav=wav_data.min(), end_wav=wav_data.max())

    # read in the noise-equivalent spectral radiance

    nesr,nesr_wav  = io.get_jiram_nesr()
    
    match_si_nesr = spectra_match(si,si_wav,nesr,nesr_wav)

    si, nesr, wav_common = match_si_nesr['new_data_1'],match_si_nesr['new_data_2'],match_si_nesr['wav_common']

    data_noise = 4*np.pi*D**2*nesr/(si)

    data_noise = data_noise/norm_factor

    # match to data wavelength axis

    data_noise_match = spectra_match(data, wav_data, data_noise,wav_common)

    data,data_noise, wav_common2 = data_noise_match['new_data_1'],data_noise_match['new_data_2'],data_noise_match['wav_common']

    ind=(wav_common2 < 3.7) | (wav_common2 > 3.9)

    wav_common2 = wav_common2[ind]
    data = data[ind]
    data_noise = data_noise[ind]

    return data_noise,data,wav_common2
'''

def get_wav_bins(wav, fix_jiram=False):
    """Function to calculate wavelength bins for a given wavelength array

    NOTE: errors are defined as half the bin widths
    
    """
    
    wav_err = np.full(wav.size,np.nan)
    
    for i in range(wav.size):
        if i==0:
            wav_err[i] = (wav[i+1] - wav[i])
        elif i==wav.size-1:
            wav_err[i] = (wav[i] - wav[i-1])
        else:
            wav_err[i] = ((wav[i]-wav[i-1]) + (wav[i+1]-wav[i]))/2.0

    # the bin sizes are exaggerated for the points closest to 3.7 and 3.9 microns
    # due to the huge gap. Correct this.
    
    if fix_jiram:
        ind = find_nearest(wav,3.7)

        wav_err[ind] = wav_err[ind-1]
        wav_err[ind+1] = wav_err[ind+2]

    # make sure that all error values are positive
    assert all(err > 0 for err in wav_err)

    return wav_err

def get_radiance(data,wav_data):
    """Function to convert an I/F spectrum to radiance.

    Parameters
    ----------

    data: 1D numpy array
        I/F spectrum
    wav_data: 1D numpy array
        corresponding wavelength array in microns

    Returns
    -------

    radiance: 1D numpy array
        spectral radiance
    wav_radiance: 1D numpy array
        spectral radiance wavelength axis
    """

    D = 5.43    # in AU, calculated using JIRAM 


    # read in the solar irradiance data

    si, si_wav = io.get_si(start_wav=wav_data.min(),end_wav=wav_data.max())

    si_data_match = spectra_match(si,si_wav,data,wav_data)


    si,data, wav_common = si_data_match['new_data_1'],si_data_match['new_data_2'],si_data_match['wav_common']

    radiance = data*si/(4*np.pi*D**2)

    return radiance,wav_common

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Parameters
    ----------
    x: numpy float array
        the input signal 
    window_len: int
        the dimension of the smoothing window; should be an odd integer
    window: str
        the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        flat window will produce a moving average smoothing.

    Returns
    -------
    numpy float array
        the smoothed signal
        
    Example
    -------
    
    >>> t=linspace(-2,2,0.1)
    >>> x=sin(t)+randn(len(t))*0.1
    >>> y=smooth(x)
    
    See also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
        
    if (window_len % 2) == 0:
        raise ValueError("Window length needs to be odd")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')  #evalutes the expression as a python code

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y[int(window_len/2):-int(window_len/2)]


def get_hms(t_sec):
    """Converts amount of time in seconds to hours, minutes and seconds
    """
    h = t_sec//3600

    m = (t_sec - h*3600)//60

    s = t_sec%60

    return h,m,s
    
