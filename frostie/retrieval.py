"""Module with functions and classes for different Bayesian retrieval frameworks

"""

import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd 
from scipy.interpolate import interp1d
import jdap.utils as utils
from pyhapke import pyhapke
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
from scipy.special import lambertw as W
from scipy.special import erfcinv
import dynesty
from dynesty import NestedSampler
from dynesty.utils import resample_equal
from dynesty import utils as dyfunc
import corner
from scipy.stats import gaussian_kde
import pickle
import warnings


'''
class Mcmc():
	"""Class to set up an mcmc run using the emcee package. Uniform distributions for prior
	assumed. More functions to be added later.

	:param data: dictionary containing 1D numpy float arrays with keys 'data','wav_data',
	'data_err','wav_err'
	:type data: dict
	:param model: function that returns the simuated data binned to data's resolution
	:type model: 1D numpy float array
	:param keys_free_param: list of keywords of :param:`model` that are free parameters
	:type keys_free_param: list of str
	:param keys_fixed_param: list of keywords of :param:`model` that are fixed parameters
	:type fixed_params_keys: list of str

	:Example:

    Here's an example that uses the package pyhapke's get_r_mix2() function

	FILL LATER
	"""


	def __init__(self,data,model, keys_free_param, keys_fixed_param):
		"""Constructor method
		"""

		self.data = data
		self.model = model
		self.keys_free_param = keys_free_param
		self.keys_fixed_param = keys_fixed_param
		
    def set_fix_param(list_fix):
    	"""Sets the values of fixed parameters

    	:param list_fix: list of values for fixed parameters
    	:type list_fix: list of diverse objects like floats and numpy arrays. 
    	"""
        
        self.dict_free_params = {}

        for param_fixed in self.keys_fixed_params:

        	self.dict_free_params{param_fixed:list_fix    }
            


	def set_prior(list_prior):
		"""Sets the min and max values of the free parameters to be explored by the MCMC
		sampler.

        :param list_prior: list of tuples containing min and max values of each free 
        parameter
        :type list_prior: list of tuples

        :Example:
        
        >>> object_mcmc.set_prior([(min_1,max_1),(min_2,max_2)])
		"""
        
        self.list_prior = list_prior


	def _lnlike(theta):
		"""Returns the natural log likelihood value of the model spectrum

		:param theta: 
		"""


	def _lnprior(theta):
        """Returns the natural log prior value given the free parameter values
        """
        



	def _lnprob():


	def run_mcmc(multiprocessing=True,store_results=True):
		"""
		:param multiprocessing: True/False flag for enabling/disabling the use of multiprocessing for
	    mcmc
	    :type model: Boolean
	    :param store_results: True/False flag to enable/disable the retrieval results 
	    being stored in an hdf5 file.
		"""

		if store_results:
			timestr = time.strftime("%Y%m%d-%H%M%S")
			self.file_name = timestr + '.hdf5'






	"""Class to set up an mcmc run using the emcee package
	"""

	
'''

'''
class Dynesty():
    
    def __init__(self,data,model, keys_free_param, keys_fixed_param):
        """Constructor method
        """

        self.data = data
        self.model = model
        self.keys_free_param = keys_free_param
        self.keys_fixed_param = keys_fixed_param 
        
    def set_prior(self,prior_bounds):
        """Sets the min and max values of the free parameters to be explored by the
        sampler.

        :param prior_bounds: list of tuples containing min and max values of each free 
        parameter
        :type prior_bounds: list of tuples

        :Example:
        
        >>> object_dynesty.prior_bounds([(min_1,max_1),(min_2,max_2)])
        """
        
        self.prior_bounds = prior_bounds
        
    def _prior_transform(self,theta):
        """ A function defining the tranform between the parameterisation in the unit hypercube
        to the true parameters. Used in dynesty (nested sampling)
    
        Parameters
        ----------
        theta: tuple of floats
            free parameter values supplied by the sampler
        
        Returns
        -------
    
        tuple containing all the transformed free parameter values
        """
    
        transformed = ()
    
        for i,param_prime in enumerate(theta):
            param_min = self.prior_bounds[i][0]
            param_max = self.prior_bounds[i][1]
        
            param = param_prime*(param_max - param_min) + param_min
        
            transformed += (param,)
        
        return transformed
        
'''

def LM_2comp(param_dict):
    """Two component Hapke reflectance model.
    
    Parameters
    ----------
    param_dict: dictionary of parameters for the model. 
       Contains the following keywords:
       'mu_0': cosine of the incidence angle (float)
       'mu': cosine of the emergence angle (float)
       'g': phase angle in degrees (float)
       'B': backscattering function value (float)
       'K': porosity coefficient
       's': internal scattering coefficient inside the particle
       'comp_1': dict containing 'n', 'k', 'wav'(in microns),'log10D' (log10 grain size in microns) 
        and 'f' (number density fraction) of component 1
       'comp_2': dict containing 'n', 'k', 'wav'(in microns),'log10D' (log10 grain size in microns) 
        and 'f' (number density fraction) of component 2
    **kwags: arguments for other functions
    
    Returns
    -------
    model: 1D nupy float array
        model reflectance spectrum
    wav_model: 1D numpy array
        corresponding wavelength axis in microns
    """
    
    pd = param_dict
    comp_1 = pd['comp_1']
    comp_2 = pd['comp_2']
    
    
    b_1 = pyhapke.get_b(comp_1['n'],comp_1['k'],comp_1['wav'],10**comp_1['log10D'],pd['s'])
    b_2 = pyhapke.get_b(comp_2['n'],comp_2['k'],comp_2['wav'],10**comp_2['log10D'],pd['s'])
    
    c_1 = pyhapke.get_c(comp_1['n'],comp_1['k'],comp_1['wav'],10**comp_1['log10D'],pd['s'])
    c_2 = pyhapke.get_c(comp_2['n'],comp_2['k'],comp_2['wav'],10**comp_2['log10D'],pd['s'])
    
    p_1 = pyhapke.get_p(pd['g'],type='HG2',b=b_1,c=c_1)
    p_2 = pyhapke.get_p(pd['g'],type='HG2',b=b_2,c=c_2)
    
    w_1 = pyhapke.get_w(comp_1['n'],comp_1['k'],comp_1['wav'],10**comp_1['log10D'],pd['s'])
    w_2 = pyhapke.get_w(comp_2['n'],comp_2['k'],comp_2['wav'],10**comp_2['log10D'],pd['s'])
    
    model_1 = pyhapke.get_r(w_1,pd['mu'],pd['mu_0'],pd['B'],p_1,pd['K'])
    wav_1 = comp_1['wav']
    
    model_2 = pyhapke.get_r(w_2,pd['mu'],pd['mu_0'],pd['B'],p_2,pd['K'])
    wav_2 = comp_2['wav']
    
    match = utils.spectra_match(model_1,wav_1,model_2,wav_2)

    model_1,model_2,wav_lm = match['new_data_1'],match['new_data_2'],match['wav_common']    
    
    model_lm = comp_1['f']*model_1 + comp_2['f']*model_2
    
    return model_lm, wav_lm 
    
def Z_to_sigma(ln_Z1, ln_Z2):
    """Convert log-evidences of two models to a sigma confidence level
    
    Prameters
    ---------
    ln_Z1: float
        log of Bayesian evidence of model 1
    ln_Z2: float
        log of Bayesian evidence of model 2
        
    Returns
    -------
    
    B: float
        Bayes factor of model 1 to model 2
    sigma: float
        sigma evidence of model 1 over model 2
    """
    np.set_printoptions(precision=50)
    B = np.exp(ln_Z1 - ln_Z2)
    if B < 1.0:
        warnings.warn('Bayes factor is less than 1; sigma-significance is invalid')
    p = np.real(np.exp(W((-1.0/(B*np.exp(1))),-1)))
    sigma = np.sqrt(2)*erfcinv(p)
    #print "p-value = ", p
    #print "n_sigma = ", sigma
    return B, sigma

def get_ML_params(samples):
    """Function to return the maximum likelihood solution from the dynesty samples
    
    Parameter
    ---------
    samples: dynesty.results.Results
        results dictionary of a NestedSampler object    
    """
    
    idx = np.where(samples['logl'] == samples['logl'].max())[0]

    return samples['samples'][idx][0]

def plot_posts(samples, labels,smooth=True, **kwargs):
    """
    Function to plot posteriors using corner.py and scipy's gaussian KDE function.
    
    Parameters
    ----------
    samples: dynesty.results.Results
        results dictionary of a NestedSampler object
    labels: list of str
        list of parameter labels
    smooth: Boolean
        flag to plot gaussian KDE smoothed version of distributions
    **kwargs: 
        truths: list of floats
            list of true values of parameters if using simulated data
    """
    
    # get function that resamples from the nested samples to give sampler with equal weight
    
    # draw posterior samples
    weights = np.exp(samples['logwt'] - samples['logz'][-1])
    samples_mod = resample_equal(samples.samples, weights)

    print('Number of posterior samples is {}'.format(len(samples_mod)))
    
    fig = corner.corner(samples_mod, labels=labels, hist_kwargs={'density': True}, **kwargs)

    # plot KDE smoothed version of distributions
    
    # get indices for diagonal subplots
    
    n = samples_mod.shape[1]
    idx_diag = []
    
    for i in range(n**2):
        if i%(n+1) == 0:
            idx_diag.append(i)
    
    if smooth:
        for axidx, samps in zip(idx_diag, samples_mod.T):
            kde = gaussian_kde(samps)
            xvals = fig.axes[axidx].get_xlim()
            xvals = np.linspace(xvals[0], xvals[1], 100)
            fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')


def best_fit_2comp(samples,param_dict,Nspectra):
    """Function to generate spectra from parameters of a subset of samples drawn from the posterior for pyhapke.two_comp_model(). These spectra can also be then summarized with their median, 1-, 2-, sigma confidence intervals.
    NOTE: Its assumed that the free parameters are param_dict['comp_1']['log10D'], param_dict['comp_2']['log10D'] and 
    param_dict['comp_1']['mf'], in that order.
    
    Parameters
    ----------
    samples: dynesty.results.Results
        results dictionary of a NestedSampler object
    Nspectra: int
        number of samples to be used (default=200)
    param_dict: dict
        input dictionary to the pyhapke.two_comp_model() function
        
    
    Returns
    -------
    
    bf_model_dict: dict
        best fit model dictionray with following keys:
    
        'high_2sig': 1D numpy array
            upper 2-sigma limit of the sampled models
        'high_1sig': 1D numpy array
            upper 1-sigma limit of the sampled models
        'median': 1D numpy array
            median of the sampled models
        'low_1sig': 1D numpy array
            lower 1-sigma limit of the sampled models
        'low_2sig: 1D numpy array
            lower 2-sigma limit of the sampled models
        'wav': 1D numpy array
            wavelength axis for the best fit model arrays
    """
    
    # weigh the posterior samples for appropriate random drawing
    samp, wts = samples.samples, np.exp(samples.logwt - samples.logz[-1])
    samples2 = dynesty.utils.resample_equal(samp, wts)

    # choose random indicies to draw from properly weighted posterior samples
    draws=np.random.randint(0, samples2.shape[0], Nspectra)

    model_list=[]
    
    for i in range(Nspectra):
    
        log10D_1,log10D_2,f_1 = samples2[draws[i],:] # read the sample array
        
        f_2 = 1 - f_1
        # update the parameter dictionary
        
        param_dict['comp_1']['log10D'] = log10D_1
        param_dict['comp_2']['log10D'] = log10D_2
        param_dict['comp_1']['f'] = f_1
        param_dict['comp_2']['f'] = f_2
        
        model, wav_model = pyhapke.two_comp_model(param_dict)

        model_list.append(model)
    
    # Use the last 'wav_model' from the for loop as the wavelength axis for the models


    model_median=np.zeros(wav_model.shape[0])
    model_high_1sig=np.zeros(wav_model.shape[0])
    model_high_2sig=np.zeros(wav_model.shape[0])
    model_low_1sig=np.zeros(wav_model.shape[0])
    model_low_2sig=np.zeros(wav_model.shape[0])

    # calculate the median, 1-, and 2-, sigma confidence intervals

    for i in range(wav_model.shape[0]):
        val_set = np.array([model_list[j][i] for j in range(Nspectra)])
        percentiles=np.percentile(val_set,[4.55, 15.9, 50, 84.1, 95.45])
        model_low_2sig[i]=percentiles[0]
        model_low_1sig[i]=percentiles[1]
        model_median[i]=percentiles[2]
        model_high_1sig[i]=percentiles[3]
        model_high_2sig[i]=percentiles[4]    
    
    bf_model_dict = {'high_2sig':model_high_2sig,'high_1sig':model_high_1sig,
                     'median':model_median,'low_1sig': model_low_1sig,
                     'low_2sig':model_low_2sig,'wav':wav_model}
    
    return bf_model_dict


def best_fit_1comp(samples,param_dict,Nspectra):
    """Function to generate spectra from parameters of a subset of samples drawn from the posterior for pyhapke.one_comp_model(). These spectra can also be then summarized with their median, 1-, 2-, sigma confidence intervals.
    NOTE: Its assumed that the free parameter is param_dict['comp']['log10D']
    
    Parameters
    ----------
    samples: dynesty.results.Results
        results dictionary of a NestedSampler object
    Nspectra: int
        number of samples to be used (default=200)
    param_dict: dict
        input dictionary to the pyhapke.two_comp_model() function
        
    
    Returns
    -------
    
    bf_model_dict: dict
        best fit model dictionray with following keys:
    
        'high_2sig': 1D numpy array
            upper 2-sigma limit of the sampled models
        'high_1sig': 1D numpy array
            upper 1-sigma limit of the sampled models
        'median': 1D numpy array
            median of the sampled models
        'low_1sig': 1D numpy array
            lower 1-sigma limit of the sampled models
        'low_2sig: 1D numpy array
            lower 2-sigma limit of the sampled models
        'wav': 1D numpy array
            wavelength axis for the best fit model arrays
    """
    
    # weigh the posterior samples for appropriate random drawing
    samp, wts = samples.samples, np.exp(samples.logwt - samples.logz[-1])
    samples2 = dynesty.utils.resample_equal(samp, wts)

    # choose random indicies to draw from properly weighted posterior samples
    draws=np.random.randint(0, samples2.shape[0], Nspectra)

    model_list=[]
    
    for i in range(Nspectra):
    
        log10D = samples2[draws[i],:] # read the sample array

        # update the parameter dictionary
        
        param_dict['comp']['log10D'] = log10D

        model, wav_model = pyhapke.one_comp_model(param_dict)

        model_list.append(model)
    
    # Use the last 'wav_model' from the for loop as the wavelength axis for the models


    model_median=np.zeros(wav_model.shape[0])
    model_high_1sig=np.zeros(wav_model.shape[0])
    model_high_2sig=np.zeros(wav_model.shape[0])
    model_low_1sig=np.zeros(wav_model.shape[0])
    model_low_2sig=np.zeros(wav_model.shape[0])

    # calculate the median, 1-, and 2-, sigma confidence intervals

    for i in range(wav_model.shape[0]):
        val_set = np.array([model_list[j][i] for j in range(Nspectra)])
        percentiles=np.percentile(val_set,[4.55, 15.9, 50, 84.1, 95.45])
        model_low_2sig[i]=percentiles[0]
        model_low_1sig[i]=percentiles[1]
        model_median[i]=percentiles[2]
        model_high_1sig[i]=percentiles[3]
        model_high_2sig[i]=percentiles[4]    
    
    bf_model_dict = {'high_2sig':model_high_2sig,'high_1sig':model_high_1sig,
                     'median':model_median,'low_1sig': model_low_1sig,
                     'low_2sig':model_low_2sig,'wav':wav_model}
    
    return bf_model_dict
    
def plot_bestfit(bf_model_dict, data,wav_data,data_err,data_label, plot_title,y_label,x_label=r'$\rm Wavelength \ in \ \mu m$',model_orig=None,wav_model_orig=None,zoom=False):
    """Function to plot the best-fit model and its 1-, 2-, sigma confidence intervals along with the data.
    
    Parameters
    ----------
    
    data: 1D numpy array
        actual/observed I/F spectrum
    wav_data: 1D numpy array
        corresponding wavelength axis in microns
    data_err: 1D numpy array
        error on the data
    bf_model_dict: dict
        best fit model dictionray with following keys:
    
        'high_2sig': 1D numpy array
            upper 2-sigma limit of the sampled models
        'high_1sig': 1D numpy array
            upper 1-sigma limit of the sampled models
        'median': 1D numpy array
            median of the sampled models
        'low_1sig': 1D numpy array
            lower 1-sigma limit of the sampled models
        'low_2sig: 1D numpy array
            lower 2-sigma limit of the sampled models
        'wav': 1D numpy array
            wavelength axis for the best fit model arrays
    data_label: str
        label to use for data when plotting
    plot_title: str
        title for the plot
    model_orig: 1D numpy array (optional)
        original model spectrum from which synthetic data was generated 
    wav_model_orig: 1D numpy array (optional)
        corresponding wavelength array
    """
    
    wav_model = bf_model_dict['wav']
    model_high_1sig = bf_model_dict['high_1sig']
    model_high_2sig = bf_model_dict['high_2sig']
    model_low_1sig = bf_model_dict['low_1sig']
    model_low_2sig = bf_model_dict['low_2sig']
    model_bf = bf_model_dict['median']
    
    fig1, ax1=plt.subplots(figsize=(9,6))
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    ax1.fill_between(wav_model,model_low_2sig,model_high_2sig,facecolor='r',alpha=0.5,edgecolor='None',label=r'$1-\sigma$')  
    ax1.fill_between(wav_model,model_low_1sig,model_high_1sig,facecolor='r',alpha=1.,edgecolor='None',label=r'$2-\sigma$') 

    ax1.errorbar(wav_data, data,yerr=data_err, xerr=0,marker='o',ls='none', 
                alpha=0.5, label=data_label)
    ax1.plot(wav_model,model_bf,label='best-fit model')
    if np.all(model_orig) != None:
        ax1.plot(wav_model_orig,model_orig,label='original model')
    ax1.legend(prop={'size':8})
    plt.title(plot_title)
    
    if zoom:
    
        fig1, (ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)

        ax1.fill_between(wav_model,model_low_2sig,model_high_2sig,facecolor='r',alpha=0.5,edgecolor='None',label=r'$1-\sigma$')  
        ax1.fill_between(wav_model,model_low_1sig,model_high_1sig,facecolor='r',alpha=1.,edgecolor='None',label=r'$2-\sigma$') 

        ax1.errorbar(wav_data, data,yerr=data_err, xerr=0,marker='o',color='r',ls='none', 
                    alpha=0.5, label=data_label)
        ax1.plot(wav_model,model_bf,label='best-fit model',color='b')
        if np.all(model_orig) != None:
            ax1.plot(wav_model_orig,model_orig,label='original model',color='g')
        ax1.legend(prop={'size':8})

        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)

        ax2.fill_between(wav_model,model_low_2sig,model_high_2sig,facecolor='r',alpha=0.5,edgecolor='None',label=r'$1-\sigma$')  
        ax2.fill_between(wav_model,model_low_1sig,model_high_1sig,facecolor='r',alpha=1.,edgecolor='None',label=r'$2-\sigma$') 

        ax2.errorbar(wav_data, data,yerr=data_err, xerr=0,marker='o',color='r',ls='none', 
                    alpha=0.5, label=data_label)
        ax2.plot(wav_model,model_bf,label='best-fit model',color='b')
        if np.all(model_orig) != None:
            ax2.plot(wav_model_orig,model_orig,label='original model',color='g')
        ax2.legend(prop={'size':8})
        ax2.set_xlim((4.5,4.6))
        ax2.set_ylim((0.0,0.2))
        plt.suptitle(plot_title)
    #plt.tight_layout()
    
    
    
def save_samples(samples,db_update= True, file_name=None, folder_path='Samples/',db_path='Samples/samples_db.csv'):
    """Function to store the samples of the posterior function generated in a Bayesian retrieval run. 
    Samples can be from dynesty, emcee, etc. The samples meta-data table (given by db_path) is also updated 
    with user-input details.
    
    Parameters
    ----------
    samples: sampler object
        dynesty.results.Results object for dynesty.
        emcee.EnsembleSampler object for emcee. 
    file_name: str (default vlaue is None)
        name of the file. Default is 'None' because its automatically generated below.
    folder_path: str
        path to folder where samples are supposed to be stored 
    db_path: str
        path to the samples metadata database. 
    """
    
    # if no file name is supplied, generate and print file name using the current data and time
    
    if not(file_name):
    
        file_name = time.strftime("%Y%m%d-%H%M%S")
        file_name = file_name + '.pic'
        print('The sampler object has been saved as ',file_name)
        
    else:
        print('The sampler object has been saved as ',file_name)
    
    file_path = folder_path + file_name
    
    f = open(file_path,'wb')
    
    pickle.dump(samples,f) 
    
    f.close()
    
    
    # update the table at db_path
    
    if db_update:
        ud_samplesdb(file_name,db_path)
    


def ud_samplesdb(samples_file,db_path='Samples/samples_db.csv'):
    """The samples meta-data table (given by db_path) is also updated 
    with user-input details.
    
    Parameters
    ----------
    samples_file: str
        name of the pickled file that contains the samples
    db_path: str
        path to the samples database file
    """
    
    df = pd.read_csv(db_path)
    
    # ask for user inputs
    
    object_type = input ("Object type (eg. dynesty.results.Results,emcee.EnsembleSampler): ")
    sampler_type = input ("Sampler type (eg. dynesty, emcee, etc.): ")
    free_params = input ("Free params (eg. log10D_am, phi): ")
    priors = input ("prior ranges (eg. (1.,3.),(0.01,0.52)): ")
    data = input ("info. on data (eg. Mean JM0081, wav:2-5 microns): ")
    data_error = input ("info. on error bars (eg. RMS, inflated 10x): ")
    notes = input ("any misc. notes (intimate v/s linear, etc.): ")
    
    df2 = {'samples file': samples_file, 'object type': object_type, 'sampler type': sampler_type,
           'free params': free_params, 'priors': priors,'data': data, 'data error':data_error,
          'notes':notes}
    df = df.append(df2, ignore_index=True)
    df.to_csv('Samples/samples_db.csv',index=False)

'''    
def plot_bestfit(samples,data,wav_data,data_err,model, Nspectra=200,**kwargs):
    """Function to generate spectra from parameters of a subset of samples drawn from the posterior. These spectra can also be then
    summarized with their median, 1-, 2-, sigma confidence intervals.
    
    Parameters
    ----------
    samples: dynesty.results.Results
        results dictionary of a NestedSampler object
    Nspectra: int
        number of samples to be used (default=200)
    data: 1D numpy array
        actual/observed I/F spectrum
    wav_data: 1D numpy array
        corresponding wavelength axis in microns
    data_err: 1D numpy array
        error on the data
    model: function
        the function to generate the model
    **kwargs: additional arguments for the 'model' function
        'param_dict': dictionary of model parameters
        'free_keys': list of tuples that are keys in param_dict coresponding to the free parameters.
        A tuple of length 2 would correspond to a nested dictionary. Example:
        
        >>> free_keys = [('P'),('B'),('comp_1','mf')]
    """
    # weigh the posterior samples for appropriate random drawing
    samp, wts = samples.samples, np.exp(samples.logwt - samples.logz[-1])
    samples2 = dyfunc.resample_equal(samp, wts)
    
    # choose random indicies to draw from properly weighted posterior samples
    draws=np.random.randint(0, samples2.shape[0], Nspectra)

    model_list=[]
    
    for i in range(Nspectra):
        free_params = samples2[draws[i],:] # read the sample array
        
        # if mass fraction of one component is a free parameter, then add the other mass fraction in the list
        for i,key in free_keys:
            if 'mf' in key and 'comp_1' in key:
                free_keys.append(('comp_2','mf'))
                np.append(free_params,1-free_params
            if 'mf' in key and 'comp_2' in key:
                free_keys.append(('comp_1','mf'))
                               
        
    
        # update the parameter dictionary for the model

        param_dict = {'mu_0':mu_0,'mu':mu,'P':P,'B':B,'comp_1':{'n':n_am,'k':k_am,'wav':wav_am,'log10D':log10D_am,'mf':mf_am},
             'comp_2':{'n':n_cr,'k':k_cr,'wav':wav_cr,'log10D':log10D_cr,'mf':mf_cr}}
        
        param_dict = pyhapke.set_dict_values(param_dict,free_keys,free_params)

        model, wav_model = pyhapke.two_comp_model(param_dict)
    
    
'''    