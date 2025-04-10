# module for Bayesian inference of data

import numpy as np
import spectres
from .hapke import regolith
from .utils import load_co2_op_cons, load_water_op_cons

def load_simulated_data(snr=50, del_wav=0.01):
    """
    Generate and return a simulated noisy spectrum of a water & CO2 ice mixture,
    at a specified signal-to-noise ratio (SNR) and spectral resolution.

    Parameters
    ----------
    snr : float
        Desired signal-to-noise ratio of the simulated data.
    del_wav : float
        Spectral resolution (wavelength bin size in microns) to which the 
        model spectrum will be downsampled.

    Returns
    -------
    data_all : dict
        Dictionary containing:
        - 'wavelengths' : ndarray
            Binned wavelength array (microns)
        - 'reflectance' : ndarray
            Simulated reflectance data with Gaussian noise
        - 'uncertainty' : ndarray
            1-sigma uncertainty array corresponding to SNR
    """

    # 1. Create two-component Hapke model (H2O + CO2)
    example_two_regolith = regolith()

    wav_water, n_water, k_water = load_water_op_cons()
    wav_co2, n_co2, k_co2 = load_co2_op_cons()

    f = 0.5  # equal fraction by number

    water = {'name': 'water', 'n': n_water, 'k': k_water, 'wav': wav_water,
             'D': 100, 'p_type': 'HG2', 'f': f}

    co2 = {'name': 'carbon dioxide', 'n': n_co2, 'k': k_co2, 'wav': wav_co2,
           'D': 100, 'p_type': 'HG2', 'f': f}

    example_two_regolith.add_components([water, co2], matched_axes=False)
    example_two_regolith.set_mixing_mode('intimate')
    example_two_regolith.set_obs_geometry(i=45, e=45, g=90)
    example_two_regolith.set_porosity(p=0.9)
    example_two_regolith.set_backscattering(B=0)
    example_two_regolith.set_s(s=0)
    example_two_regolith.calculate_reflectance()

    wav_model = example_two_regolith.wav_model
    model_true = example_two_regolith.model

    # 2. Bin to desired spectral resolution
    buffer = 1e-3
    wav_binned = np.linspace(wav_model.min() + buffer, wav_model.max() - buffer,
                         num=int((wav_model.max() - wav_model.min()) / del_wav))
    # Clip just in case
    wav_binned = np.clip(wav_binned, wav_model.min(), wav_model.max())
    
    model_binned = spectres.spectres(wav_binned, wav_model, model_true)


    # Remove NaNs from interpolation
    mask = ~np.isnan(model_binned)
    wav_binned = wav_binned[mask]
    model_binned = model_binned[mask]

    # 3. Add Gaussian noise based on SNR
    uncertainty = model_binned / snr
    np.random.seed(42)  # for reproducibility
    noisy_data = model_binned + uncertainty * np.random.randn(model_binned.size)

    # 4. Package output
    data_all = {
        'wavelengths': wav_binned,
        'reflectance': noisy_data,
        'uncertainty': uncertainty,
        'wav_model': wav_model,
        'model_true': model_true,
    }

    return data_all