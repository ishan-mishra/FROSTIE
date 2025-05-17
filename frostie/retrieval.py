# module for Bayesian inference of data

from multiprocessing import Pool
import multiprocessing
import functools
import pickle
import os
import time

import numpy as np
import spectres
from .hapke import regolith
from .utils import load_co2_op_cons, load_water_op_cons, Z_to_sigma
from .plotting import plot_spectrum_with_uncertainty, plot_posteriors

from dynesty import NestedSampler
from dynesty import utils as dyfunc


def load_simulated_data(snr=50, del_wav=0.01, f_h2o=0.5):
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

    f_co2 = 1.0 - f_h2o

    water = {'name': 'water', 'n': n_water, 'k': k_water, 'wav': wav_water,
             'D': 100, 'p_type': 'HG2', 'f': f_h2o}

    co2 = {'name': 'carbon dioxide', 'n': n_co2, 'k': k_co2, 'wav': wav_co2,
           'D': 100, 'p_type': 'HG2', 'f': f_co2}

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

class solo_retrieval:
    def __init__(self):
        self.initialized = False

    def initialize(self, fixed_params, free_params, components, data, instrument_response=None):
        """
        Initialize the retrieval setup.

        Parameters
        ----------
        fixed_params : dict
            Dictionary of fixed model parameters (i, e, g, B, s, etc.)
        free_params : list
            List of [parameter_name, (min, max)] tuples for sampling
        components : dict
            Dictionary of {component_name: [wav, n, k]} entries
        data : list
            [wavelengths, reflectance, uncertainty]
        instrument_response : ndarray or None
            Optional response function array (must match wavelength grid)
        """
        self.fixed_params = fixed_params
        self.free_params = free_params
        self.instrument_response = instrument_response

        # Extract original data
        wav_data, reflectance, uncertainty = data
        self.wav_data = wav_data
        self.components_original = components

        # Force all interpolation onto the data wavelength grid
        data_list = [reflectance, uncertainty]
        wav_list = [wav_data, wav_data]

        for name, (wav, n, k) in components.items():
            data_list.extend([n, k])
            wav_list.extend([wav, wav])

        matched_list = [np.interp(wav_data, w, d) for d, w in zip(data_list, wav_list)]

        # Reassemble matched data and components
        reflectance_matched = matched_list[0]
        uncertainty_matched = matched_list[1]
        self.data_matched = [wav_data, reflectance_matched, uncertainty_matched]

        components_matched = {}
        idx = 2
        for name in components:
            n_matched = matched_list[idx]
            k_matched = matched_list[idx + 1]
            components_matched[name] = {
                'wav': wav_data,
                'n': n_matched,
                'k': k_matched
            }
            idx += 2

        self.components = components_matched
        self.initialized = True

        # Automatically determine abundance mode
        component_names = list(self.components.keys())
        n_species = len(component_names)
        n_abundances = len([p for p in self.free_params if p[0].startswith("log10f_")])

        if n_species == 1 and n_abundances == 0:
            self.abundance_mode = "one_species"
        elif n_species == 2 and n_abundances == 1:
            self.abundance_mode = "two_species"
        elif n_species > 2 and n_abundances == n_species - 1:
            self.abundance_mode = "dirichlet"
        else:
            raise ValueError("Invalid abundance configuration: expected log10f parameters for one fewer species.")


    def _make_model(self, theta):
        """
        Generate model spectrum for a given theta vector (free parameter values).
        Handles abundance logic, fixed + free parameter resolution, and builds the Hapke model.
        """
        component_names = list(self.components.keys())
        n_species = len(component_names)
        param_dict = {}

         # --- Abundance handling ---
        if self.abundance_mode == "one_species":
            # No abundance parameters needed
            param_dict[f"f_{component_names[0]}"] = 1.0
            theta_offset = 0

        elif self.abundance_mode == "two_species":
            # One free abundance parameter in log10(f), one dependent
            f_name = [p[0] for p in self.free_params if p[0].startswith("log10f_")][0]
            f_free_component = f_name.replace("log10f_", "")
            f_val = 10 ** theta[0]
            other_component = [name for name in component_names if name != f_free_component][0]

            param_dict[f"f_{f_free_component}"] = f_val
            param_dict[f"f_{other_component}"] = 1 - f_val
            theta_offset = 1

        elif self.abundance_mode == "dirichlet":
            # All abundance values are log10(f) and supplied as first N values in theta
            for name, log10f in zip(component_names, theta[:n_species]):
                param_dict[f"f_{name}"] = 10 ** log10f
            theta_offset = n_species

        else:
            raise ValueError(f"Unknown abundance_mode: {self.abundance_mode}")

        # --- Process remaining (non-abundance) parameters ---
        for (name, _), val in zip(self.free_params[theta_offset:], theta[theta_offset:]):
            param_dict[name] = val

        # --- Pull in required fixed parameters (or raise errors) ---
        required_fixed_keys = ['i', 'e', 'g', 'B', 's', 'p']
        for key in required_fixed_keys:
            if key in param_dict:
                continue
            elif key in self.fixed_params:
                param_dict[key] = self.fixed_params[key]
            else:
                raise ValueError(f"Required parameter '{key}' not found in free or fixed parameters.")

        # --- Build the model ---
        model = regolith()
        comp_list = []

        for name in component_names:
            comp = self.components[name]
            f = param_dict.get(f"f_{name}", 1.0)
            D = param_dict.get(f"D_{name}", 100)

            comp_dict = {
                'name': name,
                'wav': comp['wav'],
                'n': comp['n'],
                'k': comp['k'],
                'f': f,
                'D': D,
                'p_type': self.fixed_params.get('p_type', 'HG2')
            }
            comp_list.append(comp_dict)

        model.add_components(comp_list, matched_axes=True)
        model.set_mixing_mode(self.fixed_params.get("mixing_mode", "intimate"))
        model.set_obs_geometry(param_dict['i'], param_dict['e'], param_dict['g'])
        model.set_backscattering(param_dict['B'])
        model.set_s(param_dict['s'])
        model.set_porosity(param_dict['p'])
        model.calculate_reflectance()

        return model.wav_model, model.model


    def _loglike(self, theta, norm_log):
        """
        Compute log-likelihood from model and data.
        Assumes model and data are already matched in wavelength.
        """
        _, model_flux = self._make_model(theta)
        wav_data, data, noise = self.data_matched

        if self.instrument_response is not None:
            model_flux *= self.instrument_response
        else:
            self.instrument_response = np.ones_like(wav_data)

        chisq = -0.5 * np.sum((data - model_flux) ** 2 / noise ** 2)
        return chisq + norm_log


    def _dirichlet_prior_transform(self, theta_unit, prior_bounds, n_species):
        limit = prior_bounds[0][0]  # assume same lower bound for all
        Dir = np.zeros(n_species)
        X = np.zeros(n_species)

        prior_lower = ((n_species - 1) / n_species) * (limit * np.log(10.0) + np.log(n_species - 1.0))
        prior_upper = ((1.0 - n_species) / n_species) * (limit * np.log(10.0))

        for i in range(n_species - 1):
            Dir[1 + i] = theta_unit[i] * (prior_upper - prior_lower) + prior_lower

        if np.abs(np.sum(Dir[1:])) > prior_upper:
            return np.ones(len(theta_unit)) * (-np.inf)

        Dir[0] = -np.sum(Dir[1:])

        if (np.max(Dir) - np.min(Dir)) > (-1.0 * limit * np.log(10.0)):
            return np.ones(len(theta_unit)) * (-np.inf)

        norm = np.sum(np.exp(Dir))
        for i in range(n_species):
            X[i] = np.exp(Dir[i]) / norm
            if X[i] < 10**limit:
                return np.ones(len(theta_unit)) * (-np.inf)

        transformed = list(np.log10(X))
        return transformed

    def _prior_transform(self, theta_unit):
        transformed = []

        # Dirichlet prior if more than 2 components and 'f' is being sampled
        use_dirichlet = any('log10f' in p[0] for p in self.free_params) and len(self.components) > 2
        if use_dirichlet:
            f_params = [p for p in self.free_params if 'log10f' in p[0]]
            dirichlet_part = self._dirichlet_prior_transform(theta_unit, f_params, len(self.components))
            transformed.extend(dirichlet_part)
            offset = len(f_params)
        else:
            offset = 0

        for i, (_, bounds) in enumerate(self.free_params[offset:]):
            lo, hi = bounds
            val = lo + (hi - lo) * theta_unit[offset + i]
            transformed.append(val)

        return tuple(transformed)

    def run(self, save_dir='.', nlive=512, dlogz=0.1,
            sample='rwalk', bound='multi', bootstrap=0,
            use_multicore=False, nproc=None):
        """
        Run nested sampling and save results.
        """

        if not self.initialized:
            raise RuntimeError("Call .initialize() before .run()")

        wav_data, data, noise = self.data_matched
        norm_log = -0.5 * np.sum(np.log(2 * np.pi * noise ** 2))
        ndims = len(self.free_params)
        loglike = functools.partial(self._loglike, norm_log=norm_log)

        if use_multicore and nproc is None:
            nproc = max(1, multiprocessing.cpu_count() - 1)

        if use_multicore:
            print(f"[INFO] solo_retrieval: using {nproc} cores for parallel processing.")
            pool = Pool(processes=nproc)
            sampler = NestedSampler(loglike, self._prior_transform, ndims,
                                    bound=bound, sample=sample, bootstrap=bootstrap,
                                    nlive=nlive, pool=pool, queue_size=nproc)
        else:
            sampler = NestedSampler(loglike, self._prior_transform, ndims,
                                    bound=bound, sample=sample, bootstrap=bootstrap,
                                    nlive=nlive)

        t0 = time.time()
        sampler.run_nested(dlogz=dlogz, print_progress=True)
        t1 = time.time()

        self.sampler = sampler
        self.results = sampler.results
        self.samples = self.results.samples
        self.param_names = [p[0] for p in self.free_params]

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'results.pic'), 'wb') as f:
            pickle.dump(self.results, f)
        with open(os.path.join(save_dir, 'samples.pic'), 'wb') as f:
            pickle.dump(self.samples, f)

        hours, rem = divmod(t1 - t0, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Time taken for retrieval: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        if use_multicore:
            pool.close()
            pool.join()


    def plot_solutions(self, plot_residuals=True, plot_uncertainty=False):
        """
        Plot posterior model fit with optional residuals and uncertainty shading.

        Parameters
        ----------
        plot_residuals : bool
            If True, show residuals below the model plot.
        plot_uncertainty : bool
            If True, shade 1σ and 2σ confidence intervals.
        """

        wav_data, reflectance, uncertainty = self.data_matched

        def model_fn(theta_sample):
            _, model = self._make_model(theta_sample)
            return wav_data, model

        plot_spectrum_with_uncertainty(
            wavelengths=wav_data,
            reflectance=reflectance,
            uncertainty=uncertainty,
            model_function=model_fn,
            samples=self.results,
            plot_residuals=plot_residuals,
            plot_uncertainty=plot_uncertainty,
            title="Posterior Model Fit"
        )

    def plot_posteriors(self, truths=None, n_sigma=2):
        
        plot_posteriors(
            results=self.results,
            param_names=self.param_names,
            transform_log10f=True,
            n_sigma=n_sigma,
            truths=truths
        )


    def print_retrieval_summary(self, n_sigma=2):
        """
        Print summary of median and uncertainty for all parameters,
        using equal-weighted posterior resampling.
        """
        import math
        from dynesty import utils as dyfunc

        lower_q = 0.5 - 0.5 * math.erf(n_sigma / np.sqrt(2))
        upper_q = 0.5 + 0.5 * math.erf(n_sigma / np.sqrt(2))

        # Resample to equal weights
        weights = np.exp(self.results.logwt - self.results.logz[-1])
        samples_equal = dyfunc.resample_equal(self.samples, weights)

        print(f"Retrieved parameters (±{n_sigma}σ intervals):")

        for i, name in enumerate(self.param_names):
            vals = samples_equal[:, i]
            median = np.median(vals)
            lo = np.percentile(vals, 100 * lower_q)
            hi = np.percentile(vals, 100 * upper_q)

            if name.startswith("log10f_"):
                f_median = 10 ** median
                f_lo = 10 ** lo
                f_hi = 10 ** hi
                print(f"{name}: {median:.3f} (+{hi - median:.3f} / -{median - lo:.3f}) "
                    f"=> f = {f_median:.3f} (+{f_hi - f_median:.3f} / -{f_median - f_lo:.3f})")
            else:
                print(f"{name}: {median:.3f} (+{hi - median:.3f} / -{median - lo:.3f})")


class nested_retrieval:
    def __init__(self):
        self.results_dict = {}
        self.model_logZs = {}
        self.model_names = []
        self.best_model_name = None

    def initialize(self, fixed_params, free_params, components, data, instrument_response=None):
        self.fixed_params = fixed_params
        self.free_params = free_params
        self.components = components
        self.data = data
        self.instrument_response = instrument_response

    def _build_model_case(self, include_components):
        """
        Dynamically build model-specific free parameter list.
        """
        comp_subset = {k: v for k, v in self.components.items() if k in include_components}
        n_species = len(include_components)

        free_subset = []

        for param, bounds in self.free_params:
            if param == "log10f":
                if n_species == 1:
                    continue  # skip abundance param in one-species case
                elif n_species == 2:
                    # give log10f to the first species only
                    species = include_components[0]
                    free_subset.append([f"log10f_{species}", bounds])
                elif n_species > 2:
                    for species in include_components[:-1]:  # leave 1 dependent
                        free_subset.append([f"log10f_{species}", bounds])
            elif param == "D":
                for species in include_components:
                    free_subset.append([f"D_{species}", bounds])
            else:
                free_subset.append([param, bounds])  # global parameter like 'p'

        return comp_subset, free_subset
    
    def run_all_models(self, use_multicore=False, nproc=None,
                    save_dir='nested_results', nlive=512, dlogz=0.1,
                    sample='rwalk', bound='multi', bootstrap=0):

        if use_multicore and nproc is None:
            import multiprocessing
            nproc = max(1, multiprocessing.cpu_count() - 1)

        if use_multicore:
            print(f"[INFO] nested_retrieval: using {nproc} cores for parallel processing.")

        os.makedirs(save_dir, exist_ok=True)

        all_species = list(self.components.keys())
        self.model_names = ['_'.join(all_species)]
        model_list = [all_species] + [[s for s in all_species if s != exclude] for exclude in all_species]

        for model_species in model_list:
            model_name = '_'.join(model_species)
            print(f"\n>>> Running retrieval for model: {model_name}")

            comp_subset, free_subset = self._build_model_case(model_species)

            sr = solo_retrieval()
            sr.initialize(
                fixed_params=self.fixed_params,
                free_params=free_subset,
                components=comp_subset,
                data=self.data,
                instrument_response=self.instrument_response
            )

            model_dir = os.path.join(save_dir, model_name)
            sr.run(save_dir=model_dir,
                use_multicore=use_multicore,
                nproc=nproc,
                nlive=nlive,
                dlogz=dlogz,
                sample=sample,
                bound=bound,
                bootstrap=bootstrap)

            self.results_dict[model_name] = sr
            self.model_logZs[model_name] = sr.results.logz[-1]

        self.best_model_name = max(self.model_logZs, key=self.model_logZs.get)


    def compare_evidences(self, max_display_sigma=10.0, max_display_logz=100.0):
        print("\n=== Bayesian Evidence Comparison ===")
        full_model = self.model_names[0]
        full_logZ = self.model_logZs[full_model]

        for model_name, logZ in self.model_logZs.items():
            if model_name == full_model:
                continue
            B, sigma = Z_to_sigma(full_logZ, logZ)
            delta_logZ = full_logZ - logZ

            # Clean up display
            sigma_disp = f">{max_display_sigma:.1f}" if sigma > max_display_sigma else f"{sigma:.2f}"
            deltaZ_disp = f">{max_display_logz:.2f}" if delta_logZ > max_display_logz else f"{delta_logZ:.2f}"

            excluded = set(full_model.split('_')) - set(model_name.split('_'))
            species = excluded.pop() if excluded else 'UNKNOWN'
            print(f"Evidence for: {species} → {sigma_disp}σ (ΔlogZ = {deltaZ_disp})")

        print(f"\nBest model: {self.best_model_name} (logZ = {self.model_logZs[self.best_model_name]:.2f})")


    def plot_best_model_solutions(self, plot_residuals=True, plot_uncertainty=False):
        best = self.results_dict[self.best_model_name]
        wav_data, reflectance, uncertainty = best.data_matched

        def model_fn(theta_sample):
            _, model = best._make_model(theta_sample)
            return wav_data, model

        plot_spectrum_with_uncertainty(
            wavelengths=wav_data,
            reflectance=reflectance,
            uncertainty=uncertainty,
            model_function=model_fn,
            samples=best.results,
            title=f"Best Model: {self.best_model_name}",
            plot_residuals=plot_residuals,
            plot_uncertainty=plot_uncertainty
        )

    def plot_best_model_posteriors(self, n_sigma=2, truths=None):
        best = self.results_dict[self.best_model_name]
        plot_posteriors(best.results, param_names=best.param_names,
                        transform_log10f=True, n_sigma=n_sigma, truths=truths)

    def print_best_model_summary(self, n_sigma=2):
        best = self.results_dict[self.best_model_name]
        best.print_retrieval_summary(n_sigma=n_sigma)

    def plot_model(self, name, plot_residuals=True, plot_uncertainty=False):
        """
        Plot solutions for any named model in the comparison set.

        Parameters
        ----------
        name : str
            Model name (e.g. 'h2o', 'co2', 'h2o_co2').
        plot_residuals : bool
            Whether to show residual panel.
        plot_uncertainty : bool
            Whether to shade 1σ / 2σ confidence bounds.
        """
        if name not in self.results_dict:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self.results_dict.keys())}")

        sr = self.results_dict[name]
        wav_data, reflectance, uncertainty = sr.data_matched

        def model_fn(theta_sample):
            _, model = sr._make_model(theta_sample)
            return wav_data, model

        plot_spectrum_with_uncertainty(
            wavelengths=wav_data,
            reflectance=reflectance,
            uncertainty=uncertainty,
            model_function=model_fn,
            samples=sr.results,
            title=f"Model: {name}",
            plot_residuals=plot_residuals,
            plot_uncertainty=plot_uncertainty
        )

    def plot_model_posteriors(self, name, n_sigma=2, truths=None):
        """
        Plot posterior distributions for any named model.

        Parameters
        ----------
        name : str
            Model name to plot.
        n_sigma : float
            Confidence interval width.
        truths : list or None
            True values to overlay.
        """
        if name not in self.results_dict:
            raise ValueError(f"Model '{name}' not found.")
        sr = self.results_dict[name]
        plot_posteriors(sr.results, param_names=sr.param_names,
                        transform_log10f=True, n_sigma=n_sigma, truths=truths)




