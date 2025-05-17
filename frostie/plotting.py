# Module to make nice-looking plots from the retrieval results


import matplotlib.pyplot as plt
import numpy as np
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import math


def plot_spectrum_with_uncertainty(wavelengths, reflectance, uncertainty,
                                    model_function, samples, num_draws=200,
                                    title="Posterior Model Fit",
                                    plot_residuals=True,
                                    plot_uncertainty=False):
    """
    Plot median and confidence regions from posterior samples.

    Parameters
    ----------
    wavelengths : ndarray
        Data wavelength grid (x-axis).
    reflectance : ndarray
        Observed data.
    uncertainty : ndarray
        1σ uncertainties.
    model_function : callable
        Function that takes a sample vector and returns (model_wavelengths, model_spectrum).
    samples : dynesty.results.Results
        Dynesty results object.
    num_draws : int
        Number of spectra to draw from the posterior.
    title : str
        Title for the plot.
    plot_residuals : bool
        If True, add a residuals panel.
    zoomed_inset : bool
        If True, adds a zoomed-in subplot in the upper right corner.
    """
    # Resample posterior
    samp = samples.samples
    weights = np.exp(samples.logwt - samples.logz[-1])
    samples_equal = dyfunc.resample_equal(samp, weights)
    draws = np.random.randint(0, samples_equal.shape[0], num_draws)

    model_stack = []

    for i in draws:
        model_wav, model_flux = model_function(samples_equal[i])
        model_interp = np.interp(wavelengths, model_wav, model_flux)
        model_stack.append(model_interp)

    model_stack = np.array(model_stack)

    # Compute posterior intervals
    model_median = np.median(model_stack, axis=0)
    low_1sig = np.percentile(model_stack, 15.9, axis=0)
    high_1sig = np.percentile(model_stack, 84.1, axis=0)
    low_2sig = np.percentile(model_stack, 4.55, axis=0)
    high_2sig = np.percentile(model_stack, 95.45, axis=0)

    # Set up plot
    fig, axes = plt.subplots(2 if plot_residuals else 1, 1,
                             figsize=(8, 6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]) if plot_residuals else None)
    ax0 = axes[0] if plot_residuals else axes

    if plot_uncertainty:
        ax0.fill_between(wavelengths, low_2sig, high_2sig, alpha=0.2, color="#d62728", label=r"$2\sigma$")
        ax0.fill_between(wavelengths, low_1sig, high_1sig, alpha=0.4, color="#d62728", label=r"$1\sigma$")
    ax0.errorbar(wavelengths, reflectance, yerr=uncertainty, fmt='o', color="#1f77b4", markersize=3, label="Data")
    ax0.plot(wavelengths, model_median, color="#d62728", label="Median Model")
    ax0.set_ylabel("Reflectance", fontsize=14)
    ax0.legend(loc="upper center", ncol=3, fontsize=12)
    ax0.tick_params(labelsize=12)

    # Residuals subplot
    if plot_residuals:
        ax1 = axes[1]
        residuals = reflectance - model_median
        ax1.scatter(wavelengths, residuals, color='black', s=10)
        ax1.axhline(0, ls='--', color='gray')
        ax1.set_xlabel("Wavelength [$\mu$m]", fontsize=14)
        ax1.set_ylabel("Residuals", fontsize=14)
        ax1.tick_params(labelsize=12)
        plt.subplots_adjust(hspace=0)
    else:
        ax0.set_xlabel("Wavelength [$\mu$m]", fontsize=14)


    # Compute reduced chi-squared
    k = samples.samples.shape[1]
    N = len(reflectance)
    chisq = np.sum(((reflectance - model_median) / uncertainty) ** 2)
    red_chisq = chisq / (N - k)

    # Add to title

    full_title = f"{title}   ($\\chi^2_{{\\rm red}}$ = {red_chisq:.2f})"
    ax0.set_title(full_title, fontsize=14)

    plt.tight_layout()
    plt.show()



def plot_posteriors(results, param_names=None, transform_log10f=True, n_sigma=2, truths=None):
    """
    Plot posterior distributions from a dynesty results object.

    Parameters
    ----------
    results : dynesty.results.Results
        Full dynesty results object.
    param_names : list
        Names of the parameters (in order).
    transform_log10f : bool
        Whether to convert log10f_* parameters to linear f.
    n_sigma : float
        Sigma level to define the confidence interval (e.g., 1 for 68%, 2 for 95%).
    truths : list or None
        True parameter values to show on the plot (same length/order as param_names),
        assumed to be in final plotted units (e.g., f, not log10f).
    """
    samples = results.samples

    if transform_log10f and param_names is not None:
        for i, name in enumerate(param_names):
            if name.startswith("log10f_"):
                samples[:, i] = 10 ** samples[:, i]
                param_names[i] = name.replace("log10f_", "f_")

    # Convert sigma to quantile range
    lower_q = 0.5 - 0.5 * math.erf(n_sigma / np.sqrt(2))
    upper_q = 0.5 + 0.5 * math.erf(n_sigma / np.sqrt(2))
    quantiles = [lower_q, 0.5, upper_q]

    fig, axes = dyplot.cornerplot(
        results,
        labels=param_names,
        show_titles=True,
        title_fmt='.3f',
        quantiles=quantiles,
        truths=truths,
        truth_color='#d62728',
        color='#1f77b4'
    )
    plt.tight_layout()
    plt.show()
