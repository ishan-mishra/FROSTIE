import pytest
import numpy as np
from frostie.retrieval import solo_retrieval, nested_retrieval, load_simulated_data
from frostie import utils

# True values used in simulated data
TRUE_PARAMS = {
    "log10f_h2o": np.log10(0.5),
    "D_h2o": 100,
    "D_co2": 100,
    "p": 0.9
}


def test_simulated_retrieval_accuracy():
    data = load_simulated_data(snr=100, del_wav=0.01, f_h2o=0.5)

    fixed_params = {
        "p_type": "HG2", "i": 45, "e": 45, "g": 90,
        "B": 0, "s": 0, "mixing_mode": "intimate"
    }

    wav_h2o, n_h2o, k_h2o = utils.load_water_op_cons()
    wav_co2, n_co2, k_co2 = utils.load_co2_op_cons()

    components = {
        "h2o": [wav_h2o, n_h2o, k_h2o],
        "co2": [wav_co2, n_co2, k_co2]
    }

    free_params = [
        ["log10f_h2o", (-3, 0)],
        ["D_h2o", (10, 1000)],
        ["D_co2", (10, 1000)],
        ["p", (0.48, 1.0)]
    ]

    sr = solo_retrieval()
    sr.initialize(fixed_params, free_params, components,
                  [data['wavelengths'], data['reflectance'], data['uncertainty']])
    sr.run(use_multicore=False, nlive=256, dlogz=0.1)

    samples = sr.results.samples
    param_names = sr.param_names

    for i, name in enumerate(param_names):
        if name not in TRUE_PARAMS:
            continue
        vals = samples[:, i]
        median = np.median(vals)
        lo = np.percentile(vals, 2.5)
        hi = np.percentile(vals, 97.5)
        truth = TRUE_PARAMS[name]
        assert lo <= truth <= hi, f"{name} = {truth:.3f} is outside 2σ: [{lo:.3f}, {hi:.3f}]"


def test_nested_model_selection():
    data = load_simulated_data(snr=100, del_wav=0.01, f_h2o=0.5)

    fixed_params = {
        "p_type": "HG2", "i": 45, "e": 45, "g": 90,
        "B": 0, "s": 0, "mixing_mode": "intimate"
    }

    wav_h2o, n_h2o, k_h2o = utils.load_water_op_cons()
    wav_co2, n_co2, k_co2 = utils.load_co2_op_cons()

    components = {
        "h2o": [wav_h2o, n_h2o, k_h2o],
        "co2": [wav_co2, n_co2, k_co2]
    }

    free_params = [
        ["log10f", (-3, 0)],  # Primary abundance param
        ["D", (10, 1000)],    # Shared grain size
        ["p", (0.48, 1.0)]    # Porosity
    ]

    nr = nested_retrieval()
    nr.initialize(
        fixed_params=fixed_params,
        free_params=free_params,
        components=components,
        data=[data['wavelengths'], data['reflectance'], data['uncertainty']]
    )

    nr.run_all_models(use_multicore=False, nlive=256, dlogz=0.1)

    assert nr.best_model_name == "h2o_co2", f"Expected best model to be 'h2o_co2', got '{nr.best_model_name}'"

    for model_name in nr.model_logZs:
        if model_name == "h2o_co2":
            continue
        _, sigma = utils.Z_to_sigma(nr.model_logZs["h2o_co2"], nr.model_logZs[model_name])
        assert sigma > 3, f"Bayesian evidence for {model_name} is not significantly worse (σ = {sigma:.2f})"
