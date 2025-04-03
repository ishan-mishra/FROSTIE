import numpy as np
import frostie.hapke as hapke
import frostie.utils as utils

def test_hapke():
    # Step 1: Generate the model
    example_regolith = hapke.regolith()
    wav_water, n_water, k_water = utils.load_water_op_cons()
    water = {
        'name': 'water',
        'n': n_water,
        'k': k_water,
        'wav': wav_water,
        'D': 100,
        'p_type': 'HG2'
    }
    example_regolith.add_components([water])
    example_regolith.set_obs_geometry(i=45, e=45, g=90)
    example_regolith.set_porosity(p=0.9)
    example_regolith.set_backscattering(B=0)
    example_regolith.set_s(s=0)
    example_regolith.calculate_reflectance()

    # Step 2: Load the reference spectrum
    reference = np.loadtxt("water_spectrum.txt")
    wav_ref = reference[:, 0]
    refl_ref = reference[:, 1]

    # Step 3: Compute and print max differences before assertions
    max_refl_diff = np.max(np.abs(example_regolith.model - refl_ref))
    
    print(f"Max reflectance difference: {max_refl_diff:.3e}")

    # Step 4: Assert match
    assert np.allclose(example_regolith.wav_model, wav_ref, rtol=1e-5, atol=1e-8), "Wavelength mismatch"
    assert np.allclose(example_regolith.model, refl_ref, rtol=1e-5, atol=1e-8), "Reflectance mismatch"

    print("Hapke model test passed!")

if __name__ == "__main__":
    test_hapke()
