Advanced Lorentz--Mahalanobis Simulation Package
================================================

Files:
1. LM_Advanced_Simulation_Colab.ipynb
   - Open this in Google Colab and run all cells.
   - The final cell runs:
       results = run_all_simulations(quick=True, make_zip=True)
   - For manuscript-quality results, change quick=True to quick=False.

2. lm_advanced_simulation_colab.py
   - Same code as a plain Python file.
   - In Colab you can run:
       %run lm_advanced_simulation_colab.py
       results = run_all_simulations(quick=True, make_zip=True)

Outputs created by the code:
- lm_advanced_simulation_outputs/figures/*.png
- lm_advanced_simulation_outputs/figures/*.pdf
- lm_advanced_simulation_outputs/tables/*.csv
- lm_advanced_simulation_outputs/tables/*.tex
- lm_advanced_simulation_outputs.zip

Recommended manuscript figures:
- fig00_equivariance_diagnostic.pdf
- fig01_size_under_boost.pdf
- fig02b_size_adjusted_power_heatmap.pdf
- fig05_robust_power.pdf
- fig06_robust_u_error.pdf
- fig07_classification_stability.pdf
- fig08_classification_effective_separation.pdf

Important:
The script separates nominal power and size-adjusted power. Use the size-adjusted
power table/figure in the paper, because a method with inflated type-I error
should not be credited for high power.
