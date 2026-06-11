Clean Lorentz--Mahalanobis real-data analysis package
=====================================================

This package replaces the earlier verbose real-data workflow with a cleaner,
manuscript-focused version.

Files
-----
1. lm_realdata_analysis_colab.py
   Backend functions from the previous complete real-data workflow.

2. lm_realdata_analysis_clean.py
   Clean, trimmed analysis runner. This is the file you should use.

3. LM_RealData_CLEAN_Colab.ipynb
   Colab notebook with installation/import/run cells.

4. requirements.txt
   Python package requirements.

How to run in Colab
-------------------
1. Upload and unzip this package in Colab.
2. Open LM_RealData_CLEAN_Colab.ipynb.
3. Run cells from top to bottom.

Or run manually:

    !pip -q install energyflow scikit-learn h5py tables tqdm

    from lm_realdata_analysis_clean import CLEAN_CONFIG, run_clean_realdata_analysis

    # Fast debugging option:
    CLEAN_CONFIG["max_jets_analysis"] = 20000
    CLEAN_CONFIG["permutation_B"] = 199

    results = run_clean_realdata_analysis(CLEAN_CONFIG, make_zip=True)

For final paper-quality output, use:

    CLEAN_CONFIG["max_jets_analysis"] = 60000
    CLEAN_CONFIG["permutation_B"] = 999
    results = run_clean_realdata_analysis(CLEAN_CONFIG, make_zip=True)

Output
------
The runner creates:
    lm_realdata_clean_outputs.zip

containing concise figures, CSV tables, LaTeX tables, and metadata.

Why this clean version?
-----------------------
The earlier output was useful but too noisy for a manuscript. This clean version
omits outputs that do not carry useful scientific information, especially raw
Tyler-score magnitudes, since Tyler's shape estimator is scale-free and its
absolute score magnitude is not directly comparable to covariance-based scores.
It also replaces raw boost absolute differences by rank stability and top-1%
overlap, which are much more interpretable on real data.
