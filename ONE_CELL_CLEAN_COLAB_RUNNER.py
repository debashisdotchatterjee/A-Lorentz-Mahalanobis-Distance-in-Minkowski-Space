# One-cell Colab runner for the clean real-data LM workflow
!pip -q install energyflow scikit-learn h5py tables tqdm

from pathlib import Path
import zipfile, sys, os

# If this ZIP is uploaded to /content, extract it
for z in Path('/content').glob('lm_realdata_clean_methodology_package*.zip'):
    print('Extracting:', z)
    with zipfile.ZipFile(z, 'r') as zf:
        zf.extractall('/content')

# Find package directory
candidates = [
    Path('/content/lm_realdata_clean_methodology_package'),
    Path.cwd(),
]
pkg_dir = None
for d in candidates:
    if (d / 'lm_realdata_analysis_clean.py').exists():
        pkg_dir = d
        break

if pkg_dir is None:
    raise FileNotFoundError('Could not find lm_realdata_analysis_clean.py. Please extract the package ZIP first.')

sys.path.insert(0, str(pkg_dir))
os.chdir(str(pkg_dir))

from lm_realdata_analysis_clean import CLEAN_CONFIG, run_clean_realdata_analysis

# For debugging, uncomment these:
# CLEAN_CONFIG['max_jets_analysis'] = 20000
# CLEAN_CONFIG['permutation_B'] = 199

results = run_clean_realdata_analysis(CLEAN_CONFIG, make_zip=True)
