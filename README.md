# Adsorption Breakthrough Curve Prediction

This repository contains scripts for generating RUPTURA breakthrough simulation inputs, cleaning simulated breakthrough data, constructing adsorption-derived descriptors, training dual deep-learning models, and reproducing evaluation figures.

## Important Note About RUPTURA

RUPTURA itself is not bundled in this repository. Install and compile it from the official repository first:

```bash
git clone https://github.com/iraspa/ruptura.git
```

After compilation, pass the path to the RUPTURA executable to `ruptura_generation/run_ruptura_batch.py` with `--ruptura_exe`.

## Expected Data Files

The model-training scripts expect cleaned data files named:

- `curve_dataset_clean.csv`: input descriptors, shape `(N, 24)`
- `curve_clean.csv`: breakthrough curves, shape `(N, 300)`

For `curve_clean.csv`, columns 0-99 are the time axis, columns 100-199 are C1 concentrations, and columns 200-299 are C2 concentrations.

Large generated data files and model checkpoints are intentionally excluded from Git by `.gitignore`. Put full data in `data/` locally or provide a Zenodo/Figshare link for publication.

## Workflow

### 1. Generate RUPTURA input folders

```bash
python ruptura_generation/breakthrough.py --base_dir break_binary --num_iterations 50000 --seed 42
```

This creates binary adsorption breakthrough simulation folders containing `params.json` and `simulation.input`.

### 2. Run RUPTURA simulations

```bash
python ruptura_generation/run_ruptura_batch.py break_binary --ruptura_exe /path/to/RUPTURA-main/src/ruptura --max_workers 8
```

On Windows, use the compiled executable path for your local RUPTURA installation.

### 3. Filter valid simulation folders

```bash
python ruptura_generation/filter_folders.py break_binary --output filtered_result_new.csv
```

### 4. Build model-ready curve files

Convert RUPTURA output files into:

- `curve_dataset.csv`
- `curve.csv`

`curve_dataset.csv` contains the 24 simulation/input descriptors. `curve.csv` contains 100 time points plus 100 C1 and 100 C2 concentration values.

```bash
python data_processing/get_curve.py break_binary --filtered_csv filtered_result_new.csv --out_x_csv curve_dataset.csv --out_y_csv curve.csv
```

### 5. Clean abnormal simulations

```bash
python data_processing/clean_dataset.py --x_csv curve_dataset.csv --y_csv curve.csv --out_x_csv curve_dataset_clean.csv --out_y_csv curve_clean.csv
```

The cleaning criteria are `Tmax <= 500` and `Cmax <= 2`.

### 6. Freeze the train, validation, and test split

From the repository root, create a single manifest for both models. Use files in the same row order throughout.

```bash
python split_protocol.py --mode isotherm --x-csv curve_dataset_clean.csv --y-csv curve_clean.csv
```

- `--mode isotherm` groups rows with the same pair of dual-site Langmuir-Freundlich isotherms. When the simulator samples a different isotherm pair for every row, this is effectively a row-level split.
- `--mode system --groups-csv system_ids.csv` holds out whole systems. Supply one system ID per cleaned row, without a header, and use the same ID for all operating conditions of the same system. For example, a system ID can identify a specific adsorbent and gas pair. The synthetic generator does not assign real adsorbent and gas identities.
- `--mode sample` creates a row-level split for an interpolation benchmark.

The default split seed is 42. Twenty percent of groups are reserved for testing, and 15% of the remaining groups are reserved for validation. The row fractions depend on group sizes. The manifest records the split, dataset fingerprint, and group counts. Keep it with the trained artifacts. Use different manifest and output paths for separate benchmark protocols.

### 7. Train the time model

Copy or symlink `curve_dataset_clean.csv` and `curve_clean.csv` into `time_model/`, then run for example:

```bash
cd time_model
python train_time.py --model lstm --features 50 --target total
```

Random search:

```bash
python random_search_time.py --targets total --trials 8 --epochs 500
```

### 8. Train the concentration model

Copy or symlink the same cleaned data into `concentration_model/`, then run for example:

```bash
cd concentration_model
python train_conc.py --model lstm --features 24
```

Random search:

```bash
python random_search_conc.py --trials 8 --epochs 500
```

The training and search scripts read `../split_manifest.json` by default when run from either model directory. Override it with `--split` if needed. Training randomness can vary by trial. The partition stays fixed, feature scalers are fitted on training rows, and training and random search report validation metrics only. Use `--x_csv` and `--y_csv` if the cleaned data are stored elsewhere.

### 9. Export and evaluate the final dual model

From the repository root, select the checkpoints on validation scores, then evaluate the fixed checkpoints on the shared test rows:

```bash
python analysis/export_best_dual_model.py --time_dir time_model --conc_dir concentration_model
python analysis/plot_final_dual_curves.py --final_dir final_dual_model --x_csv curve_dataset_clean.csv --y_csv curve_clean.csv
```

Use `--metrics_only` to write the final `test_metrics.json` without plotting. The exporter rejects legacy search results without the validation split identity. Rerun both searches before generating manuscript results. The RUPTURA holdout, including a grouped holdout, evaluates generalization within simulated data. Experimental breakthrough curves are a separate external benchmark.

## Feature Sets

- 24 features: original column, gas, transport, and Langmuir-Freundlich parameters
- 38 features: original L-F parameters replaced by uptake values at 13 pressure points for each component
- 50 features: original 24 features plus 26 uptake features

The pressure points are `10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000 Pa`.

## RUPTURA Reference

Sharma, S.; Balestra, S. R. G.; Baur, R.; Agarwal, U.; Zuidema, E.; Rigutto, M. S.; Calero, S.; Vlugt, T. J. H.; Dubbeldam, D. RUPTURA: Simulation Code for Breakthrough, Ideal Adsorption Solution Theory Computations, and Fitting of Isotherm Models. *Molecular Simulation* **2023**, *49*, 893-953. https://doi.org/10.1080/08927022.2023.2202757

## AI-Assisted Coding Disclosure

Some data-processing and model-training scripts were drafted with assistance from an AI-based coding tool. All code, analyses, and interpretations were reviewed, revised, and validated by the authors.
