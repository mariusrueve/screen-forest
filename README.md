
# Ligand-Based Virtual Screening CLI

This Python script provides a command-line interface (CLI) for performing ligand-based virtual screening using molecular fingerprints (Morgan fingerprints) and a Random Forest classifier. It leverages the RDKit library for generating molecular fingerprints from SMILES strings and uses scikit-learn for the machine learning model.

## Features

- **Convert SMILES to Fingerprints**: Uses Morgan fingerprints (radius 2, 2048 bits) to represent molecules.
- **Random Forest Classifier**: Train and predict activity of molecules using a Random Forest model.
- **Joblib Model Saving/Loading**: Easily save and load models using the `joblib` package.
- **Active/Inactive Molecule Handling**: Accepts active and inactive molecules as input files for training.

## Requirements

The script depends on the following Python libraries:

- `click`: For building the CLI.
- `joblib`: For saving and loading trained models.
- `numpy`: For handling numerical data.
- `pandas`: For reading and processing datasets.
- `rdkit`: For working with molecular representations.
- `scikit-learn`: For machine learning models, specifically Random Forest.

You can install these dependencies using `pip`:

```bash
pip install click joblib numpy pandas rdkit scikit-learn
```

## Usage

### General CLI Command

The script uses `click` to provide a command-line interface. To access the available commands, run:

```bash
python screen_forest.py --help
```

### Commands

1. **Training the Model**:

   Train a Random Forest classifier using a set of active and inactive molecules in SMILES format.

   ```bash
   python screen_forest.py train --active <active_molecules.smi> --inactive <inactive_molecules.smi> --output <model_output_path>
   ```

   - `--active`: Path to a file containing active molecules in SMILES format.
   - `--inactive`: Path to a file containing inactive molecules in SMILES format.
   - `--output`: Path where the trained model should be saved.

2. **Predicting Activity**:

   Use a pre-trained model to predict the activity of new molecules.

   ```bash
   python screen_forest.py predict --input <molecules.smi> --model <model_path> --output <predictions.csv>
   ```

   - `--input`: Path to a file containing molecules in SMILES format.
   - `--model`: Path to a pre-trained Random Forest model (saved with `joblib`).
   - `--output`: Path to save the prediction results.

## Example

1. **Training**:

   ```bash
   python screen_forest.py train --active data/active_molecules.smi --inactive data/inactive_molecules.smi --output models/random_forest_model.pkl
   ```

2. **Predicting**:

   ```bash
   python screen_forest.py predict --input data/query_molecules.smi --model models/random_forest_model.pkl --output results/predictions.csv
   ```
