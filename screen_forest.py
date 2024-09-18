import click
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

morgan_fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def mol_to_fp(smile):
    """Convert a SMILES string to a molecular fingerprint."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return np.zeros(2048)  # Return zero vector for invalid molecules
    fp = morgan_fp_generator.GetFingerprint(mol)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


@click.group()
def cli():
    """Ligand-Based Virtual Screening CLI."""
    pass


@cli.command()
@click.option(
    "--active",
    required=True,
    help="Path to file with active molecules (SMILES format).",
)
@click.option(
    "--inactive",
    required=True,
    help="Path to file with inactive molecules (SMILES format).",
)
@click.option(
    "--output-model",
    default="trained_model.pkl",
    help="File to save the trained model.",
)
def train(active, inactive, output_model):
    """Train a model using active and inactive molecules."""
    click.echo("Loading molecules...")
    active_smiles = pd.read_csv(active, header=None, sep=" ")[0].tolist()
    inactive_smiles = pd.read_csv(inactive, header=None, sep=" ")[0].tolist()

    click.echo("Generating fingerprints...")
    active_fps = [mol_to_fp(smile) for smile in active_smiles]
    inactive_fps = [mol_to_fp(smile) for smile in inactive_smiles]

    X = np.array(active_fps + inactive_fps)
    y = np.array([1] * len(active_fps) + [0] * len(inactive_fps))

    click.echo("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, output_model)
    click.echo(f"Model trained and saved to {output_model}")


@cli.command()
@click.option(
    "--model-file", default="trained_model.pkl", help="Path to the trained model file."
)
@click.option(
    "--input",
    "input_file",
    required=True,
    help="File with molecules to screen (SMILES format).",
)
@click.option(
    "--output", default="predictions.csv", help="File to save the predictions."
)
def screen(model_file, input_file, output):
    """Screen new molecules using the trained model."""
    click.echo("Loading model...")
    model = joblib.load(model_file)

    click.echo("Loading molecules...")
    smiles_list = pd.read_csv(input_file, header=None, sep=" ")[0].tolist()

    click.echo("Generating fingerprints...")
    fps = [mol_to_fp(smile) for smile in smiles_list]
    X = np.array(fps)

    click.echo("Predicting activities...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame(
        {"SMILES": smiles_list, "Prediction": predictions, "Probability": probabilities}
    )
    results.to_csv(output, index=False)
    click.echo(f"Predictions saved to {output}")


if __name__ == "__main__":
    cli()
