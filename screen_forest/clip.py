import pathlib

import click
from ligand_screening_tool import LigandScreeningTool
from rdkit import Chem, RDLogger
from sklearn.metrics import accuracy_score

RDLogger.DisableLog("rdApp.*")

"""
One cli command for training and predicting. The train command takes active and inactive molecules as input and
trains a model and saves it to a file if specified.
The predict command takes a model and a list of molecules and predicts the activity of each molecule.
"""


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-a",
    "--active",
    help="Path to the file containing active molecules in SMILES format",
    required=True,
    type=pathlib.Path,
)
@click.option(
    "-i",
    "--inactive",
    help="Path to the file containing inactive molecules in SMILES format",
    required=True,
    type=pathlib.Path,
)
@click.option("-s", "--save", help="Path to save the trained model", required=False, type=pathlib.Path)
def train(active, inactive, save):
    active_molecules = Chem.SmilesMolSupplier(active)
    inactive_molecules = Chem.SmilesMolSupplier(inactive)
    model = LigandScreeningTool(active_molecules, inactive_molecules)
    accuracy = model.train()
    print(f"Accuracy on training data: {accuracy}")
    if save:
        model.save(save)


@cli.command()
@click.option("--model", help="Path to the file containing the trained model")
@click.option("--molecules", help="Path to the file containing molecules in SMILES format")
def predict(model, molecules):
    model = LigandScreeningTool.load(model)
    molecules = Chem.SmilesMolSupplier(molecules)
    for molecule in molecules:
        result = model.predict(molecule)
        if result == 1:
            print(f"{Chem.MolToSmiles(molecule)}: active")
        else:
            print(f"{Chem.MolToSmiles(molecule)}: inactive")


if __name__ == "__main__":
    cli()
