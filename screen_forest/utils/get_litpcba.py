import os
import random

from rdkit import Chem

random.seed(42)


def get_litpcba(path_to_litpcba, folder_name) -> dict:
    # Check if the specified folder exists
    if not os.path.isdir(os.path.join(path_to_litpcba, folder_name)):
        raise ValueError(
            f"Folder '{folder_name}' does not exist in '{path_to_litpcba}'"
        )

    # Load the files for the specified folder
    data = {}
    for data_type in ["T", "V"]:
        active_molecules = []
        inactive_molecules = []
        for label in ["active", "inactive"]:
            path = os.path.join(
                path_to_litpcba, folder_name, f"{label}_{data_type}.smi"
            )
            mols = Chem.SmilesMolSupplier(path)
            for mol in mols:
                if mol:
                    if label == "active":
                        active_molecules.append(mol)
                    else:
                        inactive_molecules.append(mol)

        data[data_type] = {
            "active": active_molecules,
            "inactive": inactive_molecules,
        }
    # if a set is over 50 molecules, take a random sample of 50
    for data_type in ["T", "V"]:
        for label in ["active", "inactive"]:
            if len(data[data_type][label]) > 50:
                data[data_type][label] = random.sample(
                    data[data_type][label], 50
                )
    return data
