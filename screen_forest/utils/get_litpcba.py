import os

from rdkit import Chem


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
    return data
