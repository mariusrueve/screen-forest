import pathlib

from ligand_screening_tool import LigandScreeningTool

# suppress rdkit warnings
from rdkit import RDLogger
from sklearn.metrics import accuracy_score
from utils.get_litpcba import get_litpcba

RDLogger.DisableLog("rdApp.*")

# lit-pcba path
path_to_litpcba = pathlib.Path("lit-pcba")

# get all targets from the lit-pcba dataset ie the folder names
targets = ["KAT2A"]

# for each target train a model and predict the validation data
for target in targets:
    print(f"Target: {target}")
    data = get_litpcba(path_to_litpcba, target)
    active_molecules = data["T"]["active"]
    inactive_molecules = data["T"]["inactive"]
    model = LigandScreeningTool(active_molecules, inactive_molecules)
    accuracy = model.train()
    print(f"Accuracy on training data: {accuracy}")
    print("Predictions:")
    validation_active_molecules = data["V"]["active"]
    validation_inactive_molecules = data["V"]["inactive"]
    validation_data = []
    for molecule in validation_active_molecules:
        validation_data.append([molecule, 1])
    for molecule in validation_inactive_molecules:
        validation_data.append([molecule, 0])
    X, y = zip(*validation_data)
    X = list(X)
    y = list(y)
    predictions = []
    for molecule in X:
        predictions.append(model.predict(molecule))
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy on validation data: {accuracy}")
    print()
