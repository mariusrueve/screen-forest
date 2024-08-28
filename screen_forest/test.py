from ligand_screening_tool import LigandScreeningTool
from lit_pcba.lit_pcba import LitPcba

# suppress rdkit warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# import svm from sklearn
from sklearn.svm import SVC

svm_params = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [
        0.1,
        1,
        10,
    ],
    "gamma": ["scale", "auto"],
    "degree": [2, 3, 4],
}

svm = GridSearchCV(SVC(), svm_params, n_jobs=-1)


# get all targets from the lit-pcba dataset ie the folder names
litpcba = LitPcba()
targets = litpcba.get_target_names()

for target in targets:
    # get the data for the target
    data = litpcba.get_data(target, max_number_of_molecules=20)
    train_data = data["T"]
    validation_data = data["V"]

    # get the active and inactive molecules
    active_molecules = train_data["active"]
    inactive_molecules = train_data["inactive"]

    n_active_train = len(active_molecules)
    n_inactive_train = len(inactive_molecules)

    scanner = LigandScreeningTool(active_molecules, inactive_molecules, model=svm)

    # train the model
    scanner.train()

    # get the validation data
    active_molecules = validation_data["active"]
    inactive_molecules = validation_data["inactive"]

    # get the features
    active_features = scanner.get_features(active_molecules)
    inactive_features = scanner.get_features(inactive_molecules)

    # zip the features and labels
    X = active_features + inactive_features
    y = [1] * len(active_features) + [0] * len(inactive_features)

    # predict the labels
    y_pred = scanner.model.predict(X)

    # calculate the accuracy
    accuracy = accuracy_score(y, y_pred)
    print(
        f"Accuracy for target {target}: {accuracy} with {n_active_train} active "
        f"and {n_inactive_train} inactive molecules during training."
    )
