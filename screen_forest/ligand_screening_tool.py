import joblib
import numpy as np
from rdkit.Chem import AllChem, Crippen, Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score


class LigandScreeningTool:
    def __init__(
        self,
        active_molecules,  # rdkit molecules
        inactive_molecules,  # rdkit molecules
        model=None,
    ):
        self.active_molecules = active_molecules
        self.inactive_molecules = inactive_molecules
        self.model = model if model else RandomForestClassifier(n_estimators=100, random_state=42)
        self.params = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }
        self.X = None
        self.y = None
        self._get_data()

    def _get_features(self, molecules):
        """Calculate features for each molecule, i.e. molecular descriptors and fingerprints."""
        features = []
        for molecule in molecules:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
            fingerprint_array = np.array(fingerprint)
            descriptors = [
                Descriptors.MolWt(molecule),
                Descriptors.TPSA(molecule),
                Descriptors.NumHDonors(molecule),
                Descriptors.NumHAcceptors(molecule),
                Crippen.MolLogP(molecule),
            ]
            descriptors_array = np.array(descriptors)
            fingerprint = np.concatenate([fingerprint_array, descriptors_array])
            features.append(fingerprint)
        return features

    def _get_data(self):
        active_features = self._get_features(self.active_molecules)
        inactive_features = self._get_features(self.inactive_molecules)
        X = active_features + inactive_features
        y = [1] * len(active_features) + [0] * len(inactive_features)
        self.X = np.array(X)
        self.y = np.array(y)

    def train(self):
        # train multiple models and return the best one
        grid_clf = GridSearchCV(self.model, self.params, cv=5)
        grid_clf.fit(self.X, self.y)
        self.model = grid_clf.best_estimator_
        accuracy = cross_val_score(self.model, self.X, self.y, cv=5).mean()

        return accuracy

    def predict(self, molecule):
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        fingerprint_array = np.array(fingerprint)
        descriptors = [
            Descriptors.MolWt(molecule),
            Descriptors.TPSA(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumHAcceptors(molecule),
            Crippen.MolLogP(molecule),
        ]
        descriptors_array = np.array(descriptors)
        return self.model.predict([np.concatenate([fingerprint_array, descriptors_array])])[0]

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
