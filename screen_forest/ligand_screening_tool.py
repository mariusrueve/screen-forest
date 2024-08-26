import joblib
import numpy as np
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class LigandScreeningTool:
    def __init__(
        self,
        active_molecules,  # rdkit molecules
        inactive_molecules,  # rdkit molecules
        model=None,
    ):
        self.active_molecules = active_molecules
        self.inactive_molecules = inactive_molecules
        self.model = (
            model
            if model
            else RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.data = self._get_data()

    def _get_fingerprints(self, molecules):
        fingerprints = []
        for molecule in molecules:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule, 2, nBits=1024
            )
            fingerprints.append(fingerprint)
        return fingerprints

    def _get_data(self):
        active_fingerprints = self._get_fingerprints(self.active_molecules)
        inactive_fingerprints = self._get_fingerprints(self.inactive_molecules)
        data = []
        for fingerprint in active_fingerprints:
            data.append([fingerprint, 1])
        for fingerprint in inactive_fingerprints:
            data.append([fingerprint, 0])
        return data

    def train(self):
        X, y = zip(*self.data)
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"Accuracy during training: {accuracy}")

    def predict(self, molecule):
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule, 2, nBits=1024
        )
        prediction = self.model.predict([fingerprint])
        return prediction[0]

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
