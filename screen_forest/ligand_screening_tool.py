import joblib
import numpy as np
from rdkit.Chem import AllChem, Crippen, Descriptors
from sklearn.ensemble import RandomForestClassifier


class LigandScreeningTool:
    def __init__(
        self,
        active_molecules,  # rdkit molecules
        inactive_molecules,  # rdkit molecules
        model=None,
        cpu_cores=-1,
    ):
        self.active_molecules = active_molecules
        self.inactive_molecules = inactive_molecules
        self.cpu_cores = cpu_cores
        self.model = (
            model
            if model
            else RandomForestClassifier(
                bootstrap=True,
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2,
                n_estimators=300,
                n_jobs=self.cpu_cores,
            )
        )
        self.X = None
        self.y = None
        self._get_data()

    def get_features(self, molecules):
        """Calculate features for each molecule, i.e. molecular descriptors
        and fingerprints."""
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
        active_features = self.get_features(self.active_molecules)
        inactive_features = self.get_features(self.inactive_molecules)
        X = active_features + inactive_features
        y = [1] * len(active_features) + [0] * len(inactive_features)
        self.X = np.array(X)
        self.y = np.array(y)

    def train(self):
        self.model.fit(self.X, self.y)

    def predict(self, molecule):
        features = self.get_features([molecule])
        return self.model.predict(features)[0]

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
