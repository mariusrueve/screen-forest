import os
import random

import pkg_resources
from rdkit import Chem, RDLogger


class LitPcba:
    def __init__(self, seed=None, verbose=False):
        """
        Initialize the LitPcba class.

        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to
            False.
        """
        if not verbose:
            RDLogger.DisableLog("rdApp.*")
        self.path_to_litpcba = pkg_resources.resource_filename("lit_pcba", "data")
        if seed:
            random.seed(seed)
        self.targets = self._get_targets()

    def _get_targets(self):
        """
        Get the list of target names.

        Returns:
            list: List of target names.
        """
        return [x.name for x in os.scandir(self.path_to_litpcba) if x.is_dir()]

    def get_data(self, target, max_number_of_molecules=50):
        """
        Get the data for a specific target.

        Args:
            target (str): Name of the target.

        Returns:
            dict: Dictionary containing the data for the target.
        Raises:
            ValueError: If the target is not found.
        """
        if target not in self.targets:
            raise ValueError(f"Target '{target}' not found")
        return self._get_litpcba(target, max_number_of_molecules)

    def _get_litpcba(self, folder_name, max_number_of_molecules) -> dict:
        """
        Get the LitPCBA data for a specific target.

        Args:
            folder_name (str): Name of the target folder.
            max_number_of_molecules (int, optional): Maximum number of molecules to
            include. Defaults to 50.

        Returns:
            dict: Dictionary containing the LitPCBA data for the target.
        Raises:
            ValueError: If the specified folder does not exist.
        """
        # Check if the specified folder exists
        if not os.path.isdir(os.path.join(self.path_to_litpcba, folder_name)):
            raise ValueError(
                f"Folder '{folder_name}' does not exist in '{self.path_to_litpcba}'"
            )

        data = {}
        for data_type in ["T", "V"]:
            active_molecules = self._load_molecules(
                os.path.join(
                    self.path_to_litpcba, folder_name, f"active_{data_type}.smi"
                )
            )
            inactive_molecules = self._load_molecules(
                os.path.join(
                    self.path_to_litpcba, folder_name, f"inactive_{data_type}.smi"
                )
            )
            min_length = min(len(active_molecules), len(inactive_molecules))
            if min_length > max_number_of_molecules:
                active_molecules = random.sample(
                    active_molecules, max_number_of_molecules
                )
                inactive_molecules = random.sample(
                    inactive_molecules, max_number_of_molecules
                )
            else:
                active_molecules = random.sample(active_molecules, min_length)
                inactive_molecules = random.sample(inactive_molecules, min_length)
            assert len(active_molecules) == len(inactive_molecules)
            data[data_type] = {
                "active": active_molecules,
                "inactive": inactive_molecules,
            }
        return data

    def _load_molecules(self, path):
        """
        Load molecules from a file.

        Args:
            path (str): Path to the file containing the molecules.

        Returns:
            list: List of molecules.
        """
        mols = Chem.SmilesMolSupplier(path)
        molecules = []
        for mol in mols:
            if mol:
                molecules.append(mol)
        return molecules

    def get_target(self, target):
        """
        Get the LitPCBA data for a specific target.

        Args:
            target (str): Name of the target.

        Returns:
            dict: Dictionary containing the LitPCBA data for the target.
        """
        return self._get_litpcba(target)

    def get_target_names(self):
        """
        Get the list of target names.

        Returns:
            list: List of target names.
        """
        return self.targets


if __name__ == "__main__":
    litpcba = LitPcba()
    target_names = litpcba.get_target_names()
    print(f"Number of targets: {len(target_names)}")
    for target in target_names:
        print(f"Target: {target}")
        data = litpcba.get_target(target)
        for data_type in data:
            for label in data[data_type]:
                print(f"{data_type} {label}: {len(data[data_type][label])} molecules")
        print()
