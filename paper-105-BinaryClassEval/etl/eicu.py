import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import scipy

# Add parent directory to path to import Model class
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Use absolute import to avoid circular imports
from etl.model import Model

class EICU(Model):
    """eICU dataset model that implements the Model interface.

    Loads and processes eICU data, focusing on Apache scores and mortality outcomes.
    Provides access to predictions for survived and deceased patients.

    Parameters
    ----------
    base_dir : str, default=None
        Base directory containing eICU data. If None, uses '../data' relative to this file.
    demo : bool, default=False
        Whether to use demo dataset (smaller) or full dataset.
    """

    def __init__(self, base_dir=None, demo=False):
        """Initialize the EICU model.

        Parameters
        ----------
        base_dir : str, default=None
            Base directory containing eICU data.
        demo : bool, default=False
            Whether to use demo dataset or full dataset.
        """
        # Set up paths
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data"
        else:
            self.base_dir = base_dir

        self.eicu_dir = os.path.join(self.base_dir, "demo" if demo else "full")
        self.image_dir = os.path.dirname(self.base_dir)
        os.makedirs(self.image_dir, exist_ok=True)

        # Define file paths
        self.apache_results_file = os.path.join(self.eicu_dir, "apachePatientResult.csv")
        self.patient_file = os.path.join(self.eicu_dir, "patient.csv")

        # Load data
        self.data = self.load_data()

        # Split data into class 0 (survived) and class 1 (died)
        self.class_0_mask = (self.data['mortality_binary'] == 0)
        self.class_1_mask = (self.data['mortality_binary'] == 1)

        # Extract predicted probabilities
        self._train_pred = pd.to_numeric(self.data['predicted_hospital_mortality'], errors='coerce')

        # Store predictions for class 0 and class 1 instances
        self._train_0 = self._train_pred[self.class_0_mask].values
        self._train_1 = self._train_pred[self.class_1_mask].values

        # Calculate prevalence
        self._train_prevalence = self.class_1_mask.mean()

    @property
    def train_0(self) -> np.ndarray:
        """Return predictions for class 0 samples (survived)."""
        return self._train_0

    @property
    def train_1(self) -> np.ndarray:
        """Return predictions for class 1 samples (died)."""
        return self._train_1

    @property
    def train_prevalence(self) -> float:
        """Return the prevalence used during training."""
        return self._train_prevalence

    def load_data(self):
        """Load and process the eICU data to extract Apache scores and binary outcomes.

        Returns
        -------
        pd.DataFrame
            DataFrame containing patient IDs, Apache scores, and mortality outcomes.
        """
        # Load the necessary CSV files
        apache_results = pd.read_csv(self.apache_results_file)
        apache_results = apache_results[apache_results['predictedhospitalmortality'] >= 0]
        patients = pd.read_csv(self.patient_file)
        data = apache_results.merge(patients, on='patientunitstayid', how='inner')

        # Extract relevant columns
        result = pd.DataFrame({
            'patientunitstayid': data['patientunitstayid'],
            'patienthealthsystemstayid': data['patienthealthsystemstayid'],
            'apache_score': data['apachescore'],
            'aps_score': data['acutephysiologyscore'],
            'apache_version': data['apacheversion'],
            'predicted_hospital_mortality': data['predictedhospitalmortality'],
            'actual_hospital_mortality': data['actualhospitalmortality'],
            'age': data['age'],
            'gender': data['gender'],
            'ethnicity': data['ethnicity'],
        })

        # Create binary outcome variable (1 for expired/died, 0 for alive)
        result['mortality_binary'] = result['actual_hospital_mortality'].apply(
            lambda x: 1 if x == 'EXPIRED' else 0
        )

        # Handle missing values for numerical features
        numerical_cols = ['apache_score', 'aps_score', 'predicted_hospital_mortality']
        for col in numerical_cols:
            result[col] = pd.to_numeric(result[col], errors='coerce')

        # Clean up age values (handles "> 89" ages)
        result['age'] = result['age'].apply(lambda x: 90 if x == '> 89' else x)
        result['age'] = pd.to_numeric(result['age'], errors='coerce')

        return result

# Main execution when script is run directly
if __name__ == "__main__":
    # Create EICU model instance
    eicu_model = EICU(demo=True)

    # Display basic information
    print(f"Loaded data for {len(eicu_model.data)} patients")

    # Print mortality rate
    mortality_rate = eicu_model.train_prevalence * 100
    print(f"Overall mortality rate: {mortality_rate:.2f}%")

    # Print average Apache score

    plt.hist(scipy.special.logit(eicu_model.data['predicted_hospital_mortality']), bins=100)
    plt.show()
