from vowpalwabbit import pyvw  ## install using pip install vowpalwabbit
import numpy as np
import pandas as pd

class StochasticContextualBandit:
    def __init__(self, vw_options="--cb_explore_adf --quiet"):
        """
        Initialize the Vowpal Wabbit workspace for contextual bandit training.

        :param vw_options: VW options for contextual bandit training.
        """
        self.vw = pyvw.Workspace(vw_options)

    def train(self, training_data):
        """
        Train the model using contextual bandit training data.

        :param training_data: List of training samples, where each sample is a dictionary
                              containing 'context' (str) and 'actions' (list of tuples).
                              Each action tuple is (action_id, I, Q).
        """
        for data in training_data:
            context = data["context"]
            actions = data["actions"]

            # Construct VW training data
            vw_data = f"shared | {context}\n"
            for action_id, I, Q in actions:
                reward = 1.0  # Assign equal reward since rewards are not provided
                vw_data += f"{action_id}:{reward} | I:{I} Q:{Q}\n"

            # Train VW
            self.vw.learn(vw_data)

    def predict(self, context, base_station_antennas, user_equipment_antennas, active_users, time_samples):
        """
        Predict the best I and Q values given a context.

        :param context: Contextual features as a string.
        :param base_station_antennas: Number of antennas at the base station.
        :param user_equipment_antennas: Number of antennas at the user equipment.
        :param active_users: Number of active users.
        :param time_samples: Number of time samples (frames).
        :return: A list of predicted (I, Q) tuples.
        """
        # Calculate the number of predictions required
        num_predictions = base_station_antennas * user_equipment_antennas * active_users * time_samples

        vw_test_data = f"shared | {context}\n"

        # Generate placeholder actions for prediction
        for i in range(1, num_predictions + 1):
            vw_test_data += f"{i}:0 | \n"

        # Predict the best action(s)
        prediction = self.vw.predict(vw_test_data)

        # Decode predictions into I, Q values (placeholders since VW does not directly output I, Q values)
        predicted_values = []
        for action in range(1, num_predictions + 1):
            predicted_values.append((0.0, 0.0))  # Placeholder for predicted I, Q values
        return predicted_values

    def load_csi_data(self, file_path):
        """
        Load CSI data from a CSV file and format it for training.

        :param file_path: Path to the CSV file containing CSI data.
        :return: List of training samples formatted for the train method.
        """
        df = pd.read_csv(file_path)
        training_data = []

        for _, row in df.iterrows():
            context = f"{row['numerology_pattern']} {row['antenna_geometry']} {row['base_station_antennas']} {row['user_equipment_antennas']} {row['carrier_frequency']} {row['antenna_spacing']} {row['scenario']} {row['azimuth']} {row['distance_to_base_station']}"
            actions = [(idx + 1, row[f"I_{idx + 1}"], row[f"Q_{idx + 1}"]) for idx in range(int(row['number_of_actions']))]
            training_data.append({"context": context, "actions": actions})

        return training_data

    def close(self):
        """
        Clean up the Vowpal Wabbit workspace.
        """
        self.vw.finish()

# Example Usage
def main():
    """
    Main function to demonstrate training and prediction using the stochastic contextual bandit model.
    """
    # Initialize the bandit model
    bandit_model = StochasticContextualBandit()

    # Load training data from CSV
    training_data = bandit_model.load_csi_data("csi_data.csv")

    # Train the model
    bandit_model.train(training_data)

    # Test data (context for prediction)
    test_context = "2 UCA 32 8 900MHz 0.25 bad_urban 30 300"

    # Predict the best I and Q values
    predicted_values = bandit_model.predict(
        test_context,
        base_station_antennas=32,
        user_equipment_antennas=8,
        active_users=10,
        time_samples=5
    )
    print(f"Predicted I and Q values: {predicted_values}")

    # Clean up
    bandit_model.close()

if __name__ == "__main__":
    main()
