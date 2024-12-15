###################################################################################
######## It is preferable to use Ubuntu/macos with C++ developing toolchain #######
############ vowpalwabbit can be installed using the below pip command ############
######################## pip install vowpalwabbit #################################
###################################################################################

from vowpalwabbit import pyvw
import pandas as pd
import os


class StochasticContextualBandit:
    def __init__(self, vw_options="--cb_explore_adf --quiet", model_file=None):
        """
        Initialize the Vowpal Wabbit workspace for contextual bandit training.

        :param vw_options: VW options for contextual bandit training.
        :param model_file: Path to an existing model file to load, if any.
        """
        if model_file:
            vw_options += f" --initial_regressor {model_file}"
        self.vw = pyvw.Workspace(vw_options)
        self.model_file = model_file
        
    def save_model(self, save_path):
        """
        Save the trained model to a file.

        :param save_path: Path where the model will be saved.
        """
        self.vw.save(save_path)
        self.model_file = save_path

    def close(self):
        """
        Clean up the Vowpal Wabbit workspace.
        """
        self.vw.finish()

    def train(self, training_data, target):
        """
        Train the model using contextual bandit training data.

        :param training_data: List of training samples, where each sample is a dictionary
                              containing 'context' (str) and 'actions' (list of tuples).
                              Each action tuple is (action_id, target_value, reward).
        :param target: Either 'I' or 'Q' to train for the respective value.
        """
        for data in training_data:
            context = data["context"]
            actions = data["actions"]

            # Construct VW training data
            vw_data = f"shared | {context}\n"
            for action_id, reward, value in actions:
                vw_data += f"{action_id}:{reward} | {target}:{value}\n"

            # Train VW
            self.vw.learn(vw_data)

    def predict(self, context, actions, target):
        """
        Predict the best target value (I or Q) given a context.

        :param context: Contextual features as a string.
        :param actions: A list of actions with placeholders for the target value.
        :param target: Either 'I' or 'Q' to predict for the respective value.
        :return: The predicted target value.
        """
        vw_test_data = f"shared | {context}\n"
        for action_id, value in actions:
            vw_test_data += f"{action_id}:0 | {target}:{value}\n"

        # Predict the best action ID
        best_action_id = self.vw.predict(vw_test_data)

        # Retrieve the predicted target value for the best action
        for action_id, value in actions:
            if action_id == best_action_id:
                return value

        return None


def load_csi_data(file_path, target, start_action_id=1):
    """
    Load CSI data from a CSV file and format it for training, ensuring unique action IDs.

    :param file_path: Path to the CSV file containing CSI data.
    :param target: Either 'I' or 'Q', specifying the target to train for.
    :param start_action_id: The starting value for action IDs to ensure uniqueness.
    :return: List of training samples formatted for the train method and the next available action ID.
    """
    df = pd.read_csv(file_path)
    training_data = []
    current_action_id = start_action_id  # Initialize action ID counter

    for _, row in df.iterrows():
        context = f"{row['antenna_geometry']} {row['base_station_antennas']} {row['user_equipment_antennas']} {row['carrier_frequency']} {row['antenna_spacing']} {row['scenario']} {row['distance_to_base_station']}"
        if target == "I":
            value = row["real"]
        elif target == "Q":
            value = row["imag"]
        else:
            raise ValueError("Target must be 'I' or 'Q'.")

        # Create a single action with a unique action ID
        actions = [
            (
                current_action_id,  # Unique action ID
                row['SNR'],         # Reward
                value               # Target value (I or Q)
            )
        ]
        current_action_id += 1
        training_data.append({"context": context, "actions": actions})

    return training_data, current_action_id


def main():
    """
    Main function to demonstrate training and prediction using the stochastic contextual bandit model.
    """

    # Model paths for I and Q
    i_model_path = "cb_model_I.vw"
    q_model_path = "cb_model_Q.vw"

    # Initialize models
    i_bandit_model = StochasticContextualBandit(model_file=i_model_path if os.path.exists(i_model_path) else None)
    q_bandit_model = StochasticContextualBandit(model_file=q_model_path if os.path.exists(q_model_path) else None)

    action_id = 1
    # Train the models if they don't already exist
    if not os.path.exists(i_model_path) or not os.path.exists(q_model_path):
        # Iterate through all files
        for part_num in range(1, 193):  # 1 to 192
            file_path = f"combined_csi_results_part_{part_num}.csv"
            if os.path.exists(file_path):
                print(f"Processing {file_path}...")

                # Load training data for I
                i_training_data, action_id = load_csi_data(file_path, target="I", start_action_id=action_id)
                # Train the I model
                i_bandit_model.train(i_training_data, target="I")

                # Load training data for Q
                q_training_data, action_id = load_csi_data(file_path, target="Q", start_action_id=action_id)
                # Train the Q model
                q_bandit_model.train(q_training_data, target="Q")

        # Save the trained models
        i_bandit_model.save_model(i_model_path)
        print(f"I model saved to {i_model_path}")
        q_bandit_model.save_model(q_model_path)
        print(f"Q model saved to {q_model_path}")

    # Test data (context for prediction)
    test_context = "2 UCA 32 8 900MHz 0.25 bad_urban 30 300"
    actions = [(1, 0.0)]  # Dummy actions for prediction

    # Predict I and Q
    predicted_i = i_bandit_model.predict(test_context, actions, target="I")
    predicted_q = q_bandit_model.predict(test_context, actions, target="Q")

    print(f"Predicted I: {predicted_i}, Predicted Q: {predicted_q}")

    # Clean up
    i_bandit_model.close()
    q_bandit_model.close()


if __name__ == "__main__":
    main()
