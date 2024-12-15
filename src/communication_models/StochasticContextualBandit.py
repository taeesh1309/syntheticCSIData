###################################################################################
######## It is preferable to use Ubuntu/macos with C++ developing toolchain #######
############ vowpalwabbit can be installed using the below pip command ############
######################## pip install vowpalwabbit #################################
###################################################################################

from vowpalwabbit import pyvw
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score
)
from scipy.stats import pearsonr
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
                            Each action tuple is (action_id, reward, target_value).
        :param target: Either 'I' or 'Q' to train for the respective value.
        """
        for data in training_data:
            context = data["context"]
            actions = data["actions"]

            # Start constructing the VW data block
            vw_data = f"shared | {context}\n"

            # Add the actions, with the first row including the reward
            for idx, (action_id, reward, value) in enumerate(actions):
                if idx == 0:  # Only the first action gets the reward (cost)
                    vw_data += f"{action_id}:{-reward} | {target}:{value}\n"
                else:  # Subsequent actions get no cost
                    vw_data += f"{action_id} | {target}:{value}\n"

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
            vw_test_data += f"{action_id} | {target}:{value}\n"

        # Predict scores for all actions
        action_scores = self.vw.predict(vw_test_data)
        print("Action scores:", action_scores)

        # Find the action with the highest score
        best_action_index = action_scores.index(max(action_scores))
        best_action_id = actions[best_action_index][0]
        print("best action ID: ", best_action_id)

        # Retrieve the predicted target value for the best action
        for action_id, value in actions:
            if action_id == best_action_id:
                return value

        return None


def load_csi_data(file_path, target, start_action_id=1):
    """
    Load CSI data from a CSV file and format it for training, grouping by distance_to_base_station.

    :param file_path: Path to the CSV file containing CSI data.
    :param target: Either 'I' or 'Q', specifying the target to train for.
    :param start_action_id: The starting value for action IDs to ensure uniqueness.
    :return: List of training samples formatted for the train method and the next available action ID.
    """
    df = pd.read_csv(file_path)
    training_data = []
    current_action_id = start_action_id  # Initialize action ID counter

    # Group rows by `distance_to_base_station`
    grouped = df.groupby('distance_to_base_station')

    for distance, group in grouped:
        try:
            # Construct the context string (assuming shared context for the group)
            context = (
                f"distance_to_base_station:{distance} "
                f"antenna_geometry={group.iloc[0]['antenna_geometry']} "
                f"base_station_antennas:{group.iloc[0]['base_station_antennas']} "
                f"user_equipment_antennas:{group.iloc[0]['user_equipment_antennas']} "
                f"carrier_frequency:{group.iloc[0]['carrier_frequency']} "
                f"antenna_spacing:{group.iloc[0]['antenna_spacing']} "
                f"scenario={group.iloc[0]['scenario']} "
                f"time_index:{group.iloc[0]['timeIndex']} "
                f"tx_index:{group.iloc[0]['txIndex']}"
            )

            # Create a list of actions for the group
            actions = []
            for _, row in group.iterrows():
                if target == "I":
                    value = row["real"]
                elif target == "Q":
                    value = row["imag"]
                else:
                    raise ValueError("Target must be 'I' or 'Q'.")

                actions.append((
                    current_action_id,  # Unique action ID
                    row['SNR'],         # Reward
                    value               # Target value (I or Q)
                ))
                current_action_id += 1

            # Append the context and actions to the training data
            training_data.append({"context": context, "actions": actions})

        except KeyError as e:
            print(f"Missing column in data: {e}")
        except Exception as e:
            print(f"Error processing group {distance}: {e}")

    return training_data, current_action_id

def test_model(i_model, q_model, test_file):
    """
    Test the trained models on a test dataset and evaluate performance.

    :param i_model: The trained model for predicting I.
    :param q_model: The trained model for predicting Q.
    :param test_file: Path to the test data CSV file.
    :return: A dictionary containing evaluation metrics for I and Q.
    """
    # Load the test data
    df = pd.read_csv(test_file)
    true_values_i = []
    true_values_q = []
    predicted_values_i = []
    predicted_values_q = []

    for _, row in df.iterrows():
        try:
            # Prepare the context
            context = (
                f"antenna_geometry={row['antenna_geometry']} "
                f"base_station_antennas:{row['base_station_antennas']} "
                f"user_equipment_antennas:{row['user_equipment_antennas']} "
                f"carrier_frequency:{row['carrier_frequency']} "
                f"antenna_spacing:{row['antenna_spacing']} "
                f"scenario={row['scenario']} "
                f"distance_to_base_station:{row['distance_to_base_station']} "
                f"time_index:{row['Time Index']} "
                f"tx_index:{row['Tx Index']}"
            )

            # True values for comparison
            true_values_i.append(row["real"])
            true_values_q.append(row["imag"])

            # Dynamic actions (e.g., based on row or predefined action space)
            actions = [
                (1, row["real"]),  # Example: action ID 1 corresponds to the real part
                (2, row["imag"])   # Example: action ID 2 corresponds to the imaginary part
            ]

            # Predict I and Q
            predicted_action_id_i = i_model.predict(context, actions, target="I")
            predicted_action_id_q = q_model.predict(context, actions, target="Q")

            # Match predicted action IDs back to their values
            predicted_i = next(value for action_id, value in actions if action_id == predicted_action_id_i)
            predicted_q = next(value for action_id, value in actions if action_id == predicted_action_id_q)

            # Append predictions
            predicted_values_i.append(predicted_i)
            predicted_values_q.append(predicted_q)

        except KeyError as e:
            print(f"Missing column in data: {e}")
        except Exception as e:
            print(f"Error during prediction: {e}")

    # Calculate evaluation metrics
    metrics = {
        "I": {
            "MAE": mean_absolute_error(true_values_i, predicted_values_i),
            "MSE": mean_squared_error(true_values_i, predicted_values_i),
            "R2": r2_score(true_values_i, predicted_values_i),
            "Explained Variance": explained_variance_score(true_values_i, predicted_values_i),
            "Pearson Correlation": pearsonr(true_values_i, predicted_values_i)[0],
        },
        "Q": {
            "MAE": mean_absolute_error(true_values_q, predicted_values_q),
            "MSE": mean_squared_error(true_values_q, predicted_values_q),
            "R2": r2_score(true_values_q, predicted_values_q),
            "Explained Variance": explained_variance_score(true_values_q, predicted_values_q),
            "Pearson Correlation": pearsonr(true_values_q, predicted_values_q)[0],
        },
    }

    return metrics




def generate_plots(true_values, predicted_values, target):
    """
    Generate evaluation plots for the target variable.

    :param true_values: List of true values.
    :param predicted_values: List of predicted values.
    :param target: Target variable ('I' or 'Q').
    """
    residuals = [true - pred for true, pred in zip(true_values, predicted_values)]

    # True vs. Predicted Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.title(f"True vs. Predicted ({target})")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid()
    plt.savefig(f"true_vs_predicted_{target}.png")
    plt.show()

    # Residuals Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(predicted_values, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f"Residuals Plot ({target})")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid()
    plt.savefig(f"residuals_{target}.png")
    plt.show()

    # Error Distribution (Histogram)
    plt.figure(figsize=(6, 6))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.title(f"Residuals Histogram ({target})")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(f"error_distribution_{target}.png")
    plt.show()

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
        for part_num in range(1, 10):  # 1 to 192
            file_path = f"src/communication_models/data/combined_csi_results_part_{part_num}.csv"
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

    # Test the models
    test_file = "src/communication_models/testbed_data_updated.csv"
    metrics = test_model(i_bandit_model, q_bandit_model, test_file)

    # Print metrics
    print("Test Results:")
    for target in ["I", "Q"]:
        print(f"{target}:")
        for metric, value in metrics[target].items():
            print(f"  {metric}: {value:.4f}")

    # Clean up
    i_bandit_model.close()
    q_bandit_model.close()


if __name__ == "__main__":
    main()
