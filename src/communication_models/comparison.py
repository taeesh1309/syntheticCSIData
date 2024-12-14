import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def debug_csi_reading(file1_path, file2_path):
    """
    Debug CSI data reading from CSV files with detailed inspection of values.
    """
    # Read CSV files with explicit data type handling
    df1 = pd.read_csv(file1_path, dtype=str)  # Read as strings first to inspect raw values
    df2 = pd.read_csv(file2_path, dtype=str)
    
    print("=== Raw Data Inspection ===")
    print("\nTestbed data first few rows (raw strings):")
    print(df1.iloc[:5, -2:])  # Last two columns
    print("\nSimulation data first few rows (raw strings):")
    print(df2.iloc[:5, -2:])
    
    # Try different reading approaches
    print("\n=== Attempting different reading methods ===")
    
    # Method 1: Direct float conversion
    try:
        I1 = df1.iloc[:, -2].astype(float)
        Q1 = df1.iloc[:, -1].astype(float)
        print("\nMethod 1 - Direct float conversion:")
        print(f"I values range: [{I1.min()}, {I1.max()}]")
        print(f"Q values range: [{Q1.min()}, {Q1.max()}]")
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try reading with explicit float parsing
    try:
        df1_float = pd.read_csv(file1_path, float_precision='high')
        print("\nMethod 2 - High precision float reading:")
        print("I values:")
        print(df1_float.iloc[:5, -2])
        print("Q values:")
        print(df1_float.iloc[:5, -1])
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Manual string cleaning and conversion
    try:
        def clean_and_convert(val):
            if isinstance(val, str):
                # Remove any whitespace and handle scientific notation
                val = val.strip()
                try:
                    return float(val)
                except ValueError:
                    print(f"Failed to convert value: '{val}'")
                    return np.nan
            return val
        
        I1_cleaned = df1.iloc[:, -2].apply(clean_and_convert)
        Q1_cleaned = df1.iloc[:, -1].apply(clean_and_convert)
        
        print("\nMethod 3 - Manual cleaning and conversion:")
        print("I values statistics:")
        print(f"Mean: {I1_cleaned.mean()}")
        print(f"Std: {I1_cleaned.std()}")
        print(f"Min: {I1_cleaned.min()}")
        print(f"Max: {I1_cleaned.max()}")
    except Exception as e:
        print(f"Method 3 failed: {e}")

    return df1, df2

def calculate_similarity_metrics(file1_path, file2_path, bins=50, interpolate=False):
    """
    Calculate similarity metrics between CSI data from testbed and simulation.
    
    Parameters:
    file1_path (str): Path to testbed CSV file
    file2_path (str): Path to simulation CSV file
    bins (int): Number of bins for histogram calculation
    interpolate (bool): If True, interpolate longer dataset to match shorter one
    
    Returns:
    dict: Dictionary containing various similarity metrics
    """
    # Read CSV files
    df1 = pd.read_csv(file1_path)  # testbed data
    df2 = pd.read_csv(file2_path)  # simulation data
    
    # Extract I and Q values
    I1, Q1 = df1.iloc[:, -2:].values.T
    I2, Q2 = df2.iloc[:, -2:].values.T
    
    print(f"Length of testbed data: {len(I1)}")
    print(f"Length of simulation data: {len(I2)}")
    
    if interpolate:
        # Interpolate the longer dataset to match the shorter one
        if len(I1) != len(I2):
            if len(I1) > len(I2):
                # Interpolate dataset 1 to match dataset 2
                x1 = np.linspace(0, 1, len(I1))
                x2 = np.linspace(0, 1, len(I2))
                I1 = interp1d(x1, I1)(x2)
                Q1 = interp1d(x1, Q1)(x2)
            else:
                # Interpolate dataset 2 to match dataset 1
                x1 = np.linspace(0, 1, len(I2))
                x2 = np.linspace(0, 1, len(I1))
                I2 = interp1d(x1, I2)(x2)
                Q2 = interp1d(x1, Q2)(x2)
    else:
        # Truncate to shorter length
        min_length = min(len(I1), len(I2))
        I1, Q1 = I1[:min_length], Q1[:min_length]
        I2, Q2 = I2[:min_length], Q2[:min_length]
    
    # Convert to complex numbers
    csi1 = I1 + 1j * Q1
    csi2 = I2 + 1j * Q2
    
    # Calculate magnitude and phase
    mag1 = np.abs(csi1)
    mag2 = np.abs(csi2)
    phase1 = np.angle(csi1)
    phase2 = np.angle(csi2)
    
    # Calculate histograms for KL divergence
    mag_hist1, mag_bins = np.histogram(mag1, bins=bins, density=True)
    mag_hist2, _ = np.histogram(mag2, bins=mag_bins, density=True)
    phase_hist1, phase_bins = np.histogram(phase1, bins=bins, density=True)
    phase_hist2, _ = np.histogram(phase2, bins=phase_bins, density=True)
    
    # Add small constant and normalize histograms
    epsilon = 1e-10
    mag_hist1 = mag_hist1 + epsilon
    mag_hist2 = mag_hist2 + epsilon
    phase_hist1 = phase_hist1 + epsilon
    phase_hist2 = phase_hist2 + epsilon
    
    mag_hist1 /= mag_hist1.sum()
    mag_hist2 /= mag_hist2.sum()
    phase_hist1 /= phase_hist1.sum()
    phase_hist2 /= phase_hist2.sum()
    
    # Calculate metrics that work with different-length datasets
    distribution_stats = {
        # KL Divergence (works with histograms)
        'kl_divergence_magnitude': entropy(mag_hist1, mag_hist2),
        'kl_divergence_phase': entropy(phase_hist1, phase_hist2),
        
        # Kolmogorov-Smirnov test (works with different-length datasets)
        'ks_test_magnitude': ks_2samp(mag1, mag2).statistic,
        'ks_test_phase': ks_2samp(phase1, phase2).statistic,
        
        # Wasserstein distance (works with different-length datasets)
        'wasserstein_magnitude': wasserstein_distance(mag1, mag2),
        'wasserstein_phase': wasserstein_distance(phase1, phase2),
    }
    
    # Calculate metrics that require same-length datasets
    same_length_stats = {
        # Basic statistical differences
        'mean_magnitude_diff': np.mean(mag1) - np.mean(mag2),
        'mean_phase_diff': np.mean(phase1) - np.mean(phase2),
        'std_magnitude_diff': np.std(mag1) - np.std(mag2),
        'std_phase_diff': np.std(phase1) - np.std(phase2),
        
        # Root Mean Square Error (RMSE)
        'rmse_I': np.sqrt(np.mean((I1 - I2) ** 2)),
        'rmse_Q': np.sqrt(np.mean((Q1 - Q2) ** 2)),
        
        # Normalized RMSE
        'nrmse_I': np.sqrt(np.mean((I1 - I2) ** 2)) / (np.max(I1) - np.min(I1)),
        'nrmse_Q': np.sqrt(np.mean((Q1 - Q2) ** 2)) / (np.max(Q1) - np.min(Q1)),
        
        # Mean Absolute Error (MAE)
        'mae_I': np.mean(np.abs(I1 - I2)),
        'mae_Q': np.mean(np.abs(Q1 - Q2)),
        
        # Maximum Absolute Error
        'max_error_I': np.max(np.abs(I1 - I2)),
        'max_error_Q': np.max(np.abs(Q1 - Q2))
    }
    
    return {**distribution_stats, **same_length_stats}

def visualize_detailed_comparison(file1_path, file2_path, bins=50, interpolate=False):
    """
    Create detailed visualizations comparing the testbed and simulation CSI data.
    """
    # Read and process data
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    I1, Q1 = df1.iloc[:, -2:].values.T
    I2, Q2 = df2.iloc[:, -2:].values.T
    
    if interpolate:
        if len(I1) != len(I2):
            if len(I1) > len(I2):
                x1 = np.linspace(0, 1, len(I1))
                x2 = np.linspace(0, 1, len(I2))
                I1 = interp1d(x1, I1)(x2)
                Q1 = interp1d(x1, Q1)(x2)
            else:
                x1 = np.linspace(0, 1, len(I2))
                x2 = np.linspace(0, 1, len(I1))
                I2 = interp1d(x1, I2)(x2)
                Q2 = interp1d(x1, Q2)(x2)
    else:
        min_length = min(len(I1), len(I2))
        I1, Q1 = I1[:min_length], Q1[:min_length]
        I2, Q2 = I2[:min_length], Q2[:min_length]
    
    csi1 = I1 + 1j * Q1
    csi2 = I2 + 1j * Q2
    
    mag1, phase1 = np.abs(csi1), np.angle(csi1)
    mag2, phase2 = np.abs(csi2), np.angle(csi2)
    
    # Print magnitude statistics for debugging
    print("\nMagnitude Statistics:")
    print(f"Testbed - Min: {np.min(mag1):.4f}, Max: {np.max(mag1):.4f}, Mean: {np.mean(mag1):.4f}, Std: {np.std(mag1):.4f}")
    print(f"Simulation - Min: {np.min(mag2):.4f}, Max: {np.max(mag2):.4f}, Mean: {np.mean(mag2):.4f}, Std: {np.std(mag2):.4f}")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot magnitude distributions with adjustments
    # Calculate bin edges based on data range
    mag_min = min(np.min(mag1), np.min(mag2))
    mag_max = max(np.max(mag1), np.max(mag2))
    bin_edges = np.linspace(mag_min, mag_max, bins)
    
    ax1.hist(mag1, bins=bin_edges, alpha=0.7, label='Testbed', density=True, color='skyblue')
    ax1.hist(mag2, bins=bin_edges, alpha=0.7, label='Simulation', density=True, color='salmon')
    ax1.set_title('Magnitude Distribution')
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Density')
    
    # Set reasonable y-axis limits based on the histogram data
    hist1, _ = np.histogram(mag1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(mag2, bins=bin_edges, density=True)
    max_density = max(np.max(hist1), np.max(hist2))
    ax1.set_ylim(0, max_density * 1.1)  # Add 10% padding
    ax1.legend()
    
    # Plot phase distributions
    ax2.hist(phase1, bins=bins, alpha=0.7, label='Testbed', density=True, color='skyblue')
    ax2.hist(phase2, bins=bins, alpha=0.7, label='Simulation', density=True, color='salmon')
    ax2.set_title('Phase Distribution')
    ax2.set_xlabel('Phase (radians)')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    # Plot I values comparison
    ax3.scatter(I1, I2, alpha=0.5, s=1)
    ax3.plot([min(I1), max(I1)], [min(I1), max(I1)], 'r--')  # Perfect match line
    ax3.set_title('I Values: Testbed vs Simulation')
    ax3.set_xlabel('Testbed I Values')
    ax3.set_ylabel('Simulation I Values')
    
    # Plot Q values comparison
    ax4.scatter(Q1, Q2, alpha=0.5, s=1)
    ax4.plot([min(Q1), max(Q1)], [min(Q1), max(Q1)], 'r--')  # Perfect match line
    ax4.set_title('Q Values: Testbed vs Simulation')
    ax4.set_xlabel('Testbed Q Values')
    ax4.set_ylabel('Simulation Q Values')
    
    plt.tight_layout()
    return fig

# This will truncate the longer dataset to match the shorter one
metrics = calculate_similarity_metrics(
    'src/communication_models/testbed_data.csv', 
    'src/communication_models/simulation_data.csv'
)

# Print results in a formatted way
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Create detailed visualization
fig = visualize_detailed_comparison(
    'src/communication_models/testbed_data.csv', 
    'src/communication_models/simulation_data.csv'
)
plt.show()

# testbed_data, sim_data = debug_csi_reading(
#     'src/communication_models/testbed_data.csv', 
#     'src/communication_models/simulation_data.csv'
# )