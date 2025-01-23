import pandas as pd
import numpy as np

# Set parameters
num_rows = 1000  # Number of rows to generate
num_machines = 50  # Number of unique machines

# Generate data
np.random.seed(42)  # For reproducibility

data = {
    "Machine_ID": np.random.choice([f"Machine_{i+1}" for i in range(num_machines)], num_rows),
    "Temperature": np.round(np.random.uniform(50.0, 120.0, num_rows), 2),  # Temperature in Fahrenheit
    "Run_Time": np.round(np.random.uniform(0, 24, num_rows), 2),  # Run time in hours
    "Downtime_Flag": np.random.choice([0, 1], num_rows, p=[0.8, 0.2]),  # 80% up, 20% downtime
}

# Create DataFrame
synthetic_data = pd.DataFrame(data)

# Display the first few rows
print(synthetic_data.head())

# Save to CSV (optional)
synthetic_data.to_csv("synthetic_machine_data.csv", index=False)
