import numpy as np
from collections import Counter

# Assuming you have an array of numbers
data = np.array([1, 2, 3, 2, 4, 3, 3, 4, 4])

# Compute the frequency of each value in the array
value_counts = Counter(data)

# Find the maximum frequency (mode)
max_frequency = max(value_counts.values())

# Get all values with the maximum frequency (modes)
modes = [value for value, count in value_counts.items() if count == max_frequency]

# Randomly select one of the modes
selected_mode = np.random.choice(modes)

print("Modes:", modes)
print("Selected Mode:", selected_mode)


filter_data = (data < 3)
print(filter_data)
print(np.multiply(data, filter_data))