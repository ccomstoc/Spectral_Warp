import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sum_labels



def scale_data(data, original_range, target_range):
    scaled_data = np.interp(data, original_range, target_range)
    return scaled_data


def scale_indices(data, original_range, target_range):
    original_indices = np.arange(original_range[0], original_range[1] + 1)
    target_indices = np.interp(original_indices,
                               [original_range[0], original_range[1]],
                               [target_range[0], target_range[1]])
    new_data = np.zeros(target_range[1] - target_range[0] + 1)

    for orig_idx, target_idx in zip(original_indices, target_indices):
        new_idx = int(round(target_idx)) - target_range[0]
        new_data[new_idx] = max(new_data[new_idx], data[orig_idx - original_range[0]])

    return new_data


# Generate the array
length = 8192
array = np.zeros(length, dtype=int)
array[::128] = 1  # Set every 4th element to 1
#anchors = [0] * 8192
#anchors[4000]=1




# Example usage:
middle_warp_anchor = 1000;

first_original_range = [0, 4095]
second_original_range = [4096, 8191]

first_target_range = [0,middle_warp_anchor]
second_target_range = [middle_warp_anchor+1,8191]



first_half = scale_indices(array, first_original_range, first_target_range)

second_half = scale_indices(array, second_original_range, second_target_range)

final_warped_array = np.append(first_half,second_half)



# Create the x-axis values
x_values = np.arange(length)

# Plot the graph
plt.figure(figsize=(10, 4))
#plt.plot(x_values, array, color='gray', linestyle='-', marker='o', markersize=2)
plt.plot(x_values, final_warped_array , color='blue', linestyle='-', marker='o', markersize=2)

plt.ylim(-0.1, 1.1)  # Set y-axis limits to range from 0 to 1
plt.xlabel('Array Index')
plt.ylabel('Value')
plt.title('Array with 1 Every 4 Spaces')
plt.grid(True)
plt.show()

