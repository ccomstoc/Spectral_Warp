import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sum_labels

import numpy as np
import math



def resize_data_with_priority(data, idx_range, target_size):
    #This function turns 010203 into 130, so this could be investigated, the reason for this is because of how lower and
    #upper bound ar calculated,
    # fisrt the range of indexes 012345 are converted into 0,2.5,5 with the linspace funtion
    # then the loop identifies uses this to idenfy the ranges in original array the qualify for remapping
    #     the problem arises because first mapping to 0 ar (0,1), then 2,0,3, then the last is 0, so this could be optimized but
    #     honestly might not cause issues on large scale
    start_idx = idx_range[0]
    end_idx = idx_range[1]
    # Extract the og_data to resize
    og_data = data[start_idx:end_idx + 1]
    original_size = len(og_data)
    #return og_data

    # Initialize the new resized array with zeros
    resized_data = np.zeros(target_size, dtype=og_data.dtype)

    #Expansion
    if original_size < target_size:
        ratio = target_size/original_size
        for old_index in range(original_size):
            new_index = round(old_index*ratio)-1
            resized_data[new_index] = og_data[old_index]
        print(f"og_data{og_data[:50]}")
        print()
        print(f"new {resized_data[:50]}")
        return resized_data
    else: #Contraction
        ratio = original_size/target_size
        for i in range(target_size):
            lower_bound = i*ratio
            upper_bound = (i+1)*ratio
            lower_frac, lower_int = math.modf(lower_bound)
            upper_frac, upper_int = math.modf(upper_bound)
            lower_int = int(lower_int)
            upper_int = int(upper_int)
            max_value = 0
            for og_idx in range(lower_int, upper_int+1):
                testing_value = 0
                if og_idx == lower_int:
                    testing_value = (1-lower_frac)*og_data[og_idx]
                elif og_idx == upper_int:
                    if upper_frac != 0:#edge case with last interation,
                        testing_value = upper_frac*og_data[og_idx]
                else:
                    testing_value = og_data[og_idx]
                max_value = max(testing_value, max_value, key=abs)
            resized_data[i] = max_value


        return resized_data

# Generate the array
length = 8192
array = np.zeros(length, dtype=float)
array[::128] = 1  # Set every 4th element to 1
#anchors = [0] * 8192
#anchors[4000]=1

for idx in range(8192):
    array[idx] = math.sin(idx/500)





# Example usage:
middle_warp_anchor = 7000

first_original_range = [0, 4095]
second_original_range = [4096, 8191]

first_target_range = [0,middle_warp_anchor]
second_target_range = [middle_warp_anchor+1,8191]

first_new_size = middle_warp_anchor;

second_new_size = 8192-middle_warp_anchor;





first_half = resize_data_with_priority(array,first_original_range,first_new_size)


second_half = resize_data_with_priority(array,second_original_range,second_new_size)

final_warped_array = np.append(first_half,second_half)



# Create the x-axis values
x_values = np.arange(length)

# Plot the graph
plt.figure(figsize=(10, 4))
#plt.plot(x_values, array, color='gray', linestyle='-', marker='o', markersize=2)
plt.plot(x_values, final_warped_array , color='blue', linestyle='-', marker='o', markersize=2)

plt.ylim(-1.1, 1.1)  # Set y-axis limits to range from 0 to 1
plt.xlabel('Array Index')
plt.ylabel('Value')
plt.title('Array with 1 Every 4 Spaces')
plt.grid(True)
plt.show()

