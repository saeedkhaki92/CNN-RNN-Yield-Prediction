import numpy as np

# Creating an array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Calculating the mean of the entire array
mean = np.mean(arr)
print("Mean of the entire array:", mean)

# Calculating the mean along a specific axis (axis=0 means along columns)
mean_axis_0 = np.mean(arr, axis=0)
print("Mean along axis 0 (columns):", mean_axis_0)

# Calculating the mean along a different axis (axis=1 means along rows)
mean_axis_1 = np.mean(arr, axis=1)
print("Mean along axis 1 (rows):", mean_axis_1)
