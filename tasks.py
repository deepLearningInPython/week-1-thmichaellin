import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.

# Task 1: 
# Instructions:
#Write a function that takes one numeric argument as input. 
#If the number is larger than zero, the function should return 1, otherwise is should return -1.
#The name of the function should be step

# Your code here:
# -----------------------------------------------

def step(input: float) -> int:
    """
    Return 1 if input is larger than 0; else return -1.

    Args:
        input (float): Input number.

    Returns:
        int: 1 if input > 0; else -1
    """

    if input > 0:
        return 1
    return -1

# -----------------------------------------------


# Task 2:
# Instructions:
#Write a function that takes in two arguments: a numpy array, and an integer (call argument "cutoff" and set default to 0).
#The function should return a numpy array of the same length, with all elements smaller than the cutoff being set to cutoff).
#The name of the function should be ReLu


# Your code here:
# -----------------------------------------------

def ReLu(array_in: np.array, cutoff: int = 0) -> np.array:
    """
    Take numpy array and return numpy array of same length where all elements < cutoff are replaced
    with the cutoff value.

    Args:
        array (np.array): Input array.
        cutoff (int, optional): Cutoff value. Defaults to 0.

    Returns:
        np.array: Output array.
    """

    array_in[array_in < cutoff] = cutoff
    return array_in

# -----------------------------------------------


# Task 3:
# Instructions:
#Write a function that takes in a two-dimensional numpy array of size (n, p) and a one-dimensional numpy array of size p.
#The function should start by multiplying the two numpy arrays (matrix multiplication).
#Next, apply the ReLu function from above to the resulting matrix and return the result.
#Name the function neural_net_layer

# Your code here:
# -----------------------------------------------

def neural_net_layer(array_in_2d: np.array, array_in_1d: np.array) -> np.array:
    """
    Find product of an (n, p) matrix and (p) matrix.
    Apply ReLu activation to the resulting matrix.

    Args:
        array_in_2d (np.array): 2D input array.
        array_in_1d (np.array): 1D input array.

    Raises:
        ValueError: If dimensions of matrices do not match.

    Returns:
        np.array: Output array.
    """
    if not array_in_2d.shape[1] == array_in_1d.shape[0]:
        raise ValueError("Input arrays should have shapes (n, p) and (p)")
    array_mult = np.matmul(array_in_2d, array_in_1d)
    return ReLu(array_mult)

# ------------------------------------------