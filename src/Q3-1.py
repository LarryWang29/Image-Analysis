import numpy as np

x_start = np.array([1, 1])
target = np.array([0, 0])

def f(x):
    return x[0] ** 2 / 2 + x[1] ** 2

counter = 0

while f(x_start) - f(target) > 0.01:
    # Calculate derivative at current position
    update = np.array([x_start[0], 2 * x_start[1]])

    # Update the position
    x_start = x_start - 0.5 * update

    # Increment counter
    counter += 1

print(f"Taken {counter} iterations to reach the target." +
      f"The final x is {x_start} and the final f(x) is {f(x_start)}")
