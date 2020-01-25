import numpy as np

# applying and operator to each pair of numbers
training_input = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
training_output = np.array([0,0,0,1])

# initializing weights to zero
# it can also be random number between -1 and 1
synaptic_weights = [0,0]

print("Synaptic Weights Before Training:")
print(synaptic_weights)

# alpha must be less than treshold
# otherwise adjsutment value will never be zero
threshold = 5
alpha = 0.3

def step_function(x):
    if x > threshold:
        return 1
    else:
        return 0

# calculates dot product for inputs and weights
def input_function(inputs, w):
    return np.dot(inputs, w)

# substract calculated outputs from desired outputs
def calculate_error(output):
    return training_output.T - output

# training loop
for each_step in range(1000):
    input_results = input_function(training_input, synaptic_weights)
    
    output = []
    for i in input_results:
        output.append(step_function(i))

    errors = calculate_error(output)
    adjustment = sum(errors) * alpha

    # update synaptic values
    for i, value in enumerate(synaptic_weights):
        synaptic_weights[i] += adjustment

print("\nTraining is done\n")

print("Synaptic Weights After Training:")
print(synaptic_weights)

print("\nOutput:",output)
