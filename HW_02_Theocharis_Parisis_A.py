import numpy as np
import matplotlib.pyplot as plt

# Load data from file
data = np.loadtxt('perceptron_non_separ.txt', delimiter='\t')

# Separate data into X_1, X_2, and y
X_1 = data[:, 0]
X_2 = data[:, 1]
y = data[:, 2]

# Initialize weights and bias
w = np.zeros(2)
b = 0

# Initialize number of misclassified samples
n_misclassified = 0

# Define learning rate
eta = 0.1

# Define maximum number of iterations
max_iter = 1000

# Implement perceptron algorithm
for i in range(max_iter):
    misclassified = False
    for j in range(len(X_1)):
        y_pred = np.sign(np.dot(w, [X_1[j], X_2[j]]) + b)
        if y_pred != y[j]:
            misclassified = True
            n_misclassified += 1
            w += eta * y[j] * np.array([X_1[j], X_2[j]])
            b += eta * y[j]
    if not misclassified:
        break

# Plot graph with number of misclassified samples
print("Number of misclassified samples:", n_misclassified)

# Plot graph with samples and separating line
plt.scatter(X_1[y==1], X_2[y==1], color='red', label='Class 1')
plt.scatter(X_1[y==-1], X_2[y==-1], color='blue', label='Class -1')
x_min, x_max = min(X_1), max(X_1)
y_min, y_max = min(X_2), max(X_2)
x = np.linspace(x_min, x_max, 100)
y = -(w[0]*x + b) / w[1]
plt.plot(x, y, color='black', label='Separating line')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend()
plt.show()