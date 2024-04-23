#Παρισης Θεοχαρης
#Π2017162

import random
import numpy as np
import matplotlib.pyplot as plt

def custom_linear_kernel_func(A, B):
    """
    Υπολογισμός του προσαρμοσμένου γραμμικού πυρήνα μεταξύ των δύο πινάκων.
    Παράμετροι:
        A (numpy.ndarray): Πίνακας με χαρακτηριστικα 1.
        B (numpy.ndarray): Πίνακας με χαρακτηριστικα 2.
    Επιστροφή:
        numpy.ndarray: Αποτελέσματα του προσαρμοσμένου γραμμικού πυρήνα.
    """
    return np.dot(A, B.T)

def custom_perceptron_with_linear_kernel(X_train, y_train, epochs=100):
    """
    Εκπαιδευση του αλγορίθμου perceptron χρησιμοποιώντας προσαρμοσμένο γραμμικό πυρήνα.
    Παράμετροι:
        X_train (numpy.ndarray): Ο πίνακας χαρακτηριστικών εκπαίδευσης.
        y_train (numpy.ndarray): Ο πίνακας με τις ετικέτες εκπαίδευσης.
        epochs (int): Ο αριθμός των εποχών εκπαίδευσης.
    Επιστροφή:
        numpy.ndarray: Οι συντελεστές weights.
        float: Η σταθερά bias.
    """
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_samples)
    bias = 0

    for epoch in range(epochs):
        for i in range(n_samples):
            # Έλεγχος εάν η πρόβλεψη είναι σωστή ή όχι και αν όχι, προσαρμόζονται οι συντελεστές weights
            if np.sign(np.dot(weights, y_train * custom_linear_kernel_func(X_train, X_train[i])) + bias) != y_train[i]:
                weights[i] += y_train[i]  # Προσθήκη της ετικέτας στον συντελεστή weights
                bias += y_train[i]         # Προσθήκη της ετικέτας στη σταθερά bias

    return weights, bias

def custom_perceptron_predict_with_linear_kernel(X_test, X_train, weights, bias):
    """
    Πρόβλεψη των ετικετών χρησιμοποιώντας τον αλγόριθμο perceptron.
    Παράμετροι:
        X_test (numpy.ndarray): Ο πίνακας χαρακτηριστικών δοκιμής.
        X_train (numpy.ndarray): Ο πίνακας χαρακτηριστικών εκπαίδευσης.
        weights (numpy.ndarray): Οι συντελεστές weights.
        bias (float): Η σταθερά bias.
    Επιστροφή:
        numpy.ndarray: Οι ετικέτες.
    """
    return np.sign(np.dot(weights, custom_linear_kernel_func(X_train, X_test)) + bias)

def custom_make_plot(X, y, weights, bias):
    """
    Σχεδιάζει τη διαχωριστική ευθεία του perceptron.
    Παράμετροι:
        X (numpy.ndarray): Ο πίνακας χαρακτηριστικών.
        y (numpy.ndarray): Ο πίνακας με τις ετικέτες.
        weights (numpy.ndarray): Οι συντελεστές weights.
        bias (float): Η σταθερά bias.
    """
    plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')  # Διαχωρίζει τα σημεία ανά κλάση
    x_vals = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)  # Δημιουργεί ένα εύρος για το x
    if np.abs(weights[1]) > 1e-3:  # Έλεγχος εάν ο δεύτερος συντελεστής του weights είναι μεγάλος
        y_vals = -(weights[0] / weights[1]) * x_vals - (bias / weights[1])  # Υπολογισμός των y τιμών για τη γραμμική ευθεία
        plt.plot(x_vals, y_vals, color='red')  # Σχεδιάζει τη γραμμική ευθεία
    else:
        x_mean = np.mean(X[:,0])  # Υπολογίζει τον μέσο όρο του x
        plt.axvline(x=x_mean, color='red', linestyle='--')  # Σχεδιάζει μια κάθετη γραμμή στον μέσο όρο του x
    plt.xlabel('X1')  # Ετικέτα x
    plt.ylabel('X2')  # Ετικέτα y
    plt.title('Διαχωριστική Ευθεία του Perceptron με προσαρμοσμένο γραμμικό πυρήνα')  # Τίτλος γραφήματος
    plt.show()  # Εμφάνιση του γραφήματος

# Φορτώνει τα δεδομένα
data = np.loadtxt('perceptron_non_separ.txt')  # Φορτώνει το αρχείο δεδομένων
X = data[:, :-1]  # Χωρίζει τα χαρακτηριστικά από τις ετικέτες
y = data[:, -1]

# Διαίρει τα δεδομένα σε σύνολα εκπαίδευσης και δοκιμής
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Εκπαιδεύει τον perceptron με προσαρμοσμένο γραμμικό πυρήνα
weights, bias = custom_perceptron_with_linear_kernel(X_train, y_train)

# Προβλέπει τις ετικέτες χρησιμοποιώντας τον perceptron με προσαρμοσμένο γραμμικό πυρήνα
y_pred = custom_perceptron_predict_with_linear_kernel(X_test, X_train, weights, bias)

# Υπολογίζει την ακρίβεια
accuracy = np.mean(y_pred == y_test)
num_errors = np.sum(y_pred != y_test)

print("Συνολικός αριθμός λαθών:", num_errors)
print("Ακρίβεια:", accuracy)

# Σχεδιάζει τη διαχωριστική ευθεία
custom_make_plot(X_train, y_train, weights, bias)
