# -----------------------------------------------------------
# 📌 PROJECT: Tumor Classification using Logistic Regression
# -----------------------------------------------------------
# GOAL:
# Predict whether a tumor is malignant (1) or benign (0)
# using patient tumor features like radius, texture, perimeter, area, and smoothness.

# We use logistic regression with gradient descent from scratch.
# -----------------------------------------------------------


import numpy as np

# -------- STEP 1: Load Data --------
def load_data():
    # Features: [radius, texture, perimeter, area, smoothness]
    x = np.array([
        [14.5, 20.2, 95.2, 600.1, 0.08],
        [20.3, 25.5, 130.0, 1200.3, 0.10],
        [12.4, 15.0, 82.0, 520.5, 0.09],
        [22.1, 30.0, 140.5, 1300.0, 0.11],
        [10.2, 12.8, 70.5, 400.6, 0.07],
        [18.7, 22.1, 120.3, 1100.1, 0.12],
        [11.3, 13.3, 75.2, 420.8, 0.06],
        [16.0, 18.9, 98.1, 670.4, 0.09],
        [13.8, 17.6, 90.0, 570.3, 0.08],
        [24.5, 33.0, 155.0, 1500.5, 0.13]
    ])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Labels
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean)/x_std, y, x_mean, x_std

def gradient(x,y,w,b):
    m,n=x.shape
    derivative_w=np.zeros(n)
    derivative_b=0
    for i in range(m):
        z = np.dot(x[i], w) + b
        predicted = 1 / (1 + np.exp(-z))
        derivative_w+=(predicted-y[i])*x[i]
        derivative_b+=(predicted-y[i])
    derivative_w=derivative_w/m
    derivative_b=derivative_b/m
    return derivative_w,derivative_b

def gradient_decent(x,y,temp_w,temp_b,iterations,alpha):
    w=temp_w
    b=temp_b
    for i in range(iterations):
        derivative_w,derivative_b=gradient(x,y,w,b)
        w -= alpha * derivative_w
        b -= alpha * derivative_b
        if i % 100 == 0 or i == iterations - 1:
            z = np.dot(x, w) + b
            pred = 1 / (1 + np.exp(-z))
            cost = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
            print(f"Iteration {i}: Cost = {cost:.4f}")
    return w, b
    
def predicted(x,w,b):
    z = np.dot(w, x) + b
    predicted=1 / (1 + np.exp(-z))
    return predicted


x_train, y_train, x_mean, x_std =  load_data()

# Parameters
init_w = np.zeros(x_train.shape[1])  # ✅ Fix: shape = (5,)
init_b = 0.0
alpha = 0.01
iterations = 1500

# Train
w, b = gradient_decent(x_train, y_train, init_w, init_b, iterations, alpha)

# Predict
new_sample = np.array([15.0, 18.0, 85.0, 650.0, 0.085])
new_sample_norm = (new_sample - x_mean) / x_std
prob = predicted(new_sample_norm, w, b)

# Output
print(f"\n🧠 Prediction probability of being malignant: {prob:.4f}")
print("🔍 Diagnosis:", "Malignant" if prob > 0.5 else "Benign")
