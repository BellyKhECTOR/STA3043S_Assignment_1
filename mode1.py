import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Set the title and layout of the app
st.set_page_config(page_title="Animal Location Analysis", layout="wide")
st.title("🐾 Animal Location Analysis with Acceptance-Rejection Sampling")

# Load data directly from the text file
data_file = "animal_location_data.txt"

# Load data
df = pd.read_csv(data_file, sep="\t", header=0)
st.sidebar.header("Data Overview")
st.sidebar.write(f"Total Records: {len(df)}")
st.sidebar.write(df.head())

# Calculate velocity
difference_x = df['x'].diff().fillna(0).to_numpy()
difference_y = df['y'].diff().fillna(0).to_numpy()

# Calculate s1 and s2
s1 = np.sum(difference_x[1:] * difference_x[:-1] + difference_y[1:] * difference_y[:-1])
s2 = np.sum(difference_x[1:] * difference_y[:-1] - difference_y[1:] * difference_x[:-1])

# Display calculated values
st.subheader("Calculated Values")
st.write(f"**s1:** {s1:.2f}")
st.write(f"**s2:** {s2:.2f}")

# User input for sample size
st.sidebar.header("Sampling Parameters")
n_samples = st.sidebar.slider("Select the number of samples for the posterior distribution:", min_value=5000, max_value=10000, value=5000)

# Generate posterior samples using Acceptance-Rejection Sampling
def ar(n_samples, s1, s2):
    samples = []
    accepted = 0
    C = max(np.exp(s1 * np.cos(np.linspace(-np.pi, np.pi, 5000)) + s2 * np.sin(np.linspace(-np.pi, np.pi, 5000)))) * (2 * np.pi)

    while accepted < n_samples:
        theta_candidate = np.random.uniform(-np.pi, np.pi)
        u = np.random.uniform(0, 1)
        accept_prob = np.exp(s1 * np.cos(theta_candidate) + s2 * np.sin(theta_candidate)) / C * (1 / (2 * np.pi))

        if u < accept_prob:
            accepted += 1
            samples.append(theta_candidate)

    return np.array(samples)

# Generate posterior samples
posterior_samples = ar(n_samples, s1, s2)

# Plot posterior distribution
st.subheader("Posterior Distribution of θ")
fig, ax = plt.subplots()
ax.hist(posterior_samples, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
ax.set_title("Posterior Distribution Histogram", fontsize=16)
ax.set_xlabel("θ", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
st.pyplot(fig)

# Additional Analysis
st.subheader("Further Analysis")
mean_theta = np.mean(posterior_samples)
std_theta = np.std(posterior_samples)

st.write(f"**Mean of θ:** {mean_theta:.2f}")
st.write(f"**Standard Deviation of θ:** {std_theta:.2f}")

# Summary statistics
st.sidebar.header("Summary Statistics")
st.sidebar.write(f"**Mean of θ:** {mean_theta:.2f}")
st.sidebar.write(f"**Standard Deviation of θ:** {std_theta:.2f}")

# 3D Plot of Multivariate Normal Distribution of Errors
st.subheader("3D Visualization of Multivariate Normal Distribution of Errors")

# Calculate the covariance matrix
errors = np.column_stack((difference_x[1:], difference_y[1:]))
cov_matrix = np.cov(errors, rowvar=False)

# Create a grid of points
x = np.linspace(np.min(difference_x), np.max(difference_x), 100)
y = np.linspace(np.min(difference_y), np.max(difference_y), 100)
X, Y = np.meshgrid(x, y)

# Calculate the multivariate normal distribution
pos = np.dstack((X, Y))
rv = multivariate_normal(mean=[0, 0], cov=cov_matrix)
Z = rv.pdf(pos)

# Plotting in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title("3D Multivariate Normal Distribution of Errors", fontsize=16)
ax.set_xlabel("X Errors", fontsize=14)
ax.set_ylabel("Y Errors", fontsize=14)
ax.set_zlabel("Density", fontsize=14)
st.pyplot(fig)

# Footer
st.markdown("---")
st.write("This analysis provides insights into animal movement patterns based on location data.")
st.write("For any questions or further information, please contact the presenter.")
