# --- DATA PREPARATION ---
# Synthetic data for the XOR gate
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from Practical9 import MLP

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Labels need to be a column vector
y = np.array([[0], [1], [1], [0]])

# --- VISUALIZATION SETUP ---

# Instantiate the MLP
mlp = MLP(input_size=2, hidden_size=5, output_size=1)

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Create a meshgrid to plot the decision boundary
h = .02  # step size in the mesh

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# This will store the loss history for the second plot
loss_history = []
frame_history = []

# Animation parameters
EPOCHS_PER_FRAME = 25
LEARNING_RATE = 0.25


# --- ANIMATION FUNCTION ---
def update(frame):
    """
    This function is called for each frame of the animation.
    It performs training and updates the plots.
    """
    # Train the network for a few epochs
    for epoch in range(EPOCHS_PER_FRAME):
        y_hat = mlp.forward(X)
        mlp.backward(X, y, y_hat, learning_rate=LEARNING_RATE)

    # --- Update Plot 1: Decision Boundary ---
    ax1.clear()

    # Get predictions for every point on the meshgrid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and margins
    ax1.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.75)

    # Plot the original XOR data points
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=120, edgecolors='k', cmap=plt.cm.RdYlBu)
    ax1.set_title(f"Decision Boundary (Epoch {(frame + 1) * EPOCHS_PER_FRAME})")
    ax1.set_xlabel("Input 1")
    ax1.set_ylabel("Input 2")
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())

    # --- Update Plot 2: Loss Curve ---
    # Calculate and store the current loss
    current_loss = np.mean((y - mlp.forward(X)) ** 2)
    loss_history.append(current_loss)
    frame_history.append((frame + 1) * EPOCHS_PER_FRAME)

    ax2.clear()
    ax2.plot(frame_history, loss_history, color='blue')
    ax2.set_title("Mean Squared Error Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_ylim(0, max(loss_history) * 1.1 if loss_history else 0.3)
    ax2.grid(True)

    # Add a global title to the figure
    fig.suptitle('MLP Training on XOR Problem', fontsize=16)



# --- CREATE AND RUN THE ANIMATION ---
# Create the animation object. `FuncAnimation` calls the `update` function for each frame.
# Total frames = 200, so total epochs = 200 * 25 = 5000
ani = FuncAnimation(fig, update, frames=220, interval=50, blit=False)
# To save the animation as a GIF:
print("Saving animation to 'mlp_training.gif'... Please wait.")
try:
    #ani.save('generated_plots/ann/mlp_training.gif', writer='pillow', fps=15)
    print("Animation saved to generated_plots/ann/mlp_training.gif")
except Exception as e:
    print(f"Could not save GIF. Error: {e}")
    print("Displaying animation instead.")

# Show the plot

plt.style.use('seaborn-v0_8-paper')
plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])  # Adjust layout to make room for suptitle
plt.show()