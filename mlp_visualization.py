import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch, bbox_artist

# --- DATA & NETWORK CONFIGURATION ---
# We'll use one sample from the XOR dataset to demonstrate a single learning step.
X_input = np.array([1, 0])
y_target = np.array([1])

# Initialize the MLP with random weights and biases.
# This ensures the network starts "untrained".
np.random.seed(42)  # for reproducible results
input_size, hidden_size, output_size = 2, 2, 1
W1 = np.random.randn(input_size, hidden_size) * 0.5
b1 = np.random.randn(1, hidden_size) * 0.5
W2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.random.randn(1, output_size) * 0.5

# Hyperparameter
LEARNING_RATE = 0.5


# --- HELPER FUNCTIONS ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


# --- VISUAL & ANIMATION SETTINGS ---
FRAME_COUNT = 320
INTERVAL = 100  # ms between frames

NODE_POS = {
    'input': [(-4, 1), (-4, -1)],
    'hidden': [(-1, 1), (-1, -1)],
    'output': [(2, 0)]
}

# Colors for different states and elements
C = {
    'inactive': 'skyblue', 'line': 'gray', 'text': '#333333',
    'input_active': '#90EE90', 'hidden_active': '#FFD700', 'output_active': '#FFA07A',
    'error': '#FF4C4C', 'gradient': '#8A2BE2', 'update': '#32CD32'
}

# --- SETUP THE PLOT ---
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlim(-5, 5)
ax.set_ylim(-3, 3)
ax.axis('off')

calc_values = {}

def draw_network(weights, biases):
    """Draws the static network structure."""
    # Nodes
    for layer, positions in NODE_POS.items():
        for i, pos in enumerate(positions):
            ax.add_patch(Circle(pos, 0.3, color=C['inactive'], ec='black', zorder=5))
            if layer == 'input':
                ax.text(pos[0], pos[1], f'x{i + 1}',font="Fira Code", ha='center', va='center', fontsize=14, zorder=10)
            elif layer == 'hidden':
                ax.text(pos[0], pos[1], f'h{i + 1}',font="Fira Code", ha='center', va='center', fontsize=14, zorder=10)
            else:
                ax.text(pos[0], pos[1], 'ŷ', font="Fira Code", ha='center', va='center', fontsize=14, zorder=10)

    # Connections and Weights
    for i in range(2):  # Input to Hidden
        for j in range(2):
            start, end = NODE_POS['input'][i], NODE_POS['hidden'][j]
            ax.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=0",
                                         arrowstyle="-", mutation_scale=20, color=C['line'], zorder=1))
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.1, f"w={weights['W1'][i, j]:.2f}", ha='center',
                    va='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.75, boxstyle="round,pad=0.1"))
    for i in range(2):  # Hidden to Output
        start, end = NODE_POS['hidden'][i], NODE_POS['output'][0]
        ax.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=0",
                                     arrowstyle="-", mutation_scale=20, color=C['line'], zorder=1))
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, f"w={weights['W2'][i, 0]:.2f}", ha='center',
                va='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.75, boxstyle="round,pad=0.1"))

    # Biases
    for i, pos in enumerate(NODE_POS['hidden']):
        ax.text(pos[0], pos[1] + 0.4, f"b={biases['b1'][0, i]:.2f}", ha='center', fontsize=11, color=C['text'])
        ax.text(NODE_POS['output'][0][0], NODE_POS['output'][0][1] + 0.4, f"b={biases['b2'][0, 0]:.2f}", ha='center',
            fontsize=11, color=C['text'])


# --- MAIN ANIMATION FUNCTION ---
def update(frame):
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    global W1, b1, W2, b2
    # Redraw network in every frame
    draw_network({'W1': W1, 'W2': W2}, {'b1': b1, 'b2': b2})

    global calc_values

    # --- SCENE MANAGEMENT ---
    # Scene 0: Initial State
    if frame < 20:
        ax.set_title("MLP for XOR: A Single Learning Step", fontsize=16)
        ax.text(-4, 2.5, f"Input: x = {X_input}", fontsize=14)
        ax.text(0, 2.6, f"Target: y = {y_target[0]}", fontsize=14)

    # == PHASE 1: FEED-FORWARD ==
    elif frame < 50:
        ax.set_title("Phase 1: Feed-Forward (Input -- Hidden)", fontsize=16)
        # Highlight input nodes
        for i, pos in enumerate(NODE_POS['input']):
            ax.add_patch(Circle(pos, 0.3, color=C['input_active'], ec='black', zorder=6))
            ax.text(pos[0], pos[1], f'{X_input[i]}', ha='center', va='center', fontsize=14, zorder=10)

        # Calculate Z1 (weighted sum for hidden layer)
        calc_values['Z1'] = np.dot(X_input.reshape(1, -1), W1) + b1
        ax.text(-1, 2,
                f"z_h1 = (x1 * w_11 + x2 * w_21) + b1 = ({X_input[0]} * {W1[0, 0]:.2f} + {X_input[1]} * {W1[1, 0]:.2f})\n + {b1[0, 0]:.2f} = {calc_values['Z1'][0, 0]:.2f}",
                fontsize=14,font="Fira Code", bbox=dict(facecolor="indigo", alpha=0.40))

        ax.text(-1, 1.6,
                f"z_h2 = (x1 * w_12 + x2 * w_22) + b2 = ({X_input[0]} * {W1[0, 1]:.2f} + {X_input[1]}*{W1[1, 1]:.2f})\n + {b1[0, 1]:.2f} = {calc_values['Z1'][0, 1]:.2f}",
                fontsize=14,font="Fira Code" ,bbox=dict(facecolor="purple", alpha=0.45))

    elif frame < 80:
        ax.set_title("Phase 1: Feed-Forward (Hidden Layer Activation)", fontsize=16)
        # Calculate A1 (activation for hidden layer)
        calc_values['A1'] = sigmoid(calc_values['Z1'])
        # Highlight hidden nodes with their activation values
        for i, pos in enumerate(NODE_POS['hidden']):
            ax.add_patch(Circle(pos, 0.3, color=C['hidden_active'], ec='black', zorder=6))
            ax.text(pos[0], pos[1], f"A={calc_values['A1'][0, i]:.2f}", ha='center', va='center', fontsize=12,
                    zorder=10, font="Fira Code")
        ax.text(-1, 2, f"A_h1 = σ(z_h1) = σ({calc_values['Z1'][0, 0]:.2f})\n = {calc_values['A1'][0, 0]:.2f}",
                fontsize=14, font="Fira Code")
        ax.text(-1, 1.6, f"A_h2 = σ(z_h2) = σ({calc_values['Z1'][0, 1]:.2f})\n = {calc_values['A1'][0, 1]:.2f}",
                fontsize=14, font="Fira Code")

    elif frame < 110:
        ax.set_title("Phase 1: Feed-Forward (Hidden -> Output)", fontsize=16)
        # Highlight hidden nodes
        for i, pos in enumerate(NODE_POS['hidden']):
            ax.add_patch(Circle(pos, 0.3, color=C['hidden_active'], ec='black', zorder=6))
        # Calculate Z2 (weighted sum for output)
        calc_values['Z2'] = np.dot(calc_values['A1'], W2) + b2
        ax.text(2, 2, f"z_out = (A_h1 * w_31 + A_h2 * w_41) + b_out",font="Fira Code", fontsize=13)
        ax.text(2, 1.6,
                f"z_out = ({calc_values['A1'][0, 0]:.2f}*{W2[0, 0]:.2f} + {calc_values['A1'][0, 1]:.2f}*{W2[1, 0]:.2f})\n + {b2[0, 0]:.2f} = {calc_values['Z2'][0, 0]:.2f}",
                fontsize=13, font="Fira Code")

    elif frame < 130:
        ax.set_title("Phase 1: Feed-Forward (Final Prediction ŷ)", fontsize=16)
        # Calculate y_hat (final prediction)
        calc_values['y_hat'] = sigmoid(calc_values['Z2'])
        # Highlight output node
        ax.add_patch(Circle(NODE_POS['output'][0], 0.3, color=C['output_active'], ec='black', zorder=6))
        ax.text(NODE_POS['output'][0][0], NODE_POS['output'][0][1], f"ŷ={calc_values['y_hat'][0, 0]:.2f}", ha='center',
                va='center', fontsize=14, zorder=10, font="Fira Code")
        ax.text(2, 2, f"ŷ = σ(z_out) = σ({calc_values['Z2'][0, 0]:.2f})\n = {calc_values['y_hat'][0, 0]:.2f}",
                fontsize=14, font="Fira Code")

    # == PHASE 2: LOSS CALCULATION ==
    elif frame < 150:
        ax.set_title("Phase 2: Loss Calculation", fontsize=16, color=C['error'])
        calc_values['error'] = y_target - calc_values['y_hat']
        calc_values['loss'] = 0.5 * np.power(calc_values['error'], 2)
        ax.text(0, -2.5,
                f"Loss = ½ * (y - ŷ)² = 0.5 * ({y_target[0]} - {calc_values['y_hat'][0, 0]:.2f})²\n = {calc_values['loss'][0, 0]:.2f}",
                fontsize=15, ha='center', bbox=dict(facecolor=C['error'], alpha=0.5), font="Fira Code")

    # == PHASE 3: BACKPROPAGATION ==
    elif frame < 180:
        ax.set_title("Phase 3: Backpropagation (Output Layer Gradient)", fontsize=16, color=C['gradient'])
        # Calculate output delta
        calc_values['delta_output'] = calc_values['error'] * sigmoid_derivative(calc_values['y_hat'])
        # Calculate gradients for W2 and b2
        calc_values['dW2'] = np.dot(calc_values['A1'].T, calc_values['delta_output'])
        calc_values['db2'] = np.sum(calc_values['delta_output'], axis=0)
        ax.text(2, -1.4, "δ_out = (ŷ - y) * σ'(z_out)", fontsize=14, font="Fira Code")
        ax.text(2, -1.9,
                f"δ_out = ({calc_values['y_hat'][0, 0]:.2f} - {y_target[0]}) *\n ... = {calc_values['delta_output'][0, 0]:.2f}",
                fontsize=12, font="Fira Code")
        ax.text(2, -2.5, f"∂Loss/∂W2 = A1.T * δ_out = {calc_values['dW2'][0, 0]:.2f}, {calc_values['dW2'][1, 0]:.2f}",
                fontsize=14, font="Fira Code")

    elif frame < 220:
        ax.set_title("Phase 3: Backpropagation (Hidden Layer Gradient)", fontsize=16, color=C['gradient'], font="Fira Code")
        # Propagate error backwards
        for i, pos in enumerate(NODE_POS['hidden']):
            start, end = NODE_POS['output'][0], pos
            ax.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=0.2",
                                         arrowstyle="->", mutation_scale=20, color=C['gradient'], zorder=1, ls='--'))
        # Calculate hidden delta
        calc_values['delta_hidden'] = np.dot(calc_values['delta_output'], W2.T) * sigmoid_derivative(calc_values['A1'])
        # Calculate gradients for W1 and b1
        calc_values['dW1'] = np.dot(X_input.reshape(-1, 1), calc_values['delta_hidden'])
        calc_values['db1'] = np.sum(calc_values['delta_hidden'], axis=0)
        ax.text(-1, -1.4, f"δ_hidden = δ_out * W2.T * σ'(z_hidden)", fontsize=14, font="Fira Code")
        ax.text(-1, -1.9, f"δ_h = ... (result is matrix)", fontsize=14, font="Fira Code")
        ax.text(-1, -2.4, f"∂Loss/∂W1 = X.T * δ_hidden = ...", fontsize=14, font="Fira Code")

    # == PHASE 4: PARAMETER UPDATE ==
    elif frame < 280:
        ax.set_title("Phase 4: Parameter Update (Gradient Descent)", fontsize=16, color=C['update'])
        ax.text(0, -2.5, "w_new = w_old - α * ∂Loss/∂w", fontsize=16, ha='center',
                bbox=dict(facecolor=C['update'], alpha=0.5), font="Fira Code")
        # Display updates for W2
        ax.text(2, 1, f"W2_new = {W2[0, 0]:.2f}\n - {LEARNING_RATE}*{calc_values['dW2'][0, 0]:.2f}", fontsize=14, font="Fira Code")
        ax.text(2, 0.5, f"W2_new = {W2[1, 0]:.2f} - {LEARNING_RATE}*{calc_values['dW2'][1, 0]:.2f}", fontsize=14, font="Fira Code")

    # Final Scene shows the updated network
    else:
        # Update weights and biases for the final frame
    #    global W1, b1, W2, b2
        W1 -= LEARNING_RATE * calc_values['dW1']
        b1 -= LEARNING_RATE * calc_values['db1']
        W2 -= LEARNING_RATE * calc_values['dW2']
        b2 -= LEARNING_RATE * calc_values['db2']

        ax.clear()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-3, 3)
        ax.axis('off')
        ax.set_title("Cycle Complete: Parameters Have Been Updated!", fontsize=16, color=C['update'])
        draw_network({'W1': W1, 'W2': W2}, {'b1': b1, 'b2': b2})
        ax.text(0, -2.5, "The network is now slightly less 'wrong'.\nThis process repeats thousands of times.",
                fontsize=16, font = "Fira Code" ,ha='center', va='center', bbox = dict(facecolor="yellow", alpha=.4))




# --- RUN THE ANIMATION ---
if __name__ == "__main__":
    print("Creating MLP learning cycle animation... This may take a minute.")
    ani = FuncAnimation(fig, update, frames=FRAME_COUNT, interval=INTERVAL, blit=False)
    try:
        ani.save('generated_plots/ann/mlp_learning_cycle.gif', writer='pillow', fps=10)
        print("Animation saved successfully")
    except Exception as e:
        print(f"Could not save GIF. Error: {e}\nDisplaying animation instead.")
    plt.show()

print({k : np.round(v, 2) for k, v in calc_values.items()})
