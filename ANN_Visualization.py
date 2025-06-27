import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arrow

# --- CONFIGURATION & DATA ---
# Let's visualize one forward pass for the AND gate.
# We will use an input that should result in '1'.
INPUT_X = [1, 1]
LABEL_Y = 1

# We'll use the pre-trained weights and bias that we know solve the AND problem.
WEIGHTS_W = [2, 2]
BIAS_B = -3

# --- ANIMATION SETTINGS ---
# We control the animation by dividing it into scenes based on the frame number.
TOTAL_FRAMES = 220
INTERVAL = 80  # Milliseconds between frames

# --- VISUAL STYLES ---
NODE_RADIUS = 0.25
INACTIVE_COLOR = 'skyblue'
ACTIVE_COLOR = '#90EE90'  # A light green
LINE_COLOR = 'gray'
TEXT_COLOR = 'black'
HIGHLIGHT_COLOR = 'gold'

# Define positions for all our visual elements
positions = {
    'input_1': (-2, 0.5),
    'input_2': (-2, -0.5),
    'perceptron': (0, 0),
    'output': (2, 0),
    'w1_text': (-1, 0.6),
    'w2_text': (-1, -0.4),
    'bias_text': (0, 0.3),
    'sum_text': (-0.1, -0.4),
    'activation_text': (-0.1, -0.6),
    'final_text': (0, -1.2)
}

# --- SETUP THE PLOT ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.axis('off')  # Hide the axes for a cleaner look


# --- ANIMATION FUNCTION ---
def update(frame):
    ax.clear()  # Clear the previous frame
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.axis('off')

    # --- STATIC ELEMENTS (present in all frames) ---
    # Draw connections
    ax.plot([positions['input_1'][0], positions['perceptron'][0]],
            [positions['input_1'][1], positions['perceptron'][1]], color=LINE_COLOR, zorder=1)
    ax.plot([positions['input_2'][0], positions['perceptron'][0]],
            [positions['input_2'][1], positions['perceptron'][1]], color=LINE_COLOR, zorder=1)
    ax.plot([positions['perceptron'][0], positions['output'][0]],
            [positions['perceptron'][1], positions['output'][1]], color=LINE_COLOR, zorder=1)

    # Draw nodes
    input1_node = Circle(positions['input_1'], NODE_RADIUS, color=INACTIVE_COLOR, ec='black', zorder=2)
    input2_node = Circle(positions['input_2'], NODE_RADIUS, color=INACTIVE_COLOR, ec='black', zorder=2)
    perceptron_node = Circle(positions['perceptron'], NODE_RADIUS * 1.25, color=INACTIVE_COLOR, ec='black', zorder=2)
    output_node = Circle(positions['output'], NODE_RADIUS, color=INACTIVE_COLOR, ec='black', zorder=2)

    ax.add_patch(input1_node)
    ax.add_patch(input2_node)
    ax.add_patch(perceptron_node)
    ax.add_patch(output_node)

    # --- ANIMATION SCENES ---

    # Scene 1: Initial State - Show inputs, weights, and bias
    if frame < 25:
        ax.set_title("Scene 1: Initial State", fontsize=16)
        # Show input values
        ax.text(*positions['input_1'], s = f'x₁ = {INPUT_X[0]}', ha='center', va='center', fontsize=12.5)
        ax.text(*positions['input_2'], s = f'x₂ = {INPUT_X[1]}', ha='center', va='center', fontsize=12.5)
        # Show weights and bias
        ax.text(*positions['w1_text'], s = f'w₁ = {WEIGHTS_W[0]}', ha='center', fontsize=12.5)
        ax.text(*positions['w2_text'], s = f'w₂ = {WEIGHTS_W[1]}', ha='center', fontsize=12.5)
        ax.text(*positions['bias_text'], s= f'bias (b) = {BIAS_B}', ha='center', fontsize=12.5, color='darkred')

    # Scene 2: Inputs are Processed - Weighted Sum Calculation
    elif frame < 80:
        ax.set_title("Scene 2: Calculating the Weighted Sum (z)", fontsize=16)
        # Highlight active inputs
        input1_node.set_color(ACTIVE_COLOR)
        input2_node.set_color(ACTIVE_COLOR)
        # Show calculation for weighted inputs
        ax.text(*positions['w1_text'], s = f'w₁ * x₁ = ({WEIGHTS_W[0]} * {INPUT_X[0]})\n = {WEIGHTS_W[0] * INPUT_X[0]}',
                ha='center', fontsize=12, bbox=dict(facecolor=HIGHLIGHT_COLOR, alpha=0.6))
        ax.text(*positions['w2_text'], s = f'w₂ * x₂ = ({WEIGHTS_W[1]} * {INPUT_X[1]})\n = {WEIGHTS_W[1] * INPUT_X[1]}',
                ha='center', fontsize=12, bbox=dict(facecolor=HIGHLIGHT_COLOR, alpha=0.5))
        ax.text(*positions['bias_text'], s = f'bias (b) = {BIAS_B}', ha='center', fontsize=13, color='darkred')
        # Show the formula for z
        ax.text(positions['perceptron'][0], positions['perceptron'][1] + 0.4, 'z = (w₁*x₁) + (w₂*x₂) + b', ha='center',
                fontsize=14)

    # Scene 3: Summation Result
    elif frame < 120:
        ax.set_title("Scene 3: Summation Result", fontsize=16)
        input1_node.set_color(ACTIVE_COLOR)
        input2_node.set_color(ACTIVE_COLOR)
        perceptron_node.set_color(HIGHLIGHT_COLOR)
        # Show the full calculation inside the perceptron
        calc_text = f"z = {WEIGHTS_W[0] * INPUT_X[0]}\n + {WEIGHTS_W[1] * INPUT_X[1]}\n + ({BIAS_B})"
        result_text = f"z = {sum(w * x for w, x in zip(WEIGHTS_W, INPUT_X)) + BIAS_B}"
        ax.text(*positions['perceptron'], s = f'{calc_text}\n{result_text}', ha='center', va='center', fontsize=14,
                weight='bold')

    # Scene 4: Applying the Activation Function
    elif frame < 170:
        ax.set_title("Scene 4: Applying the Activation Function", fontsize=16)
        z_value = sum(w * x for w, x in zip(WEIGHTS_W, INPUT_X)) + BIAS_B
        perceptron_node.set_color(ACTIVE_COLOR)  # Activate the neuron
        ax.text(*positions['perceptron'], s = f'z = {z_value}', ha='center', va='center', fontsize=14, weight='bold')
        # Explain the step function
        ax.text(*positions['sum_text'], s = 'Step Function Logic:', ha='right', fontsize=13)
        ax.text(*positions['activation_text'], s = f'if z >= 0, output is 1\nSince {z_value} >= 0, we activate!',
                ha='right', fontsize=13, weight='bold', bbox=dict(facecolor=ACTIVE_COLOR, alpha=0.7))

    # Scene 5: Output Prediction
    elif frame < 200:
        ax.set_title("Scene 5: Final Output Prediction (ŷ)", fontsize=16)
        perceptron_node.set_color(ACTIVE_COLOR)
        output_node.set_color(ACTIVE_COLOR)
        predicted_y = 1
        ax.text(*positions['output'], s = f'ŷ = {predicted_y}', ha='center', va='center', fontsize=14, weight='bold')

    # Scene 6: Conclusion - Compare with True Label
    else:
        ax.set_title("Scene 6: Conclusion - Prediction is Correct!", fontsize=16)
        perceptron_node.set_color(ACTIVE_COLOR)
        output_node.set_color(ACTIVE_COLOR)
        predicted_y = 1
        ax.text(*positions['output'], s = f'ŷ = {predicted_y}', ha='center', va='center', fontsize=14, weight='bold')
        # Show comparison
        ax.text(positions['final_text'][0], positions['final_text'][1],
                f'Predicted Value (ŷ): {predicted_y}\n'
                f'True Label (y): {LABEL_Y}\n\n'
                'The prediction matches the label! ✅',
                ha='center', va='center', fontsize=14,
                bbox=dict(facecolor='lightgreen', alpha=0.6))


# --- RUN THE ANIMATION ---
if __name__ == "__main__":
    print("Starting perceptron animation... Close the plot window to exit.")
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=INTERVAL, blit=False)
    plt.tight_layout()
    plt.show()
    print("Animation finished.")