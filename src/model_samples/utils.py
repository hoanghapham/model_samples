import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def plot_series(losses, title=None, time_unit='epoch', figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(losses, linewidth=1)

    # Add labels and title
    ax.set_xlabel(time_unit, fontsize=10)
    ax.set_title(title, fontsize=12)

    # Customize x-axis ticks
    batch_count = len(losses)
    step = max(1, batch_count // 10)  # Show at most 10 ticks
    ax.set_xticks(range(0, batch_count, step))

    # Clean look
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    plt.show()


def plot_x_y_pred(x, y, pred, figsize=(8, 5),
                  title='Actual vs Predicted Values',
                  xlabel='X Values', ylabel='Y Values',
                  marker_size=50, alpha=0.7):
    # Validate inputs
    if len(x) != len(y) or len(x) != len(pred):
        raise ValueError("x, y, and pred must all have the same length")

    fig, ax = plt.subplots(figsize=figsize)

    # Create the scatter plots for actual (red) and predicted (blue) values
    actual = ax.scatter(x, y, s=marker_size, color='red', alpha=alpha,
                        edgecolors='white', label='Actual')
    prediction = ax.scatter(x, pred, s=marker_size, color='blue', alpha=alpha,
                           edgecolors='white', label='Predicted probability')

    # Add legend
    ax.legend()

    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)

    # Clean look
    plt.tight_layout()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()

def generate_data():
  data = torch.rand(1000, 2)
  label = ((data[:,0]+0.3*data[:,1]) > 0.5).to(torch.int)
  return data[:,0], label


