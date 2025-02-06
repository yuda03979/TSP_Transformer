import ast
import linecache
import os
from pathlib import Path
import random

import dask.dataframe as dd

from matplotlib import pyplot as plt


def save_graph(board_size: tuple[int, int], board: list, expected: list, predicted: list, save_path: str):
    """
    Visualize two TSP solutions using Matplotlib with larger arrows and save the image.
    The first and last points of the paths are NOT connected.
    """
    if predicted is None:
        predicted = expected

    plt.figure(figsize=board_size)
    x_board, y_board = zip(*board)

    # Plot cities
    plt.scatter(x_board, y_board, color='gray', s=100, label="Points")

    def plot_path(path, color, label, arrow_style, linestyle, lw=2, arrow_size=20):
        for i in range(len([i for i in path if
                            isinstance(i, tuple)]) - 1):  # Stop before the last point to avoid closing the loop
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle=arrow_style, color=color, lw=lw, mutation_scale=arrow_size))
            plt.plot([x1, x2], [y1, y2], linestyle=linestyle, color=color, lw=lw)

    # Plot first solution path (expected) with bold black arrows
    plot_path(expected, color="black", label="Expected", arrow_style='->', linestyle='-', lw=6, arrow_size=40)

    # Plot second solution path (predicted) with orange arrows
    plot_path(predicted, color="orange", label="Predicted", arrow_style='->', linestyle='-', lw=3, arrow_size=40)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Comparison of TSP_ Solutions")
    plt.legend()
    plt.grid()

    plt.xlim(0, board_size[0])
    plt.ylim(0, board_size[1])

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    os.utime(save_path)


def load_middle_row_from_file(datafile_path, idx):
    if isinstance(datafile_path, Path):
        datafile_path = str(datafile_path.resolve())
    with open(datafile_path, 'r') as data_file:

        # Skip lines before target
        for _ in range(idx):
            data_file.readline()

        line = data_file.readline().strip()
        board_str, solution_str, length_str = line.split('|')

        board = eval(board_str)
        solution = eval(solution_str)
        length = float(length_str)

        return board, solution, length


def gen_batch_line(file_path, idx, batch_rows):
    """Generator that reads a batch of valid lines from a file starting at `idx` and yields them one by one."""
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip lines until we reach the starting idx
        for _ in range(idx):
            f.readline()

        batch = []
        count = 0
        while count < batch_rows:
            line = f.readline()

            if not line:  # End of file
                break

            if line.count('|') == 2 and line.strip():  # Valid line check
                board_str, solution_str, length_str = line.strip().split('|')
                board, solution, length = eval(board_str), eval(solution_str), float(length_str)
                batch.append((board, solution, length))
                count += 1

    random.shuffle(batch)
    for line in batch:
        yield line

def get_len_data(datafile_path):
    with open(datafile_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
