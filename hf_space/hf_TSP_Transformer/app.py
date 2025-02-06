import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os

from solver import Solver
from config import Config, API, load_pretrained_model
from infer import Infer
from model import TSPTransformer
from tokenizer import Tokenizer


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 165, 255)


class App:

    def __init__(self, board_size: tuple, infer):
        self.board_size = board_size

        self.infer = infer

        self.grid_size = board_size[0]
        self.cell_size = 50
        self.canvas_size = self.grid_size * self.cell_size

        self.clicked_cells = set()

    def create_grid(self):
        img = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255

        for (cell_x, cell_y) in self.clicked_cells:
            x1, y1 = cell_x * self.cell_size, (self.grid_size - 1 - cell_y) * self.cell_size
            x2, y2 = x1 + self.cell_size, y1 + self.cell_size
            img[y1:y2, x1:x2] = ORANGE

        for i in range(0, self.canvas_size, self.cell_size):
            img[:, i:i + 1] = BLACK  # Vertical lines
            img[i:i + 1, :] = BLACK  # Horizontal lines

        return img

    def capture_clicks(self, evt: gr.SelectData):
        x, y = evt.index[0], evt.index[1]

        cell_x = x // self.cell_size
        cell_y = self.grid_size - 1 - (y // self.cell_size)
        cell = (cell_x, cell_y)

        self.clicked_cells.add(cell)

        return self.create_grid()

    def save_graph(self, board_size, board, expected, predicted, save_path):
        if predicted is None:
            predicted = expected

        plt.figure(figsize=board_size)
        x_board, y_board = zip(*[(x, y) for x, y in board])
        plt.scatter(x_board, y_board, color='gray', s=100, label="Points")

        def plot_path(path, color, label, arrow_style, linestyle, lw=2, arrow_size=20):
            for i in range(len([i for i in path if isinstance(i, tuple)]) - 1):
                x1, y1 = path[i][0], path[i][1]
                x2, y2 = path[i + 1][0], path[i + 1][1]
                plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                             arrowprops=dict(arrowstyle=arrow_style, color=color, lw=lw, mutation_scale=arrow_size))
                plt.plot([x1, x2], [y1, y2], linestyle=linestyle, color=color, lw=lw)

        plot_path(expected, color="black", label="Expected", arrow_style='->', linestyle='-', lw=6, arrow_size=40)
        plot_path(predicted, color="orange", label="Predicted", arrow_style='->', linestyle='-', lw=3, arrow_size=40)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Comparison of TSP Solutions")
        plt.legend()
        plt.grid()
        plt.xlim(0, board_size[0])
        plt.ylim(0, board_size[1])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        os.utime(save_path)

    # Function to process and return a path and the generated graph
    def process_and_draw_path(self):
        expected = Solver.solve(list(self.clicked_cells))[0]  # Example: Sort by x, then y (basic path)
        predicted = self.infer.infer(list(self.clicked_cells))
        save_path = "tsp_solution.png"
        self.save_graph(self.board_size, list(self.clicked_cells), expected, predicted, save_path)
        return self.create_grid(), save_path

    def clear_grid(self):
        self.clicked_cells = set()
        return self.create_grid(), None


if __name__ == '__main__':

    config = Config()
    model = TSPTransformer(config.config_model).to(config.device)
    load_pretrained_model(config, model)
    tokenizer = API.tokenizer(config)
    infer = API.infer(config, model, tokenizer)

    app = App(board_size=config.config_datasets.board_size, infer=infer)

    with gr.Blocks() as demo:
        with gr.Row():
            image = gr.Image(value=app.create_grid(), interactive=True)
            output_image = gr.Image()
        send_button = gr.Button("Send")
        clear_button = gr.Button("Clear")

        image.select(app.capture_clicks, None, image)
        send_button.click(app.process_and_draw_path, None, [image, output_image])
        clear_button.click(app.clear_grid, None, [image, output_image])

    demo.launch()
