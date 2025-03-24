import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from inference import process_file


class FileProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI prediction for Class III")
        self.setWindowIcon(QIcon('./assets/logo.png'))
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        #show as icon of the program the logo contained in ./assets/logo.png

        # Label to show the selected file path
        self.file_label = QLabel("No file selected", self)
        self.layout.addWidget(self.file_label)

        # Button to select a file
        self.select_button = QPushButton("Select File", self)
        self.select_button.clicked.connect(self.select_file)
        self.layout.addWidget(self.select_button)

        # Label for numerical value input
        self.num_label = QLabel("Enter the prediction time (yrs):", self)
        self.layout.addWidget(self.num_label)

        # Input field for numerical value
        self.num_entry = QLineEdit(self)
        self.layout.addWidget(self.num_entry)

        # Button to process and plot
        self.process_button = QPushButton("Predict", self)
        self.process_button.clicked.connect(self.process_and_plot)
        self.layout.addWidget(self.process_button)

        # Canvas for displaying the plot
        self.canvas_frame = QWidget(self)
        self.layout.addWidget(self.canvas_frame)

        self.selected_file = None
        self.setLayout(self.layout)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a File", "")

        if file_path:
            self.selected_file = file_path
            self.file_label.setText(f"Selected file: {file_path}")

    def process_and_plot(self):
        if not self.selected_file:
            QMessageBox.critical(self, "Error", "Please select a file first.")
            return

        try:
            # Get the numerical value from the entry box
            num_value = float(self.num_entry.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Please enter a valid numerical value.")
            return

        try:
            # Process the file using inference.py's function and pass the numerical value
            result = process_file(self.selected_file, num_value)

            # Plot the result
            self.plot_result(result, self.selected_file)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def plot_result(self, result, file_path):
        # Create a plot with the result (assuming result is a list or array)
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.plot()
        ax.plot(result[1][:,0], result[1][:,1], 'ro', label='Model Prediction')
        ax.plot(result[0][:,0], result[0][:,1], 'bo', label='Starting Coordinates')
        ax.set_title(f'Prediction after {self.num_entry.text()} years')
        ax.legend()
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Remove any previous plot from the canvas
        for widget in self.canvas_frame.findChildren(QWidget):
            widget.deleteLater()

        # Create a canvas to display the plot
        canvas = FigureCanvas(fig)
        layout = QVBoxLayout(self.canvas_frame)
        layout.addWidget(canvas)

        # Save the plot to the same directory as the input file
        output_path = os.path.join(os.path.dirname(file_path), "output_plot.png")
        fig.savefig(output_path)

        # Show a message box confirming the file is saved
        QMessageBox.information(self, "Success", f"Predictions saved to {output_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileProcessorApp()
    window.show()
    sys.exit(app.exec_())
