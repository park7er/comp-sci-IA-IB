import sys
import numpy as np
import time
import soundfile as sf
from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QPushButton, 
    QVBoxLayout, 
    QFileDialog, 
    QComboBox, 
    QWidget, 
    QLabel, 
    QGridLayout, 
    QMessageBox, 
    QLineEdit
)
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from pedals1.pedals import *
from GA import *



sound_effects = {
    "Tube Screamer": tube_screamer,
    "Big Muff": big_muff,
    "Boss DS1": boss_ds1,
    "Proco RAT": proco_rat
}

class GAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # setting up variables needed for later
        self.ga = None
        self.audio = None
        self.left_channel = None
        self.fitness_values = []
        self.best_fitness_values = []
        self.average_knobs_wave = None
        self.best_knobs_wave = None
        self.start_time = None
        self.elapsed_time = 0
        self.params = {}  # Dictionary to store user inputs

    def initUI(self):
        self.setWindowTitle('Genetic Algorithm for Sound Effects') #window title
        layout = QGridLayout() #the GUI is organized in a grid layout

        right_column = QVBoxLayout() # Right-side input column for algorithm parameters and device selection


        # Upload Button
        self.upload_btn = QPushButton('Upload Sound File', self)
        self.upload_btn.clicked.connect(self.upload_sound)
        right_column.addWidget(self.upload_btn)

        # Dropdown to select sound devices
        right_column.addWidget(QLabel('Select Device 1'))
        self.device1_combo = QComboBox(self)
        self.device1_combo.addItems(sound_effects.keys())
        right_column.addWidget(self.device1_combo)

        right_column.addWidget(QLabel('Select Device 2'))
        self.device2_combo = QComboBox(self)
        self.device2_combo.addItems(sound_effects.keys())
        right_column.addWidget(self.device2_combo)

        # Generation size
        right_column.addWidget(QLabel('Generation Size'))
        self.gen_size_input = QLineEdit(self)
        self.gen_size_input.setPlaceholderText("e.g., 300")
        right_column.addWidget(self.gen_size_input)

        right_column.addWidget(QLabel('Selection Size'))
        self.selection_size_input = QLineEdit(self)
        self.selection_size_input.setPlaceholderText("e.g., 100")
        right_column.addWidget(self.selection_size_input)
        
        self.mutation_rate_input = QLineEdit(self)
        self.mutation_rate_input.setPlaceholderText("e.g., 0.2")
        right_column.addWidget(QLabel('Mutation Rate (in decimal)'))
        right_column.addWidget(self.mutation_rate_input)

        right_column.addWidget(QLabel('Guassian mutation std (suggested between 0.05 and 0.025)'))
        self.guassian_std_input = QLineEdit(self)
        self.guassian_std_input.setPlaceholderText("e.g., 0.037")
        right_column.addWidget(self.guassian_std_input)

        # Distribution index
        right_column.addWidget(QLabel('Distribution Index for crossover'))
        self.dist_index_input = QLineEdit(self)
        self.dist_index_input.setPlaceholderText("e.g., 90")
        right_column.addWidget(self.dist_index_input)

        # Selection type
        right_column.addWidget(QLabel('Selection Type'))
        self.selection_type_combo = QComboBox(self)
        self.selection_type_combo.addItems(['Roulette', 'Tournament', 'Elite'])
        self.selection_type_combo.currentTextChanged.connect(self.toggle_tournament_input)
        right_column.addWidget(self.selection_type_combo)

        # Tournament size (disabled by default)
        right_column.addWidget(QLabel('Tournament Size'))
        self.tournament_size_input = QLineEdit(self)
        self.tournament_size_input.setDisabled(True)
        self.tournament_size_input.setPlaceholderText("e.g., 3")
        right_column.addWidget(self.tournament_size_input)

        layout.addLayout(right_column, 0, 4, 8, 1)  # Add right column layout to the grid

        # Run/Stop Buttons
        self.run_btn = QPushButton('Run Algorithm', self)
        self.run_btn.clicked.connect(self.run_algorithm)
        layout.addWidget(self.run_btn, 2, 0, 1, 2)

        self.stop_btn = QPushButton('Stop Algorithm', self)
        self.stop_btn.clicked.connect(self.stop_algorithm)
        layout.addWidget(self.stop_btn, 3, 0, 1, 2)

        # Display labels for stats
        self.generation_label = QLabel('Generation: N/A')
        self.runtime_label = QLabel('Run Time: 0 s')
        self.avg_rmse_label = QLabel('Average RMSE: N/A')
        self.best_rmse_label = QLabel('Smallest RMSE: N/A')

        layout.addWidget(self.generation_label, 4, 0, 1, 2)
        layout.addWidget(self.runtime_label, 4, 1, 1, 2)
        layout.addWidget(self.avg_rmse_label, 5, 0, 1, 2)
        layout.addWidget(self.best_rmse_label, 5, 1, 1, 2)

        # Graphs
        self.graphs = []
        graph_titles = [
            "Device 1 Waveform", "Device 2 Waveform",
            "Average RMSE", "Best RMSE",
            "Average Knobs Wave", "Best Knobs Wave"
        ]
        for row in range(3):
            for col in range(2):
                plot = pg.PlotWidget()
                plot.setTitle(graph_titles[row * 2 + col])
                layout.addWidget(plot, row + 7, col)
                self.graphs.append(plot)

        # Set main widget and layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer for GA steps and run time
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_ga)
        self.runtime_timer = QTimer()
        self.runtime_timer.timeout.connect(self.update_runtime)

    def toggle_tournament_input(self):
        """Enable or disable tournament size input based on selection type."""
        if self.selection_type_combo.currentText() == 'Tournament':
            self.tournament_size_input.setDisabled(False)
        else:
            self.tournament_size_input.setDisabled(True)


    def upload_sound(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Sound File", "", "Audio Files (*.wav *.flac *.ogg *.aiff)")
        if file_path:
            self.audio, self.sample_rate = sf.read(file_path)
            self.left_channel = self.audio[:, 0] if len(self.audio.shape) > 1 else self.audio
            self.upload_btn.setEnabled(False)

    def run_algorithm(self):
        if self.left_channel is None:
            self.show_error_message("No Sound File Uploaded", "Please upload a sound file before running the algorithm.")
            return

        try:
            # Retrieve and store user inputs, use defaults if empty
            mutation_rate_text = self.mutation_rate_input.text().strip()
            self.params['mutation_rate'] = float(mutation_rate_text) if mutation_rate_text else 0.3

            dist_index_text = self.dist_index_input.text().strip()
            self.params['dist_index'] = float(dist_index_text) if dist_index_text else 50.0

            pop_size_text = self.gen_size_input.text().strip()
            self.params['pop_size'] = int(pop_size_text) if pop_size_text else 300

            selection_size_text = self.selection_size_input.text().strip()
            self.params['selection_size'] = int(selection_size_text) if selection_size_text else 100

            selection_type = self.selection_type_combo.currentText()
            self.params['selection_type'] = selection_type

            guassian_std_text = self.guassian_std_input.text().strip()
            self.params['Guassian mutation std'] = float(guassian_std_text) if guassian_std_text else 0.037

            if selection_type == 'Tournament':
                tournament_size_text = self.tournament_size_input.text().strip()
                if not tournament_size_text:
                    self.params['tournament_size'] = 3  # default tournament size
                else:
                    self.params['tournament_size'] = int(tournament_size_text)
            else:
                self.params['tournament_size'] = None 

            # Validate inputs
            errors = []
            if self.params['pop_size'] < self.params['selection_size']:
                errors.append("Generation size must be larger than or equal to Selection size.")
            if self.params['mutation_rate'] < 0 or self.params['mutation_rate'] > 1:
                errors.append("Mutation rate cannot be negative or more than 1")
            if self.params['Guassian mutation std'] < 0:
                errors.append("Gaussian mutation std cannot be negative.")
            if self.params['dist_index'] < 0:
                errors.append("Distribution index cannot be negative.")
            if self.params['selection_type'] == 'Tournament' and (self.params['tournament_size'] is None or self.params['tournament_size'] <= 0):
                errors.append("Tournament size must be a positive integer.")

            if errors:
                self.show_error_message("Invalid Input", "\n".join(errors))
                return

        except ValueError:
            self.show_error_message("Invalid Input", "Please enter valid numbers for the parameters.")
            return

        device1_name = self.device1_combo.currentText()
        device2_name = self.device2_combo.currentText()

        self.ga = GeneticAlgorithm(
            self.params.get('pop_size'),  # population size
            self.left_channel,
            sound_effects[device1_name],
            sound_effects[device2_name],
            self.params
        )
        self.fitness_values = []
        self.best_fitness_values = []

        # Plot initial sound waves for both devices
        wave1 = self.get_waveform(device1_name, self.left_channel)
        wave2 = self.get_waveform(device2_name, self.left_channel)

        # Clear previous plots and plot sound waves
        self.clear_and_plot(self.graphs[0], wave1, 'b')  # Device 1
        self.clear_and_plot(self.graphs[1], wave2, 'r')  # Device 2

        # Start timers
        self.start_time = time.time()
        self.runtime_timer.start(1000)  # Update every second
        self.timer.start(100)  # GA step timer

        
    def stop_algorithm(self):
        """Stop the algorithm and save the best individual's nulled sound wave signal to a file."""
        self.timer.stop()
        self.runtime_timer.stop()

        # Get the best individual's knobs
        best_knobs1 = self.ga.best_individual.knobs1  # Best knobs for device 1
        best_knobs2 = self.ga.best_individual.knobs2  # Best knobs for device 2

        # Calculate the nulled sound wave using the calculate_knobs_wave function
        nulled_wave = self.calculate_knobs_wave(best_knobs1, best_knobs2)

        # Save the nulled sound wave to a file in the user's directory
        save_dialog = QFileDialog(self)
        save_path, _ = save_dialog.getSaveFileName(self, "Save Nulled Sound Wave", "", "WAV Files (*.wav)")
        
        if save_path:
            sf.write(save_path, nulled_wave, self.sample_rate)  # Save the nulled wave as a .wav file
            QMessageBox.information(self, "Save Successful", f"File saved to: {save_path}")
        else:
            QMessageBox.warning(self, "Save Cancelled", "No file was saved.")


    def step_ga(self):
        avg_fitness, best_fitness, best_knobs1, best_knobs2, avg_knobs1, avg_knobs2 = self.ga.run_generation()
        self.fitness_values.append(avg_fitness)
        self.best_fitness_values.append(best_fitness)

        # updating labels
        self.current_generation = self.ga.generation        
        self.generation_label.setText(f'Generation: {self.current_generation}')
        self.avg_rmse_label.setText(f'Average RMSE: {avg_fitness:.4f}')
        self.best_rmse_label.setText(f'Smallest RMSE: {best_fitness:.4f}')

        # Only show last 20 generations
        self.plot_generations(self.graphs[2], self.fitness_values, 'g')  # Avg RMSE
        self.plot_generations(self.graphs[3], self.best_fitness_values, 'r')  # Best RMSE

        self.average_knobs_wave = self.calculate_knobs_wave(avg_knobs1, avg_knobs2)
        self.best_knobs_wave = self.calculate_knobs_wave(best_knobs1, best_knobs2)
        print(best_knobs1, best_knobs2)

        if self.current_generation%5==0 or self.current_generation==1:
            self.clear_and_plot(self.graphs[4], self.average_knobs_wave, 'b')  # Avg knobs
            self.clear_and_plot(self.graphs[5], self.best_knobs_wave, 'r')  # Best knobs

    def update_runtime(self):
        elapsed_time = int(time.time() - self.start_time)
        self.runtime_label.setText(f"Run Time: {elapsed_time} s")
    
    def calculate_knobs_wave(self, knobs1, knobs2):
        wave1 = self.ga.device1(self.left_channel, *knobs1)
        wave2 = self.ga.device2(self.left_channel, *knobs2)
        return np.array(wave1) - np.array(wave2)

    def plot_generations(self, graph, values, color):
        graph.clear()
        if len(values) > 20:
            x_values = range(len(values) - 20, len(values))
            y_values = values[-20:]
        else:
            x_values = range(len(values))
            y_values = values
        graph.plot(x_values, y_values, pen=pg.mkPen(color, width=2))
    
    def clear_and_plot(self, graph, data, color):
        graph.clear()
        graph.plot(np.arange(len(data)), data, pen=pg.mkPen(color, width=1))
    
    def get_waveform(self, device_name, audio):
        return np.array(sound_effects[device_name](audio, 4, 5, 5))

    def show_error_message(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()


# Run the PyQt5 Application
app = QApplication([])
window = GAApp()
window.show()
sys.exit(app.exec_())
#sys.argv
