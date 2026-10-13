from pedals1.pedals import *
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error

class Individual:
    def __init__(self, knobs1, knobs2):
        self.knobs1 = knobs1  # For Device 1
        self.knobs2 = knobs2  # For Device 2
        self.fitness = None

    def evaluate(self, audio, device1, device2):
        wave1 = device1(audio, *self.knobs1)
        wave2 = device2(audio, *self.knobs2)
        self.fitness = np.sqrt(mean_squared_error(wave1, wave2))
        return self.fitness

class GeneticAlgorithm:
    def __init__(self, population_size, audio, device1, device2, params):
        self.population_size = population_size
        self.audio = audio
        self.device1 = device1
        self.device2 = device2
        self.params = params  # Store the params dictionary
        self.generation = 0
        self.population = [self.create_individual() for _ in range(population_size)]
        self.best_individual = None
        self.best_fitness = float('inf')
        self.best_knobs1 = None
        self.best_knobs2 = None
        self.avg_knobs1 = None
        self.avg_knobs2 = None
        self.mutation_rate = self.params.get('mutation_rate')  # Starting mutation rate
        self.eta_c = self.params.get('dist_index')
        self.selection_size = self.params.get('selection_size')
        self.selection_type = self.params.get('selection_type')
        self.guassianstd = self.params.get('Guassian mutation std')
        if self.selection_type == 'Tournament':
            self.tourn_size = self.params.get('tournament_size')

    def create_individual(self):
        # First knob (volume) initialized between 1 and 10
        knobs1 = np.random.uniform(1, 10, 1)
        knobs2 = np.random.uniform(1, 10, 1)
        
        # Other knobs initialized between 0 and 10
        knobs1 = np.append(knobs1, np.random.uniform(0, 10, 2))
        knobs2 = np.append(knobs2, np.random.uniform(0, 10, 2))
        
        return Individual(knobs1, knobs2)

    def run_generation(self):
        # Evaluate fitness for all individuals
        fitness_values = []
        for individual in self.population:
            fitness = individual.evaluate(self.audio, self.device1, self.device2)
            fitness_values.append(fitness)
            
            # Track the best individual with the lowest RMSE
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual
                self.best_knobs1 = individual.knobs1
                self.best_knobs2 = individual.knobs2
            
            if fitness == 0:
                print("Solution found!")
                print(f"Knobs1 (Device 1): {self.best_knobs1}")
                print(f"Knobs2 (Device 2): {self.best_knobs2}")
                return None  # Terminate the algorithm
        
        # Compute average fitness
        avg_fitness = np.mean(fitness_values)
        
        # Calculate average knobs for the entire population
        self.avg_knobs1 = np.mean([ind.knobs1 for ind in self.population], axis=0)
        self.avg_knobs2 = np.mean([ind.knobs2 for ind in self.population], axis=0)


        # Select individuals using roulette wheel selection
        if self.selection_type == 'Tournament':
            selected_individuals = tournament_selection(self.population, self.selection_size, self.tourn_size)
        elif self.selection_type == 'Elite':
            selected_individuals = elite_selection(self.population, self.selection_size)
        else:
            selected_individuals = roulette_wheel_selection(self.population, self.selection_size)
   
        # Perform SBX crossover with mutation
        while len(selected_individuals) < self.population_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child1, child2 = sbx_crossover(parent1, parent2, self.generation, self.mutation_rate, self.eta_c, self.guassianstd)
            
            # Add new children to the selected individuals
            selected_individuals.extend([child1, child2])

        # Update the population to the newly formed selected individuals
        self.population = selected_individuals[:self.population_size]  # Ensure we maintain the original size
        
        self.generation += 1
        
        # Return fitness values for tracking
        return avg_fitness, self.best_fitness, self.best_knobs1, self.best_knobs2, self.avg_knobs1, self.avg_knobs2

    def get_best_individual(self):
        return self.best_knobs1, self.best_knobs2

    def get_average_knobs(self):
        return self.avg_knobs1, self.avg_knobs2



def roulette_wheel_selection(population, num_individuals):
    selection = []
    # Calculate total fitness (using 1 / fitness to minimize the error being treated as fitness)
    total_fitness = sum(1 / ind.fitness for ind in population)  # Avoid None fitness
    
    for _ in range(num_individuals):
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual in population:
            current += 1 / individual.fitness 
            if current > pick:
                selection.append(individual)
                break
                
    return selection


def sbx_crossover(parent1, parent2, generation, mutation_rate, eta_c, guassstd):
    # Simulated Binary Crossover (SBX) with Gaussian mutation
    child1_knobs1 = np.copy(parent1.knobs1)
    child2_knobs1 = np.copy(parent2.knobs1)
    child1_knobs2 = np.copy(parent1.knobs2)
    child2_knobs2 = np.copy(parent2.knobs2)

    for i in range(len(child1_knobs1)):
        # Generate a random beta value for each gene
        beta = np.random.uniform(0, 1)
        beta_q = (2 * beta)**(1 / (eta_c + 1)) if beta <= 0.5 else (1 / (2 * (1 - beta)))**(1 / (eta_c + 1))

        # Generate child genes
        child1_knobs1[i] = 0.5 * ((1 + beta_q) * parent1.knobs1[i] + (1 - beta_q) * parent2.knobs1[i])
        child2_knobs1[i] = 0.5 * ((1 - beta_q) * parent1.knobs1[i] + (1 + beta_q) * parent2.knobs1[i])
        
        child1_knobs2[i] = 0.5 * ((1 + beta_q) * parent1.knobs2[i] + (1 - beta_q) * parent2.knobs2[i])
        child2_knobs2[i] = 0.5 * ((1 - beta_q) * parent1.knobs2[i] + (1 + beta_q) * parent2.knobs2[i])

    # Apply Gaussian mutation to both children
    child1_knobs1 = gaussian_mutation(child1_knobs1, mutation_rate, generation, guassstd)[0]
    child2_knobs1 = gaussian_mutation(child2_knobs1, mutation_rate, generation, guassstd)[0]
    child1_knobs2 = gaussian_mutation(child1_knobs2, mutation_rate, generation, guassstd)[0]
    child2_knobs2 = gaussian_mutation(child2_knobs2, mutation_rate, generation, guassstd)[0]

    # Return two new Individuals as children
    child1 = Individual(child1_knobs1, child1_knobs2)
    child2 = Individual(child2_knobs1, child2_knobs2)
    return child1, child2


def gaussian_mutation(knobs, mutation_rate, generation, sigma):
    # Applies Gaussian mutation with a decaying mutation rate.
    # Decay mutation rate as generations increase
    adjusted_mutation_rate = (mutation_rate/np.sqrt(generation+1))

    for i in range(len(knobs)):
        if np.random.rand() < adjusted_mutation_rate:
            # Add Gaussian noise to the knob value
            knobs[i] += np.random.normal(0, sigma)
            # Ensure the volume knob (first knob) stays above 1

    knobs = np.clip(knobs, 0, 10)
    knobs[0] = max(knobs[0], 0.1)

    return knobs, adjusted_mutation_rate


def tournament_selection(population, num_individuals, tournament_size):
    selection = []
    
    for _ in range(num_individuals):
        tournament = random.sample(population, tournament_size)
        best_individual = min(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else float('inf'))

        selection.append(best_individual)
    

        
    return selection

def elite_selection(population, num_individuals):
    # Sort the population based on fitness (lower fitness is better)
    sorted_population = sorted(
        population, 
        key=lambda ind: ind.fitness if ind.fitness is not None else float('inf')
    )
    
    # Select the top `num_individuals`
    selected = sorted_population[:num_individuals]
    
    return selected
