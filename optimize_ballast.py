import pandas as pd
import numpy as np

def load_data():
    """Loads all the necessary data from the CSV files."""
    target_properties = pd.read_csv("mass_properties_target.csv", index_col=0)
    ballast_coords = pd.read_csv("random_ballast_coordinates.csv")

    initial_mass = target_properties.loc['Mass (kg)', 'InitialValue']
    initial_cg = np.array([
        target_properties.loc['COG_x (mm)', 'InitialValue'],
        target_properties.loc['COG_y (mm)', 'InitialValue'],
        target_properties.loc['COG_z (mm)', 'InitialValue']
    ])
    initial_moi = np.array([
        target_properties.loc['MOI_Ixx (kg*mm^2)', 'InitialValue'],
        target_properties.loc['MOI_Iyy (kg*mm^2)', 'InitialValue'],
        target_properties.loc['MOI_Izz (kg*mm^2)', 'InitialValue']
    ])

    target_mass = target_properties.loc['Mass (kg)', 'TargetValue']
    target_cg = np.array([
        target_properties.loc['COG_x (mm)', 'TargetValue'],
        target_properties.loc['COG_y (mm)', 'TargetValue'],
        target_properties.loc['COG_z (mm)', 'TargetValue']
    ])
    target_moi = np.array([
        target_properties.loc['MOI_Ixx (kg*mm^2)', 'TargetValue'],
        target_properties.loc['MOI_Iyy (kg*mm^2)', 'TargetValue'],
        target_properties.loc['MOI_Izz (kg*mm^2)', 'TargetValue']
    ])

    return (initial_mass, initial_cg, initial_moi), (target_mass, target_cg, target_moi), ballast_coords

def calculate_combined_properties(initial_mass, initial_cg, initial_moi, ballast_coords, ballast_selection, ballast_mass=1.0):
    """
    Calculates the combined mass, CG, and MOI of the assembly with the added ballasts.
    """
    selected_ballasts = ballast_coords[ballast_selection]
    num_ballasts = len(selected_ballasts)
    total_ballast_mass = num_ballasts * ballast_mass

    new_mass = initial_mass + total_ballast_mass

    if new_mass > 0:
        initial_weighted_cg = initial_cg * initial_mass
        ballast_weighted_cg = selected_ballasts.sum(axis=0).to_numpy() * ballast_mass
        new_cg = (initial_weighted_cg + ballast_weighted_cg) / new_mass
    else:
        new_cg = np.zeros(3)

    new_moi = initial_moi.copy()
    for _, ballast_row in selected_ballasts.iterrows():
        ballast_pos = ballast_row.to_numpy()
        dx, dy, dz = ballast_pos
        new_moi[0] += ballast_mass * (dy**2 + dz**2) # Ixx
        new_moi[1] += ballast_mass * (dx**2 + dz**2) # Iyy
        new_moi[2] += ballast_mass * (dx**2 + dy**2) # Izz

    return new_mass, new_cg, new_moi

def calculate_fitness(solution, initial_mass, initial_cg, initial_moi, target_cg, target_moi, ballast_coords, ballast_mass=1.0):
    """
    Calculates the fitness of a solution based on the new component-wise constraints.
    A lower fitness score is better. Fitness is the number of constraint violations.
    """
    _, new_cg, new_moi = calculate_combined_properties(initial_mass, initial_cg, initial_moi, ballast_coords, solution, ballast_mass)

    violations = 0
    # Check CG component errors
    if abs(new_cg[0] - target_cg[0]) / abs(target_cg[0]) > 0.01: violations += 1
    if abs(new_cg[1] - target_cg[1]) / abs(target_cg[1]) > 0.01: violations += 1
    if abs(new_cg[2] - target_cg[2]) / abs(target_cg[2]) > 0.01: violations += 1

    # Check MOI component errors
    if abs(new_moi[0] - target_moi[0]) / target_moi[0] > 0.03: violations += 1
    if abs(new_moi[1] - target_moi[1]) / target_moi[1] > 0.03: violations += 1
    if abs(new_moi[2] - target_moi[2]) / target_moi[2] > 0.03: violations += 1

    return violations

def print_results_breakdown(final_mass, final_cg, final_moi, target_mass, target_cg, target_moi):
    """Prints a detailed breakdown of the final results."""

    mass_error = abs(final_mass - target_mass) / target_mass * 100 if target_mass != 0 else 0
    cg_x_error = abs(final_cg[0] - target_cg[0]) / abs(target_cg[0]) * 100 if abs(target_cg[0]) > 1e-9 else 0
    cg_y_error = abs(final_cg[1] - target_cg[1]) / abs(target_cg[1]) * 100 if abs(target_cg[1]) > 1e-9 else 0
    cg_z_error = abs(final_cg[2] - target_cg[2]) / abs(target_cg[2]) * 100 if abs(target_cg[2]) > 1e-9 else 0
    moi_ixx_error = abs(final_moi[0] - target_moi[0]) / target_moi[0] * 100 if target_moi[0] != 0 else 0
    moi_iyy_error = abs(final_moi[1] - target_moi[1]) / target_moi[1] * 100 if target_moi[1] != 0 else 0
    moi_izz_error = abs(final_moi[2] - target_moi[2]) / target_moi[2] * 100 if target_moi[2] != 0 else 0

    print("\n--- Detailed Results Breakdown ---")
    print(f"{'Property':<10} | {'Target':<25} | {'Achieved':<25} | {'Error (%)':<10}")
    print("-" * 80)
    print(f"{'Mass (kg)':<10} | {target_mass:<25.6f} | {final_mass:<25.6f} | {mass_error:<10.4f}")
    print(f"{'COG_x':<10} | {target_cg[0]:<25.6f} | {final_cg[0]:<25.6f} | {cg_x_error:<10.4f}")
    print(f"{'COG_y':<10} | {target_cg[1]:<25.6f} | {final_cg[1]:<25.6f} | {cg_y_error:<10.4f}")
    print(f"{'COG_z':<10} | {target_cg[2]:<25.6f} | {final_cg[2]:<25.6f} | {cg_z_error:<10.4f}")
    print(f"{'MOI_Ixx':<10} | {target_moi[0]:<25.2f} | {final_moi[0]:<25.2f} | {moi_ixx_error:<10.4f}")
    print(f"{'MOI_Iyy':<10} | {target_moi[1]:<25.2f} | {final_moi[1]:<25.2f} | {moi_iyy_error:<10.4f}")
    print(f"{'MOI_Izz':<10} | {target_moi[2]:<25.2f} | {final_moi[2]:<25.2f} | {moi_izz_error:<10.4f}")

    if max(cg_x_error, cg_y_error, cg_z_error) < 1 and max(moi_ixx_error, moi_iyy_error, moi_izz_error) < 3:
        print("\n--- STATUS: Solution meets the requirements! ---")
    else:
        print("\n--- STATUS: Solution does not meet all requirements. ---")

if __name__ == '__main__':
    (initial_mass, initial_cg, initial_moi), (target_mass, target_cg, target_moi), ballast_coords = load_data()

    ballast_mass = 5.0
    num_ballasts_to_add = int(np.floor((target_mass - initial_mass) / ballast_mass))

    print("--- Optimization Setup ---")
    print(f"Ballast Weight: {ballast_mass} kg")
    print(f"Required number of ballasts to add: {num_ballasts_to_add}")

    num_locations = len(ballast_coords)
    if num_ballasts_to_add > num_locations:
        print(f"\nError: Required number of ballasts ({num_ballasts_to_add}) is greater than the number of available locations ({num_locations}).")
        exit()

    # --- Strategy B: Test the original 24 ballast locations ---
    # There are 35 parts in mass_properties.csv, parts 1-11 are the initial assembly, so 24 are ballasts.
    num_original_ballasts = 24
    print(f"\n\n--- Strategy B: Testing the Original {num_original_ballasts} Ballast Locations ---")

    if num_ballasts_to_add == num_original_ballasts:
        original_selection = np.array([True] * num_original_ballasts + [False] * (num_locations - num_original_ballasts))
        print(f"Calculating properties for the configuration using all {num_original_ballasts} original ballast locations...")
        final_mass_orig, final_cg_orig, final_moi_orig = calculate_combined_properties(initial_mass, initial_cg, initial_moi, ballast_coords, original_selection, ballast_mass)
        print_results_breakdown(final_mass_orig, final_cg_orig, final_moi_orig, target_mass, target_cg, target_moi)
    else:
        print("The required number of ballasts to add does not match the number of original ballasts. Skipping this test.")


    # --- Strategy A: Search all 100 locations using GA ---
    print(f"\n\n--- Strategy A: Searching All {num_locations} Locations with Genetic Algorithm ---")

    # GA Parameters
    population_size = 200
    num_generations = 500
    mutation_rate = 0.1

    # Initialize population
    population = []
    for _ in range(population_size):
        individual = np.zeros(num_locations, dtype=bool)
        true_indices = np.random.choice(num_locations, num_ballasts_to_add, replace=False)
        individual[true_indices] = True
        population.append(individual)

    for gen in range(num_generations):
        fitness_scores = [calculate_fitness(ind, initial_mass, initial_cg, initial_moi, target_cg, target_moi, ballast_coords, ballast_mass) for ind in population]
        best_idx = np.argmin(fitness_scores)

        if (gen + 1) % 50 == 0:
            print(f"Generation {gen+1}, Best Fitness (violations): {fitness_scores[best_idx]}")

        if fitness_scores[best_idx] == 0:
            print(f"Feasible solution found at generation {gen+1}!")
            break

        # Elitism
        new_population = [population[best_idx]]

        # Tournament Selection
        tournament_size = 5
        for _ in range(population_size - 1):
            tournament_indices = np.random.randint(0, population_size, size=tournament_size)
            tournament_fitnesses = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitnesses)]
            new_population.append(population[winner_index])

        # Crossover and Mutation
        offspring_population = []
        for i in range(0, population_size, 2):
            parent1 = new_population[i]
            parent2 = new_population[i+1]

            crossover_point = np.random.randint(1, num_locations)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            num_true = np.sum(child1)
            if num_true > num_ballasts_to_add:
                true_indices = np.where(child1)[0]
                indices_to_flip = np.random.choice(true_indices, num_true - num_ballasts_to_add, replace=False)
                child1[indices_to_flip] = False
            elif num_true < num_ballasts_to_add:
                false_indices = np.where(~child1)[0]
                indices_to_flip = np.random.choice(false_indices, num_ballasts_to_add - num_true, replace=False)
                child1[indices_to_flip] = True

            if np.random.rand() < mutation_rate:
                true_indices = np.where(child1)[0]
                false_indices = np.where(~child1)[0]
                if len(true_indices) > 0 and len(false_indices) > 0:
                    true_to_swap = np.random.choice(true_indices)
                    false_to_swap = np.random.choice(false_indices)
                    child1[true_to_swap], child1[false_to_swap] = False, True

            offspring_population.append(child1)
            offspring_population.append(parent2)

        population = offspring_population

    # Final evaluation for GA
    fitness_scores_ga = [calculate_fitness(ind, initial_mass, initial_cg, initial_moi, target_cg, target_moi, ballast_coords, ballast_mass) for ind in population]
    best_solution_ga = population[np.argmin(fitness_scores_ga)]

    print("\nBest solution found by GA:")
    final_mass_ga, final_cg_ga, final_moi_ga = calculate_combined_properties(initial_mass, initial_cg, initial_moi, ballast_coords, best_solution_ga, ballast_mass)
    print_results_breakdown(final_mass_ga, final_cg_ga, final_moi_ga, target_mass, target_cg, target_moi)
