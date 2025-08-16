import pandas as pd
import numpy as np

def load_data():
    """Loads all the necessary data from the CSV files."""
    target_properties = pd.read_csv("mass_properties_target.csv", index_col='Property')
    ballast_coords = pd.read_csv("random_ballast_coordinates.csv")

    initial_mass = target_properties.loc['Mass (kg)', 'InitialValue']
    initial_cg = target_properties.loc[['COG_x (mm)', 'COG_y (mm)', 'COG_z (mm)'], 'InitialValue'].to_numpy()

    Ixx_i = target_properties.loc['MOI_Ixx (kg*mm^2)', 'InitialValue']
    Iyy_i = target_properties.loc['MOI_Iyy (kg*mm^2)', 'InitialValue']
    Izz_i = target_properties.loc['MOI_Izz (kg*mm^2)', 'InitialValue']
    Ixy_i = target_properties.loc['MOI_Ixy (kg*mm^2)', 'InitialValue']
    Ixz_i = target_properties.loc['MOI_Ixz (kg*mm^2)', 'InitialValue']
    Iyz_i = target_properties.loc['MOI_Iyz (kg*mm^2)', 'InitialValue']
    initial_moi = np.array([[Ixx_i, -Ixy_i, -Ixz_i], [-Ixy_i, Iyy_i, -Iyz_i], [-Ixz_i, -Iyz_i, Izz_i]])

    target_mass = target_properties.loc['Mass (kg)', 'TargetValue']
    target_cg = target_properties.loc[['COG_x (mm)', 'COG_y (mm)', 'COG_z (mm)'], 'TargetValue'].to_numpy()

    Ixx_t = target_properties.loc['MOI_Ixx (kg*mm^2)', 'TargetValue']
    Iyy_t = target_properties.loc['MOI_Iyy (kg*mm^2)', 'TargetValue']
    Izz_t = target_properties.loc['MOI_Izz (kg*mm^2)', 'TargetValue']
    Ixy_t = target_properties.loc['MOI_Ixy (kg*mm^2)', 'TargetValue']
    Ixz_t = target_properties.loc['MOI_Ixz (kg*mm^2)', 'TargetValue']
    Iyz_t = target_properties.loc['MOI_Iyz (kg*mm^2)', 'TargetValue']
    target_moi = np.array([[Ixx_t, -Ixy_t, -Ixz_t], [-Ixy_t, Iyy_t, -Iyz_t], [-Ixz_t, -Iyz_t, Izz_t]])

    return (initial_mass, initial_cg, initial_moi), (target_mass, target_cg, target_moi), ballast_coords

def _get_point_mass_inertia(mass, d):
    """Helper to get inertia tensor of a point mass relative to a parallel frame."""
    dx, dy, dz = d
    Ixx = mass * (dy**2 + dz**2)
    Iyy = mass * (dx**2 + dz**2)
    Izz = mass * (dx**2 + dy**2)
    Ixy = -mass * dx * dy
    Ixz = -mass * dx * dz
    Iyz = -mass * dy * dz
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

def calculate_combined_properties(initial_mass, initial_cg, initial_moi, ballast_coords, ballast_selection, ballast_mass=5.0):
    """
    Calculates the combined mass, CG, and MOI of the assembly with the added ballasts.
    This version correctly applies the Parallel Axis Theorem for full 3x3 tensors.
    """
    selected_ballasts_df = ballast_coords[ballast_selection]
    num_ballasts = len(selected_ballasts_df)
    total_ballast_mass = num_ballasts * ballast_mass

    # 1. Calculate new total mass and CG
    new_mass = initial_mass + total_ballast_mass
    if new_mass <= 0:
        return 0, np.zeros(3), np.zeros((3, 3))

    initial_weighted_cg = initial_cg * initial_mass
    ballast_positions = selected_ballasts_df.to_numpy()
    ballast_weighted_cg_sum = ballast_positions.sum(axis=0) * ballast_mass
    new_cg = (initial_weighted_cg + ballast_weighted_cg_sum) / new_mass

    # 2. Transfer initial assembly's MOI to the new combined CG
    d_initial = initial_cg - new_cg
    I_initial_displaced = initial_moi + _get_point_mass_inertia(initial_mass, d_initial)

    # 3. Sum the MOI of all ballasts (as point masses) relative to the new combined CG
    total_moi = I_initial_displaced
    for i in range(num_ballasts):
        ballast_pos = ballast_positions[i]
        d_ballast = ballast_pos - new_cg
        total_moi += _get_point_mass_inertia(ballast_mass, d_ballast)

    return new_mass, new_cg, total_moi

def calculate_fitness(solution, initial_mass, initial_cg, initial_moi, target_cg, target_moi, ballast_coords, ballast_mass=5.0):
    """
    Calculates the fitness of a solution based on the new component-wise constraints.
    A lower fitness score is better. Fitness is the number of constraint violations.
    """
    _, new_cg, new_moi = calculate_combined_properties(initial_mass, initial_cg, initial_moi, ballast_coords, solution, ballast_mass)

    violations = 0
    # Check CG component errors (1% tolerance)
    cg_error = np.abs((new_cg - target_cg) / target_cg)
    violations += np.sum(cg_error > 0.01)

    # Check MOI component errors (3% tolerance)
    # Compare the 6 unique terms of the inertia tensor
    moi_error_diag = np.abs((np.diag(new_moi) - np.diag(target_moi)) / np.diag(target_moi))
    violations += np.sum(moi_error_diag > 0.03)

    # For off-diagonal, protect against division by zero if target is near zero
    target_ixy, target_ixz, target_iyz = -target_moi[0, 1], -target_moi[0, 2], -target_moi[1, 2]
    new_ixy, new_ixz, new_iyz = -new_moi[0, 1], -new_moi[0, 2], -new_moi[1, 2]

    if abs(target_ixy) > 1e-9 and abs((new_ixy - target_ixy) / target_ixy) > 0.03: violations += 1
    if abs(target_ixz) > 1e-9 and abs((new_ixz - target_ixz) / target_ixz) > 0.03: violations += 1
    if abs(target_iyz) > 1e-9 and abs((new_iyz - target_iyz) / target_iyz) > 0.03: violations += 1

    return violations

def print_results_breakdown(final_mass, final_cg, final_moi, target_mass, target_cg, target_moi):
    """Prints a detailed breakdown of the final results."""

    def safe_error(achieved, target):
        if abs(target) < 1e-9:
            return 0 if abs(achieved) < 1e-9 else float('inf')
        return abs((achieved - target) / target) * 100

    mass_error = safe_error(final_mass, target_mass)
    cg_x_error = safe_error(final_cg[0], target_cg[0])
    cg_y_error = safe_error(final_cg[1], target_cg[1])
    cg_z_error = safe_error(final_cg[2], target_cg[2])

    # Extract 6 components from final and target tensors
    f_ixx, f_iyy, f_izz = final_moi[0,0], final_moi[1,1], final_moi[2,2]
    f_ixy, f_ixz, f_iyz = -final_moi[0,1], -final_moi[0,2], -final_moi[1,2]
    t_ixx, t_iyy, t_izz = target_moi[0,0], target_moi[1,1], target_moi[2,2]
    t_ixy, t_ixz, t_iyz = -target_moi[0,1], -target_moi[0,2], -target_moi[1,2]

    moi_ixx_error = safe_error(f_ixx, t_ixx)
    moi_iyy_error = safe_error(f_iyy, t_iyy)
    moi_izz_error = safe_error(f_izz, t_izz)
    moi_ixy_error = safe_error(f_ixy, t_ixy)
    moi_ixz_error = safe_error(f_ixz, t_ixz)
    moi_iyz_error = safe_error(f_iyz, t_iyz)

    print("\n--- Detailed Results Breakdown ---")
    print(f"{'Property':<10} | {'Target':<25} | {'Achieved':<25} | {'Error (%)':<10}")
    print("-" * 80)
    print(f"{'Mass (kg)':<10} | {target_mass:<25.6f} | {final_mass:<25.6f} | {mass_error:<10.4f}")
    print(f"{'COG_x':<10} | {target_cg[0]:<25.4f} | {final_cg[0]:<25.4f} | {cg_x_error:<10.4f}")
    print(f"{'COG_y':<10} | {target_cg[1]:<25.4f} | {final_cg[1]:<25.4f} | {cg_y_error:<10.4f}")
    print(f"{'COG_z':<10} | {target_cg[2]:<25.4f} | {final_cg[2]:<25.4f} | {cg_z_error:<10.4f}")
    print(f"{'MOI_Ixx':<10} | {t_ixx:<25.2f} | {f_ixx:<25.2f} | {moi_ixx_error:<10.4f}")
    print(f"{'MOI_Iyy':<10} | {t_iyy:<25.2f} | {f_iyy:<25.2f} | {moi_iyy_error:<10.4f}")
    print(f"{'MOI_Izz':<10} | {t_izz:<25.2f} | {f_izz:<25.2f} | {moi_izz_error:<10.4f}")
    print(f"{'MOI_Ixy':<10} | {t_ixy:<25.2f} | {f_ixy:<25.2f} | {moi_ixy_error:<10.4f}")
    print(f"{'MOI_Ixz':<10} | {t_ixz:<25.2f} | {f_ixz:<25.2f} | {moi_ixz_error:<10.4f}")
    print(f"{'MOI_Iyz':<10} | {t_iyz:<25.2f} | {f_iyz:<25.2f} | {moi_iyz_error:<10.4f}")

    cg_errors = [cg_x_error, cg_y_error, cg_z_error]
    moi_errors = [moi_ixx_error, moi_iyy_error, moi_izz_error, moi_ixy_error, moi_ixz_error, moi_iyz_error]

    if max(cg_errors) < 1 and max(moi_errors) < 3:
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
