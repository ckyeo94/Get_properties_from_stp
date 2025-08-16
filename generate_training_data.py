import pandas as pd
import numpy as np
import re

def parse_vector_from_string(s):
    """Parses a string like '(...)' into a numpy array."""
    s = str(s).strip().replace('"', '')
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    return np.array([float(x.strip()) for x in s.split(',')])

def calculate_mass_properties(parts_df):
    """
    Calculates the combined mass properties for a given DataFrame of parts.
    Returns a dictionary with the calculated properties.
    """
    if parts_df.empty:
        return {
            'Mass (kg)': 0,
            'Center of Mass (mm)': np.zeros(3),
            'Moment of Inertia (kg*mm^2)': np.zeros(3)
        }

    moi_col_name = 'Moment of Inertia (kg*mm^2, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)'

    if 'COG_vec' not in parts_df.columns:
        parts_df['COG_vec'] = parts_df['Center of Mass (mm)'].apply(parse_vector_from_string)

    total_mass = parts_df['Mass (kg)'].sum()
    if total_mass == 0:
        return {'Mass (kg)': 0, 'Center of Mass (mm)': np.zeros(3), 'Moment of Inertia (kg*mm^2)': np.zeros(3)}

    cog_vectors = np.stack(parts_df['COG_vec'].values)
    masses = parts_df['Mass (kg)'].values[:, np.newaxis]
    weighted_cog_sum = (cog_vectors * masses).sum(axis=0)
    combined_cog = weighted_cog_sum / total_mass

    total_moi_tensor = np.zeros((3, 3))
    for index, row in parts_df.iterrows():
        # The MOI values from the CSV are already relative to the assembly origin.
        # We just need to sum the tensors of the individual parts.
        moi_vals = parse_vector_from_string(row[moi_col_name])
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = moi_vals

        moi_tensor_part = np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]])
        total_moi_tensor += moi_tensor_part

    combined_moi = np.diag(total_moi_tensor)

    return {
        'Mass (kg)': total_mass,
        'Center of Mass (mm)': combined_cog,
        'Moment of Inertia (kg*mm^2)': combined_moi
    }

def main(mass_scale_factor=1.0):
    """
    Main function to generate the two CSV files for ML training.
    mass_scale_factor: Factor to scale the mass and MOI of parts 1-11.
    """
    full_df = pd.read_csv('mass_properties.csv', skiprows=7)
    full_df['PartNum'] = full_df['Name'].str.extract(r'Part-(\d+)').astype(int)

    # --- Task 1: Create mass_properties_target.csv ---

    initial_df = full_df[full_df['PartNum'] <= 11].copy()

    if mass_scale_factor != 1.0:
        initial_df['Mass (kg)'] *= mass_scale_factor

        moi_col = 'Moment of Inertia (kg*mm^2, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)'
        def scale_moi_string(s):
            vec = parse_vector_from_string(s)
            scaled_vec = vec * mass_scale_factor
            return f"({', '.join(map(str, scaled_vec))})"

        initial_df[moi_col] = initial_df[moi_col].apply(scale_moi_string)

    initial_props = calculate_mass_properties(initial_df)
    target_props = calculate_mass_properties(full_df.copy())

    properties_data = {
        'Property': ['Mass (kg)', 'COG_x (mm)', 'COG_y (mm)', 'COG_z (mm)', 'MOI_Ixx (kg*mm^2)', 'MOI_Iyy (kg*mm^2)', 'MOI_Izz (kg*mm^2)'],
        'InitialValue': [
            initial_props['Mass (kg)'], initial_props['Center of Mass (mm)'][0], initial_props['Center of Mass (mm)'][1], initial_props['Center of Mass (mm)'][2],
            initial_props['Moment of Inertia (kg*mm^2)'][0], initial_props['Moment of Inertia (kg*mm^2)'][1], initial_props['Moment of Inertia (kg*mm^2)'][2]
        ],
        'TargetValue': [
            target_props['Mass (kg)'], target_props['Center of Mass (mm)'][0], target_props['Center of Mass (mm)'][1], target_props['Center of Mass (mm)'][2],
            target_props['Moment of Inertia (kg*mm^2)'][0], target_props['Moment of Inertia (kg*mm^2)'][1], target_props['Moment of Inertia (kg*mm^2)'][2]
        ]
    }
    target_df = pd.DataFrame(properties_data)
    target_df.to_csv('mass_properties_target.csv', index=False)
    print(f"Successfully created mass_properties_target.csv with a mass scale factor of {mass_scale_factor}")

    # --- Task 2: Create random_ballast_coordinates.csv ---

    if 'Coord_vec' not in full_df.columns:
        full_df['Coord_vec'] = full_df['Coordinates (mm)'].apply(parse_vector_from_string)

    all_coords = np.stack(full_df['Coord_vec'].values)
    min_bounds = all_coords.min(axis=0)
    max_bounds = all_coords.max(axis=0)

    ballast_df = full_df[full_df['PartNum'] > 11]
    ballast_coords = np.stack(ballast_df['Coord_vec'].values)
    num_ballast_parts = len(ballast_coords)

    num_random_coords_needed = 100 - num_ballast_parts
    random_coords = np.random.uniform(low=min_bounds, high=max_bounds, size=(num_random_coords_needed, 3))

    combined_coords = np.vstack((ballast_coords, random_coords))
    # np.random.shuffle(combined_coords) # Removed shuffling to keep original ballasts first

    coords_df = pd.DataFrame(combined_coords, columns=['x', 'y', 'z'])
    coords_df.to_csv('random_ballast_coordinates.csv', index=False)
    print("Successfully created random_ballast_coordinates.csv")


if __name__ == "__main__":
    # To change the density of Parts 1-11, modify the mass_scale_factor.
    # For example, a factor of 1.1 increases mass by 10%.
    # A value of 1.0 means no change from the original.
    mass_scale_factor_to_use = 1.0
    main(mass_scale_factor=mass_scale_factor_to_use)
