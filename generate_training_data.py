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

    # Ensure COG_vec column exists and is populated
    if 'COG_vec' not in parts_df.columns:
        parts_df['COG_vec'] = parts_df['Center of Mass (mm)'].apply(parse_vector_from_string)

    # 1. Total Mass
    total_mass = parts_df['Mass (kg)'].sum()
    if total_mass == 0:
        return {
            'Mass (kg)': 0,
            'Center of Mass (mm)': np.zeros(3),
            'Moment of Inertia (kg*mm^2)': np.zeros(3)
        }

    # 2. Combined Center of Mass (COG)
    cog_vectors = np.stack(parts_df['COG_vec'].values)
    masses = parts_df['Mass (kg)'].values[:, np.newaxis]
    weighted_cog_sum = (cog_vectors * masses).sum(axis=0)
    combined_cog = weighted_cog_sum / total_mass

    # 3. Combined Moment of Inertia (MOI)
    total_moi_tensor = np.zeros((3, 3))
    for index, row in parts_df.iterrows():
        mass = row['Mass (kg)']
        cog = row['COG_vec']
        moi_cg_vals = parse_vector_from_string(row[moi_col_name])

        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = moi_cg_vals

        moi_tensor_cg = np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]])

        dx, dy, dz = cog
        d_squared_matrix = np.array([
            [dy**2 + dz**2, -dx*dy, -dx*dz],
            [-dx*dy, dx**2 + dz**2, -dy*dz],
            [-dx*dz, -dy*dz, dx**2 + dy**2]
        ])

        moi_tensor_origin = moi_tensor_cg + mass * d_squared_matrix
        total_moi_tensor += moi_tensor_origin

    combined_moi = np.diag(total_moi_tensor)

    return {
        'Mass (kg)': total_mass,
        'Center of Mass (mm)': combined_cog,
        'Moment of Inertia (kg*mm^2)': combined_moi
    }

def main():
    """
    Main function to generate the two CSV files for ML training.
    """
    full_df = pd.read_csv('mass_properties.csv', skiprows=7)
    full_df['PartNum'] = full_df['Name'].str.extract(r'Part-(\d+)').astype(int)

    # --- Task 1: Create mass_properties_target.csv ---

    # Calculate Initial Properties (Parts 1-11)
    initial_df = full_df[full_df['PartNum'] <= 11].copy()
    initial_props = calculate_mass_properties(initial_df)

    # Calculate Target Properties (All Parts)
    target_props = calculate_mass_properties(full_df.copy())

    # Create the comparison DataFrame
    properties_data = {
        'Property': [
            'Mass (kg)', 'COG_x (mm)', 'COG_y (mm)', 'COG_z (mm)',
            'MOI_Ixx (kg*mm^2)', 'MOI_Iyy (kg*mm^2)', 'MOI_Izz (kg*mm^2)'
        ],
        'InitialValue': [
            initial_props['Mass (kg)'],
            initial_props['Center of Mass (mm)'][0],
            initial_props['Center of Mass (mm)'][1],
            initial_props['Center of Mass (mm)'][2],
            initial_props['Moment of Inertia (kg*mm^2)'][0],
            initial_props['Moment of Inertia (kg*mm^2)'][1],
            initial_props['Moment of Inertia (kg*mm^2)'][2]
        ],
        'TargetValue': [
            target_props['Mass (kg)'],
            target_props['Center of Mass (mm)'][0],
            target_props['Center of Mass (mm)'][1],
            target_props['Center of Mass (mm)'][2],
            target_props['Moment of Inertia (kg*mm^2)'][0],
            target_props['Moment of Inertia (kg*mm^2)'][1],
            target_props['Moment of Inertia (kg*mm^2)'][2]
        ]
    }
    target_df = pd.DataFrame(properties_data)
    target_df.to_csv('mass_properties_target.csv', index=False)
    print("Successfully created mass_properties_target.csv")

    # --- Task 2: Create random_ballast_coordinates.csv ---

    # Determine bounding box from all parts' original coordinates (not their COG)
    full_df['Coord_vec'] = full_df['Coordinates (mm)'].apply(parse_vector_from_string)
    all_coords = np.stack(full_df['Coord_vec'].values)
    min_bounds = all_coords.min(axis=0)
    max_bounds = all_coords.max(axis=0)

    # Get ballast part coordinates (Parts 12 onwards)
    ballast_df = full_df[full_df['PartNum'] > 11]
    ballast_coords = np.stack(ballast_df['Coord_vec'].values)
    num_ballast_parts = len(ballast_coords)

    # Generate new random coordinates
    num_random_coords_needed = 100 - num_ballast_parts
    random_coords = np.random.uniform(low=min_bounds, high=max_bounds, size=(num_random_coords_needed, 3))

    # Combine and shuffle
    combined_coords = np.vstack((ballast_coords, random_coords))
    np.random.shuffle(combined_coords)

    # Save to CSV
    coords_df = pd.DataFrame(combined_coords, columns=['x', 'y', 'z'])
    coords_df.to_csv('random_ballast_coordinates.csv', index=False)
    print("Successfully created random_ballast_coordinates.csv")


if __name__ == "__main__":
    main()
