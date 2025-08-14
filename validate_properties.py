import pandas as pd
import numpy as np
import re

def parse_vector_from_string(s):
    """Parses a string like '(...)' or 'Ixx=...' into a numpy array."""
    s = s.strip().replace('"', '')
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]

    if 'Ixx=' in s:
        # Format is "Ixx=..., Iyy=..., Izz=..."
        parts = s.split(',')
        vals = [float(p.split('=')[1]) for p in parts]
        return np.array(vals)
    else:
        # Format is "x, y, z"
        return np.array([float(x.strip()) for x in s.split(',')])

def get_assembly_properties(filepath):
    """Reads the assembly properties from the top of the CSV file."""
    props = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Corrected line indices
        props['Mass'] = float(lines[2].split(',')[1])
        props['Center of Mass'] = parse_vector_from_string(lines[3].split(',', 1)[1])
        props['Moment of Inertia'] = parse_vector_from_string(lines[4].split(',', 1)[1])
    return props

def validate_mass_properties(filepath):
    """
    Validates the mass properties of an assembly by summing the properties of its individual parts.
    """
    assembly_props = get_assembly_properties(filepath)

    # Read individual parts properties.
    parts_df = pd.read_csv(filepath, skiprows=7)
    moi_col_name = 'Moment of Inertia (kg*mm^2, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)'

    # --- Calculations ---
    calculated_mass = parts_df['Mass (kg)'].sum()

    parts_df['COG_vec'] = parts_df['Center of Mass (mm)'].apply(parse_vector_from_string)
    cog_vectors = np.stack(parts_df['COG_vec'].values)
    masses = parts_df['Mass (kg)'].values[:, np.newaxis]
    weighted_cog_sum = (cog_vectors * masses).sum(axis=0)
    calculated_cog = weighted_cog_sum / calculated_mass

    total_moi_tensor = np.zeros((3, 3))
    for index, row in parts_df.iterrows():
        mass = row['Mass (kg)']
        cog = row['COG_vec']
        moi_cg_vals = parse_vector_from_string(row[moi_col_name])

        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = moi_cg_vals

        moi_tensor_cg = np.array([
            [Ixx, -Ixy, -Ixz],
            [-Ixy, Iyy, -Iyz],
            [-Ixz, -Iyz, Izz]
        ])

        dx, dy, dz = cog
        d_squared_matrix = np.array([
            [dy**2 + dz**2, -dx*dy, -dx*dz],
            [-dx*dy, dx**2 + dz**2, -dy*dz],
            [-dx*dz, -dy*dz, dx**2 + dy**2]
        ])

        moi_tensor_origin = moi_tensor_cg + mass * d_squared_matrix
        total_moi_tensor += moi_tensor_origin

    calculated_moi = np.diag(total_moi_tensor)

    # --- Validation Report ---
    print("--- Mass Properties Validation Report ---")
    print(f"\nComparing assembly properties from '{filepath}' with calculated values.")

    print("\n1. Mass (kg):")
    print(f"  - Assembly Value: {assembly_props['Mass']:.6f}")
    print(f"  - Calculated Sum: {calculated_mass:.6f}")
    print(f"  - Matches: {np.isclose(assembly_props['Mass'], calculated_mass)}")

    print("\n2. Center of Mass (x, y, z) (mm):")
    print(f"  - Assembly Value: {np.array2string(assembly_props['Center of Mass'], precision=4, floatmode='fixed')}")
    print(f"  - Calculated Sum: {np.array2string(calculated_cog, precision=4, floatmode='fixed')}")
    print(f"  - Matches: {np.allclose(assembly_props['Center of Mass'], calculated_cog)}")

    print("\n3. Moment of Inertia (Ixx, Iyy, Izz) (kg*mm^2):")
    print(f"  - Assembly Value: {np.array2string(assembly_props['Moment of Inertia'], precision=4, floatmode='fixed', suppress_small=True)}")
    print(f"  - Calculated Sum: {np.array2string(calculated_moi, precision=4, floatmode='fixed', suppress_small=True)}")
    print(f"  - Matches: {np.allclose(assembly_props['Moment of Inertia'], calculated_moi, atol=1e-3, rtol=1e-3)}")

if __name__ == "__main__":
    validate_mass_properties('mass_properties.csv')
