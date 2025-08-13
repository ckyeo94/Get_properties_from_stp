# Mass Properties Extractor for STEP Files

This project provides a Python script to extract mass properties from a STEP file (`.stp` or `.step`). It calculates the mass, center of gravity, and moment of inertia for the entire assembly and for each individual component within the assembly.

## Requirements

*   Python 3
*   `cadquery` library

To install the necessary library, run:
```bash
pip install cadquery
```

## Usage

1.  Place your STEP file in the same directory as the script and name it `Assem1.STEP`. If your file has a different name, you will need to edit the `INPUT_FILE` variable in `process_step_cadquery.py`.
2.  Run the script from your terminal:
    ```bash
    python process_step_cadquery.py
    ```

The script will generate a file named `mass_properties.csv` in the same directory.

## Output

The `mass_properties.csv` file contains two sections:

### Assembly Mass Properties
This section provides the overall mass properties for the entire assembly, including:
*   **Mass (kg):** The total mass of the assembly.
*   **Center of Mass (mm):** The coordinates (X, Y, Z) of the assembly's center of mass.
*   **Moment of Inertia (kg*mm^2):** The moments of inertia (Ixx, Iyy, Izz) for the assembly, calculated about the assembly's center of mass.

### Individual Object Properties
This section lists the properties for each individual solid component found in the STEP file:
*   **Name:** A unique name assigned to the part (e.g., "Part-1").
*   **Coordinates (mm):** The coordinates (X, Y, Z) of the center of the component's bounding box.
*   **Mass (kg):** The mass of the individual component.
*   **Center of Mass (mm):** The coordinates (X, Y, Z) of the component's own center of mass.
*   **Moment of Inertia (kg*mm^2):** The moments of inertia (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) for the component, calculated about its own center of mass.

**Note:** The script assumes a constant density for all parts, corresponding to steel (7.85e-6 kg/mm^3). You can change this value by modifying the `STEEL_DENSITY` variable in the script.
