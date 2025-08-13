import csv
import cadquery as cq

def get_mass_properties(solid, density):
    """
    Calculates the mass properties of a single solid using the correct
    static method calls on the Shape class.
    """
    volume = solid.Volume()
    mass = volume * density

    center_of_mass = cq.Shape.centerOfMass(solid)
    moi = cq.Shape.matrixOfInertia(solid)

    return {
        "mass": mass,
        "center_of_mass": (center_of_mass.x, center_of_mass.y, center_of_mass.z),
        "moment_of_inertia": (
            moi[0][0], moi[1][1], moi[2][2], # Ixx, Iyy, Izz
            moi[0][1], moi[0][2], moi[1][2]  # Ixy, Ixz, Iyz
        )
    }

def process_step_file(file_path, output_csv_path, density):
    """
    Processes a STEP file to extract mass properties and saves them to a CSV file.
    """
    # Load the STEP file
    imported_obj = cq.importers.importStep(file_path)

    solids = []
    if isinstance(imported_obj, cq.Workplane):
        solids.extend(imported_obj.solids().vals())
    elif isinstance(imported_obj, cq.Assembly):
        for obj in imported_obj.objects.values():
            if obj.shape:
                if isinstance(obj.shape, cq.Compound):
                    solids.extend(obj.shape.Solids())
                elif isinstance(obj.shape, cq.Solid):
                    solids.append(obj.shape)
    elif isinstance(imported_obj, cq.Compound):
        solids.extend(imported_obj.Solids())
    elif isinstance(imported_obj, cq.Solid):
        solids.append(imported_obj)


    components_data = []
    for i, solid in enumerate(solids):
        if isinstance(solid, cq.Solid) and solid.Volume() > 1e-9:
            props = get_mass_properties(solid, density)

            bb = solid.BoundingBox()
            center = bb.center

            components_data.append({
                "name": f"Part-{i+1}",
                "coordinates": (center.x, center.y, center.z),
                "mass": props["mass"],
                "center_of_mass": props["center_of_mass"],
                "moment_of_inertia": props["moment_of_inertia"]
            })

    # Assembly properties
    total_mass = sum(c["mass"] for c in components_data)

    if total_mass > 0:
        cg_x = sum(c["mass"] * c["center_of_mass"][0] for c in components_data) / total_mass
        cg_y = sum(c["mass"] * c["center_of_mass"][1] for c in components_data) / total_mass
        cg_z = sum(c["mass"] * c["center_of_mass"][2] for c in components_data) / total_mass
    else:
        cg_x, cg_y, cg_z = 0, 0, 0

    assembly_cg = (cg_x, cg_y, cg_z)

    # Correctly calculate assembly moment of inertia using the parallel axis theorem
    moi_xx = 0
    moi_yy = 0
    moi_zz = 0
    for comp in components_data:
        mass = comp["mass"]
        comp_cg = comp["center_of_mass"]
        comp_moi = comp["moment_of_inertia"]

        dx = comp_cg[0] - assembly_cg[0]
        dy = comp_cg[1] - assembly_cg[1]
        dz = comp_cg[2] - assembly_cg[2]

        moi_xx += comp_moi[0] + mass * (dy**2 + dz**2)
        moi_yy += comp_moi[1] + mass * (dx**2 + dz**2)
        moi_zz += comp_moi[2] + mass * (dx**2 + dy**2)


    # Write to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["Assembly Mass Properties"])
        writer.writerow(["Property", "Value"])
        writer.writerow(["Mass (kg)", f"{total_mass:.6f}"])
        writer.writerow(["Center of Mass (mm)", f"({cg_x:.4f}, {cg_y:.4f}, {cg_z:.4f})"])
        writer.writerow(["Moment of Inertia (kg*mm^2)", f"(Ixx={moi_xx:.4f}, Iyy={moi_yy:.4f}, Izz={moi_zz:.4f})"])
        writer.writerow([])

        writer.writerow(["Individual Object Properties"])
        writer.writerow(["Name", "Coordinates (mm)", "Mass (kg)", "Center of Mass (mm)", "Moment of Inertia (kg*mm^2, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)"])
        for comp in components_data:
            writer.writerow([
                comp["name"],
                f"({comp['coordinates'][0]:.4f}, {comp['coordinates'][1]:.4f}, {comp['coordinates'][2]:.4f})",
                f"{comp['mass']:.6f}",
                f"({comp['center_of_mass'][0]:.4f}, {comp['center_of_mass'][1]:.4f}, {comp['center_of_mass'][2]:.4f})",
                f"({comp['moment_of_inertia'][0]:.4f}, {comp['moment_of_inertia'][1]:.4f}, {comp['moment_of_inertia'][2]:.4f}, {comp['moment_of_inertia'][3]:.4f}, {comp['moment_of_inertia'][4]:.4f}, {comp['moment_of_inertia'][5]:.4f})"
            ])

if __name__ == "__main__":
    STEEL_DENSITY = 7.85e-6  # kg/mm^3
    INPUT_FILE = "Assem1.STEP"
    OUTPUT_FILE = "mass_properties.csv"

    process_step_file(INPUT_FILE, OUTPUT_FILE, STEEL_DENSITY)
    print(f"Mass properties extracted and saved to {OUTPUT_FILE}")
