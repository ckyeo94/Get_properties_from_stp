
import json
import sys
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.TopLoc import TopLoc_Location

def get_shape_properties(shape, density=1.0):
    """
    Calculate mass properties for a single TopoDS_Shape.
    """
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)

    mass = props.Mass() * density
    cg = props.CentreOfMass()
    moi = props.MatrixOfInertia()

    return {
        "mass": mass,
        "cg": (cg.X(), cg.Y(), cg.Z()),
        "moi": [
            [moi.Value(1, 1), moi.Value(1, 2), moi.Value(1, 3)],
            [moi.Value(2, 1), moi.Value(2, 2), moi.Value(2, 3)],
            [moi.Value(3, 1), moi.Value(3, 2), moi.Value(3, 3)],
        ],
    }

def process_component(shape_tool, label, parent_trsf, density, components_data):
    """
    Recursively process components of the assembly.
    """
    # Get component name
    name_entry = TDF_Label()
    if label.FindAttribute(XCAFDoc_DocumentTool.GetShapeNameAttributeID(), name_entry):
        name = name_entry.Get().GetString()
    else:
        name = "Unnamed"

    # Get location of the component
    loc = shape_tool.GetLocation(label)
    trsf = loc.Transformation()
    
    # Combine with parent transformation
    global_trsf = parent_trsf * trsf

    # Check if it's a reference to another part or a simple shape
    if shape_tool.IsReference(label):
        ref_label = TDF_Label()
        shape_tool.GetReferredShape(label, ref_label)
        process_component(shape_tool, ref_label, global_trsf, density, components_data)

    # Check if it's an assembly
    elif shape_tool.IsAssembly(label):
        sub_labels = TDF_LabelSequence()
        shape_tool.GetComponents(label, sub_labels)
        for i in range(sub_labels.Length()):
            process_component(shape_tool, sub_labels.Value(i + 1), global_trsf, density, components_data)

    # It's a simple part
    else:
        shape = shape_tool.GetShape(label)
        if not shape.IsNull():
            # Move the shape to its global position before calculating properties
            moved_shape = shape.Moved(TopLoc_Location(global_trsf))
            
            properties = get_shape_properties(moved_shape, density)
            
            # Get the coordinates from the transformation
            t = global_trsf.TranslationPart()
            
            components_data.append({
                "name": name,
                "coordinates": [t.X(), t.Y(), t.Z()],
                "mass_properties": properties
            })

def get_properties_from_step(file_path, density=7.85e-6): # Density for steel in kg/mm^3
    """
    Reads a STEP file and extracts mass properties for the assembly and its components.
    """
    doc = TDocStd_Document()
    reader = STEPCAFControl_Reader()
    if not reader.ReadFile(file_path):
        print(f"Error: Could not read file {file_path}")
        return None

    reader.Transfer(doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())

    root_labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(root_labels)

    components_data = []
    assembly_props = GProp_GProps()
    
    # Identity transformation for the root
    identity_trsf = gp_Trsf()

    for i in range(root_labels.Length()):
        root_label = root_labels.Value(i + 1)
        process_component(shape_tool, root_label, identity_trsf, density, components_data)

    # Aggregate properties for the whole assembly
    total_mass = 0
    total_cg = gp_Vec(0, 0, 0)
    
    for comp in components_data:
        mass = comp["mass_properties"]["mass"]
        cg = comp["mass_properties"]["cg"]
        
        total_mass += mass
        total_cg += gp_Vec(cg[0], cg[1], cg[2]) * mass

    if total_mass > 0:
        final_cg = total_cg / total_mass
    else:
        final_cg = gp_Vec(0,0,0)

    # For MOI, a more complex aggregation is needed (Parallel Axis Theorem)
    # This is a simplified aggregation. For accurate MOI, a full calculation is needed.
    assembly_moi = [[0,0,0],[0,0,0],[0,0,0]] # Placeholder for simplicity

    return {
        "assembly_mass_properties": {
            "mass": total_mass,
            "cg": [final_cg.X(), final_cg.Y(), final_cg.Z()],
            "moi": assembly_moi # Placeholder
        },
        "individual_parts": components_data
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        properties = get_properties_from_step(file_path)
        if properties:
            print(json.dumps(properties, indent=4))
    else:
        print("Usage: python Get_properties_from_stp.py <path_to_step_file>")

