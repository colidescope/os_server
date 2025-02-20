import httplib  # For Python 2.7 and IronPython
import json
import os
import base64
import rhinoscriptsyntax as rs
import scriptcontext as sc
import Rhino
from System.IO import Path


def call_create_object_api(script):
    """
    Calls the FastAPI service to generate the STEP file (base64-encoded) 
    and returns the response JSON.
    """
    
    # DEV / LOCAL
    # conn = httplib.HTTPConnection("127.0.0.1", 8000)
    # PROD
    conn = httplib.HTTPSConnection("py-occ-server-1056113226980.us-central1.run.app", 443)
    
    # Prepare the JSON request body
    body = json.dumps({
        "script": script,
        "format": "STEP"
    })
    
    # Set the headers for a JSON POST request
    headers = {
        "Content-type": "application/json",
        "Accept": "application/json"
    }

    # Make the request
    conn.request("POST", "/run-script", body, headers)
    response = conn.getresponse()
    
    # Check the response status
    if response.status != 200:
        raise ValueError("API call failed with status {}: {}".format(response.status, response.read().decode("utf-8")))

    # Parse the response
    data = json.loads(response.read().decode("utf-8"))
    
    return data

def get_all_object_ids():
    """
    Returns a set of GUIDs for all objects in the active Rhino document.
    """
    doc = Rhino.RhinoDoc.ActiveDoc
    settings = Rhino.DocObjects.ObjectEnumeratorSettings()
    # settings.HiddenObjects = True  # Example: if you also want hidden objects
    # settings.LockedObjects = True  # Example: if you also want locked objects

    obj_list = doc.Objects.GetObjectList(settings)
    return set(obj.Id for obj in obj_list if obj is not None)


# step_base64_to_brep

def decode_and_save_step(step_data_b64):
    """
    Decodes the Base64-encoded STEP file from the API response 
    and saves it to a temporary file.
    """
    # step_data_b64 = json_response["step_data"]
    step_bytes = base64.b64decode(step_data_b64)

    temp_filename = Path.GetTempFileName() + ".step"
    with open(temp_filename, "wb") as f:
        f.write(step_bytes)

    return temp_filename  # Return the temp file path

def import_step_to_brep(file_path):
    """
    Uses Rhino's import command to load a STEP file invisibly,
    extracts the Breps, and removes the temporary geometry from Rhino.
    """
    if not os.path.exists(file_path):
        raise ValueError("File not found: {}".format(file_path))

    # Track existing object IDs
    existing_ids = get_all_object_ids()

    # Import STEP file invisibly
    rs.Command('-Import "{}" Enter'.format(file_path), echo=False)

    # Identify newly added objects (difference)
    new_ids = get_all_object_ids() - existing_ids

    doc = Rhino.RhinoDoc.ActiveDoc
    breps = []

    for obj_id in new_ids:
        rh_obj = doc.Objects.Find(obj_id)
        if not rh_obj:
            continue

        # Extract BREP geometry
        geo = rh_obj.Geometry
        if isinstance(geo, Rhino.Geometry.Brep):
            breps.append(geo)

        # Delete the temporary object from the Rhino doc
        doc.Objects.Delete(rh_obj, quiet=True)

    return breps if breps else None

def step_base64_to_brep(step_data_base64):
    temp_file = decode_and_save_step(step_data_base64)
    breps = import_step_to_brep(temp_file)

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return breps


# step_base64_to_brep

def decode_and_save_stl(stl_data_b64):
    """
    Decodes the Base64-encoded STL file from the API response 
    and saves it to a temporary file.
    """
    # step_data_b64 = json_response["step_data"]
    stl_bytes = base64.b64decode(stl_data_b64)

    temp_filename = Path.GetTempFileName() + ".stl"
    with open(temp_filename, "wb") as f:
        f.write(stl_bytes)

    return temp_filename  # Return the temp file path

def import_stl_to_mesh(file_path):
    """
    Uses Rhino's import command to load a STL file invisibly,
    extracts the Meshes, and removes the temporary geometry from Rhino.
    """
    if not os.path.exists(file_path):
        raise ValueError("File not found: {}".format(file_path))

    # Track existing object IDs
    existing_ids = get_all_object_ids()

    # Import STEP file invisibly
    rs.Command('-Import "{}" Enter'.format(file_path), echo=False)

    # Identify newly added objects (difference)
    new_ids = get_all_object_ids() - existing_ids

    doc = Rhino.RhinoDoc.ActiveDoc
    meshes = []

    for obj_id in new_ids:
        rh_obj = doc.Objects.Find(obj_id)
        if not rh_obj:
            continue

        # Extract BREP geometry
        geo = rh_obj.Geometry
        if isinstance(geo, Rhino.Geometry.Mesh):
            meshes.append(geo)

        # Delete the temporary object from the Rhino doc
        doc.Objects.Delete(rh_obj, quiet=True)

    return meshes if meshes else None

def stl_base64_to_brep(stl_data_base64):
    temp_file = decode_and_save_stl(stl_data_base64)
    meshes = import_stl_to_mesh(temp_file)

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return meshes


# brep_to_step_base64

def add_brep_to_rhino(gh_brep):
    # 1. Switch to the active Rhino document
    sc.doc = Rhino.RhinoDoc.ActiveDoc

    # 2. Add the Brep to the Rhino doc
    obj_id = sc.doc.Objects.AddBrep(gh_brep)

    # 3. Redraw Rhino views to see the new object
    sc.doc.Views.Redraw()

    return obj_id

def brep_to_step_base64(gh_brep):
    """
    1. Temporarily add the GH Brep to the Rhino document.
    2. Use an invisible '-Export' to write a STEP file.
    3. Read and Base64-encode the STEP data.
    4. Delete the temp geometry & file.
    """
    if not gh_brep:
        return None

    # 1. Add the Brep to Rhino
    obj_id = add_brep_to_rhino(gh_brep)

    # 2. Export to a temporary STEP file
    temp_filename = Path.GetTempFileName() + ".step"
    rs.SelectObject(obj_id)
    rs.Command('-Export "{}" Enter'.format(temp_filename), echo=False)

    # 3. Read & Base64-encode
    with open(temp_filename, 'rb') as f:
        step_data = f.read()
    step_data_base64 = base64.b64encode(step_data).decode('utf-8')

    # Clean up: delete the object and temp file
    rs.DeleteObject(obj_id)
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return step_data_base64


def process_script(script, vars):
    script_decoded = json.loads(script)
    
    for component in script_decoded:
        inputs = component["inputs"]
        
        for input in inputs:
            if input["type"] == "brep":
                input["value"] = brep_to_step_base64(vars[input["value"]])
    
    return json.dumps(script_decoded)


def run_script(script):
    """
    1) Calls the API (which returns base64-encoded STEP data).
    2) Decodes & saves the STEP file to a temp location.
    3) Imports it invisibly, extracts BREP objects, and deletes them from the Rhino doc.
    4) Returns the BREP(s) to Grasshopper.
    """
    outputs = call_create_object_api(script)["outputs"]

    print(outputs)

    for output in outputs:
        if output["type"] == "brep":
            if output["format"] == "STEP":
                output["value"] = step_base64_to_brep(output["value"])
            elif output["format"] == "STL":
                output["value"] = stl_base64_to_brep(output["value"])
                print(output["value"])

    return outputs
