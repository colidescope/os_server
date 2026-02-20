from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from geom.occ_helpers import mesh_surface_to_occ_face, face_area, read_shape_with_occ, shape_to_tri_mesh
from geom.occ_backend import OCCBackend
from scripts.bracket import main

import os, json, tempfile, requests
from uuid import uuid4
import base64
from collections import deque
from typing import Optional, Tuple, List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file if not in production
ENV = os.getenv("ENV", "DEV").upper()
if ENV != "PROD":
    load_dotenv()

app = FastAPI()

origins = [
    "https://openshape.netlify.app",
    "https://synthyon.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] to allow all
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],         # or list e.g. ["GET", "POST"]
    allow_headers=["*"],         # or list e.g. ["Authorization", "Content-Type"]
)

## Helpers ##

def geo_to_step_base64(geo):
    # Save file locally
    local_filename = "{}.step".format(uuid4())
    write_step_file(geo, local_filename)

    # Read STEP file and encode it as base64
    with open(local_filename, "rb") as f:
        step_data = base64.b64encode(f.read()).decode("utf-8")

    # Remove temp file
    os.remove(local_filename)
    
    return step_data

def geo_to_stl_base64(geo):
    # 2) Mesh the shape so it can be exported as STL
    #    (parameters: linear deflection=0.1, angular deflection=0.1, parallel=False)
    BRepMesh_IncrementalMesh(geo, 0.1, True, 0.1, True)

    # Save file locally
    local_filename = "{}.stl".format(uuid4())
    writer = StlAPI_Writer()
    writer.Write(geo, local_filename)

    # 4) Read file & encode to base64
    with open(local_filename, "rb") as f:
        # stl_data = f.read()
        stl_data = base64.b64encode(f.read()).decode("utf-8")

    # Remove temp file
    os.remove(local_filename)
    
    return stl_data

# Pydantic model for request
class ConvertRequest(BaseModel):
    srcUrl: str
    filename: Optional[str] = None
    contentType: Optional[str] = None

@app.post('/convert')
def convert(req: ConvertRequest):
        
    # 1) Download source via HTTPS to a temp file
    with tempfile.TemporaryDirectory() as td:
        fname = req.filename or 'source'
        src_path = os.path.join(td, fname)
        try:
            with requests.get(req.srcUrl, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(src_path, 'wb') as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Download failed: {e}')

        # 2) Convert
        try:
            shape = read_shape_with_occ(src_path)
            surfaces, num_verts, num_faces = shape_to_tri_mesh(shape)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Conversion failed: {e}')

        # 3) Return the exact JSON string format
        mesh_str = json.dumps(surfaces, separators=(',', ':'))
        return {
            'mesh': mesh_str,
            'vertexCount': num_verts,
            'triangleCount': num_faces,
        }

# Pydantic model for request
class BracketRequest(BaseModel):
    format: str
    geo: str

@app.post('/run-bracket')
async def run_bracket(req: BracketRequest):

    try:
    
        geo = json.loads(req.geo)
        output_format = req.format

        if len(geo) != 2:
            raise HTTPException(status_code=400, detail="Expected exactly 2 surfaces in srf[]")

        face1 = mesh_surface_to_occ_face(
            positions=geo[0]["geometry"]["positions"],
            indices=geo[0]["geometry"]["indices"],
            item_size=geo[0]["geometry"]["itemSize"],
        )
        face2 = mesh_surface_to_occ_face(
            positions=geo[1]["geometry"]["positions"],
            indices=geo[1]["geometry"]["indices"],
            item_size=geo[1]["geometry"]["itemSize"],
        )

        a1 = face_area(face1)
        a2 = face_area(face2)
        print(f"face1 area: {a1:.6f}")
        print(f"face2 area: {a2:.6f}")

        geom = OCCBackend()

        result = main(
            geom=geom,
            srf_1=face1,
            srf_2=face2,
            free_objects=[],

            min_edge_clearance=50.0,
            num_ribs=1,
            init_plate_thickness=10.0,
            
            bolt_spacing_1=2.0,
            bolt_spacing_2=2.0,
            init_bolt_radius=8.0,
            max_bolt_dia=20.0,

            material_dictionary={
                "Steel": {"Density": "7850", "Yield Strength": "250"},  # example
                "Aluminum": {"Density": "2700", "Yield Strength": "150"},
            },
            material_free="Steel",
            material_bracket="Steel",
        )

        bracket_shape = result["bracket_solid"]._topods()        # TopoDS_Shape (or OCCSolid.shape)
        bolt_shapes = result.get("bolts_geo", [])

        output = {}

        if output_format == "STEP":
            output["value"] = geo_to_step_base64(bracket_shape)
            output["format"] = output_format
        elif output_format == "STL":
            output["value"] = geo_to_stl_base64(bracket_shape)
            output["format"] = output_format
        
        return {
            "outputs": [output]
        }

        ## Return the exact JSON string format ##
        
        # Convert
        surfaces, num_verts, num_faces = shape_to_tri_mesh(bracket_shape)

        mesh_str = json.dumps(surfaces, separators=(',', ':'))

        return {
            'mesh': mesh_str,
            'vertexCount': num_verts,
            'triangleCount': num_faces,
        }
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute run-bracket script: {e}")


# OLD # 

from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Extend.DataExchange import read_step_file, write_step_file
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

class CreateObjectBody(BaseModel):
    script: str
    format: str

def create_vector(params):
    print("Generating vector with params:", params["X"], params["Y"], params["Z"])

    vec = gp_Vec(params["X"], params["Y"], params["Z"])

    return [{ "name": "V", "type": "vector", "value": vec }]

def move(geo, vec):
    transform = gp_Trsf()
    transform.SetTranslation(vec)
    transformer = BRepBuilderAPI_Transform(transform)
    transformer.Perform(geo, True)  # True = copy geometry; you can pass False to apply in place.
    return transformer.Shape()

def create_box(params):
    print("Generating cube with params:", params["X"], params["Y"], params["Z"])

    cube = BRepPrimAPI_MakeBox(params["X"] * 2, params["Y"] * 2, params["Z"] * 2).Shape()
    moved_cube = move(cube, gp_Vec(-params["X"], -params["Y"], -params["Z"]))

    return [{ "name": "B", "type": "brep", "value": moved_cube }]

def create_sphere(params):
    print("Generating sphere with params:", params["R"])

    geo = BRepPrimAPI_MakeSphere(params["R"]).Shape()

    return [{ "name": "S", "type": "brep", "value": geo }]

def move_geo(params):
    print("Moving geo with params:", params)

    geo =  move(params["G"], params["T"])

    return [{ "name": "G", "type": "brep", "value": geo }]

function_map = {
    "Vec": create_vector,
    "Box": create_box,
    "Sphere": create_sphere,
    "Move": move_geo
}

@app.get("/items")
def get_items():
    return list(function_map)

def find_execution_order(components):
    """
    Given a JSON string of components with their dependencies,
    return a list of component names in a valid execution order.
    """

    # Extract all component names
    all_components = [comp["name"] for comp in components]

    # Build a dictionary to map component name -> list of components that depend on it
    dependents_map = {comp["name"]: [] for comp in components}

    # Build a dictionary to keep track of in-degrees (number of dependencies for each component)
    in_degree = {comp["name"]: 0 for comp in components}

    # Fill the above structures
    for comp in components:
        name = comp["name"]
        dependencies = comp.get("depends_on", [])
        # For every dependency, mark 'name' as a dependent
        # and increment 'name''s in-degree
        for dep in dependencies:
            if dep not in dependents_map:
                # Optional: handle case where dep isn't defined in the list
                raise ValueError(f"Dependency '{dep}' of component '{name}' is not defined.")
            dependents_map[dep].append(name)
            in_degree[name] += 1

    # Queue (or list) for components with no dependencies (in-degree = 0)
    queue = deque([comp_name for comp_name, deg in in_degree.items() if deg == 0])

    execution_order = []

    while queue:
        # Pop a component that has no remaining dependencies
        current = queue.popleft()
        execution_order.append(current)

        # Decrease the in-degree of the dependent components
        for dep in dependents_map[current]:
            in_degree[dep] -= 1
            # If in-degree becomes zero, add it to the queue
            if in_degree[dep] == 0:
                queue.append(dep)

    # If we processed all components, execution_order should have the same length
    # as all_components. If not, there is a cycle or missing dependencies.
    if len(execution_order) != len(all_components):
        raise ValueError("There is a cycle or unsatisfiable dependency in the components.")

    return execution_order

@app.post("/run-script")
async def run_script(object_data: CreateObjectBody):

    try:
    
        output_format = object_data.format

        components = json.loads(object_data.script)
        
        component_dictionary = {}

        for component_index, component in enumerate(components):
            id = component.get("id")

            if id is not None:
                component_dictionary[id] = component_index

        component_def = []

        for component_index, component in enumerate(components):
            inputs = component.get("inputs")

            links = []

            for input in inputs:
                if input["type"] == "link":
                    link, link_param = input["value"].split(".")
                    link_index = component_dictionary[link]
                    if link_index not in links:
                        links.append(link_index)
                    input["value"] = {
                        "link_index": link_index,
                        "param": link_param
                    }
            
            component_def.append(
                {
                    "name": component_index,
                    "depends_on": links
                },
            )

        order = find_execution_order(component_def)

        results_dict = {}

        for component_index in order:
            component = components[component_index]

            name = component.get("name")
            inputs = component.get("inputs")

            params = {}

            for input in inputs:
                if input["type"] == "brep":
                    step_data_base64 = input["value"]

                    # 1) Decode the Base64 STEP
                    step_bytes = base64.b64decode(step_data_base64)

                    # 2) Write to a temp file
                    temp_filename = f"{uuid4()}.step"
                    with open(temp_filename, "wb") as f:
                        f.write(step_bytes)

                    # 3) Convert to python-occ shape
                    occ_shape = read_step_file(temp_filename)
                    
                    input["value"] = occ_shape

                    # Clean up temp file
                    os.remove(temp_filename)
                
                if input["type"] == "link":
                    prev_outputs = results_dict[input["value"]["link_index"]]
                    output_dictionary = { output["name"]: output["value"] for output in prev_outputs }
                    input["value"] = output_dictionary[input["value"]["param"]]

                params[input["name"]] = input["value"]

            outputs = function_map[name](params)

            results_dict[component_index] = outputs
        
        output_results = results_dict[len(components) - 1]

        for output in output_results:
                if output["type"] == "brep":

                    if output_format == "STEP":
                        output["value"] = geo_to_step_base64(output["value"])
                        output["format"] = output_format
                    elif output_format == "STL":
                        output["value"] = geo_to_stl_base64(output["value"])
                        output["format"] = output_format
        
        return {
            "outputs": output_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute script: {e}")
    