from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Extend.DataExchange import read_step_file, write_step_file

import os, json
from collections import deque
from uuid import uuid4
import base64

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
    
        # ---

        # s3_key = "geo_files/{}".format(local_filename)
        # with open(local_filename, "rb") as file:
        #     s3_client.upload_fileobj(file, S3_BUCKET_NAME, s3_key)

        # # Generate a pre-signed URL for downloading the file
        # s3_file_url = s3_client.generate_presigned_url(
        #     "get_object",
        #     Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
        #     ExpiresIn=3600  # URL expiration time in seconds
        # )

        # return {
        #     "path_to_file": s3_file_url
        # }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute script: {e}")
    