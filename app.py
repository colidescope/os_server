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

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, topods
from OCC.Core.BRep import BRep_Builder, BRep_Tool
import OCC.Core.BRepTools as BRepTools
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location

import os, json
from collections import deque
from uuid import uuid4
import base64
import tempfile
import requests
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

# Pydantic model for request
class ConvertRequest(BaseModel):
    srcUrl: str
    filename: Optional[str] = None
    contentType: Optional[str] = None

def read_shape_with_occ(path: str) -> TopoDS_Shape:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.step', '.stp']:
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != 1:
            raise RuntimeError('Failed to read STEP file')
        reader.TransferRoots()
        return reader.OneShape()

    if ext in ['.iges', '.igs']:
        reader = IGESControl_Reader()
        status = reader.ReadFile(path)
        if status != 1:
            raise RuntimeError('Failed to read IGES file')
        reader.TransferRoots()
        return reader.OneShape()

    if ext == '.brep':
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        if not BRepTools.Read(shape, path, builder):
            raise RuntimeError('Failed to read BREP file')
        return shape

    # STL/OBJ fallback via trimesh if available
    if ext in ['.stl', '.obj'] and trimesh is not None:
        return None # signal to use trimesh path

    raise RuntimeError(f'Unsupported extension: {ext}')

def shape_to_tri_mesh(
    shape: TopoDS_Shape,
    lin_deflection: float = 0.5,
    ang_deflection: float = 0.5,
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Triangulate a TopoDS_Shape and return (vertices, faces) with deduped vertices.
    Faces are triangles [i1,i2,i3].
    """
    # Ensure the shape is meshed
    # (args signature varies by version; this form is broadly compatible)
    BRepMesh_IncrementalMesh(shape, lin_deflection, False, ang_deflection, True)

    surfaces = []
    num_verts = 0
    num_faces = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        
        vertices: List[List[float]] = []
        faces: List[List[int]] = []
        vmap: Dict[Tuple[int, int, int], int] = {}

        face = topods.Face(exp.Current())

        loc = TopLoc_Location()
        h_triangulation = BRep_Tool.Triangulation(face, loc)

        # Some versions return None; others a null handle
        if not h_triangulation:
            exp.Next()
            continue
        # Handle -> object
        tri = getattr(h_triangulation, "GetObject", lambda: h_triangulation)()

        # Still nothing? skip
        if not tri or tri.NbTriangles() == 0:
            exp.Next()
            continue

        trsf = loc.Transformation()

        # Iterate triangles (1-based indices)
        for ti in range(1, tri.NbTriangles() + 1):
            t = tri.Triangle(ti)
            i1, i2, i3 = t.Get()

            tri_idx: List[int] = []
            for ii in (i1, i2, i3):
                # Use Node(i) and apply the face location transform
                gp = tri.Node(ii).Transformed(trsf)
                key = (round(gp.X() * 1_000_000), round(gp.Y() * 1_000_000), round(gp.Z() * 1_000_000))
                idx = vmap.get(key)
                if idx is None:
                    idx = len(vertices)
                    vmap[key] = idx
                    vertices.append([float(gp.X()), float(gp.Y()), float(gp.Z())])
                tri_idx.append(idx)

            faces.append(tri_idx)

        surfaces.append({ "vertices": vertices, "faces": faces})
        num_verts += len(vertices)
        num_faces += len(faces)

        exp.Next()

    return surfaces, num_verts, num_faces

def trimesh_to_arrays(mesh: 'trimesh.Trimesh') -> Tuple[List[List[float]], List[List[int]]]:
    v = mesh.vertices.tolist()
    f = mesh.faces.tolist()
    return { "vertices": v, "faces": f}, len(v), len(f)

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
            if shape is None:
                if trimesh is None:
                    raise RuntimeError('trimesh not available for OBJ/STL fallback')
                mesh = trimesh.load(src_path, force='mesh') # type: ignore
                surfaces, num_verts, num_faces = trimesh_to_arrays(mesh)
            else:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute script: {e}")
    