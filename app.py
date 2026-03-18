from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from geom.occ_helpers import (
    mesh_surface_to_occ_face,
    face_area,
    read_shape_with_occ,
    shape_to_tri_mesh,
)
from geom.occ_backend import OCCBackend
from scripts.bracket import main

import os
import json
import tempfile
import requests
import csv
import base64

from uuid import uuid4
from typing import Optional, List, Dict, Any, Literal
from dotenv import load_dotenv

from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Extend.DataExchange import read_step_file, write_step_file


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
    allow_origins=origins,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def geo_to_step_base64(geo):
    local_filename = f"{uuid4()}.step"
    write_step_file(geo, local_filename)

    with open(local_filename, "rb") as f:
        step_data = base64.b64encode(f.read()).decode("utf-8")

    os.remove(local_filename)
    return step_data


def geo_to_stl_base64(geo):
    BRepMesh_IncrementalMesh(geo, 0.1, True, 0.1, True)

    local_filename = f"{uuid4()}.stl"
    writer = StlAPI_Writer()
    writer.Write(geo, local_filename)

    with open(local_filename, "rb") as f:
        stl_data = base64.b64encode(f.read()).decode("utf-8")

    os.remove(local_filename)
    return stl_data


def parse_csv_value(value: str):
    if value is None:
        return None

    value = value.strip()
    if value == "":
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


# -------------------------------------------------------------------
# Materials endpoint
# -------------------------------------------------------------------

@app.get("/materials")
def get_materials():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "data", "materials", "materials.csv")

        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404,
                detail=f"Materials file not found: {csv_path}",
            )

        materials = []
        with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                materials.append({
                    key: parse_csv_value(value)
                    for key, value in row.items()
                })

        return {
            "materials": materials,
            "count": len(materials),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read materials CSV: {e}",
        )


# -------------------------------------------------------------------
# Convert endpoint
# -------------------------------------------------------------------

class ConvertRequest(BaseModel):
    srcUrl: str
    filename: Optional[str] = None
    contentType: Optional[str] = None


@app.post("/convert")
def convert(req: ConvertRequest):
    with tempfile.TemporaryDirectory() as td:
        fname = req.filename or "source"
        src_path = os.path.join(td, fname)

        try:
            with requests.get(req.srcUrl, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(src_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {e}")

        try:
            shape = read_shape_with_occ(src_path)
            elements, num_verts, num_faces = shape_to_tri_mesh(shape)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Conversion failed: {e}")

        mesh_str = json.dumps(elements, separators=(",", ":"))
        return {
            "mesh": mesh_str,
            "vertexCount": num_verts,
            "triangleCount": num_faces,
        }


# -------------------------------------------------------------------
# Bracket request models
# -------------------------------------------------------------------

class GeometryPayload(BaseModel):
    positions: List[float]
    indices: Optional[List[int]] = None
    item_size: int = Field(..., alias="itemSize")
    space: Literal["world", "local"]

    model_config = ConfigDict(populate_by_name=True)


class MaterialRecord(BaseModel):
    material: str = Field(..., alias="Material")
    elastic_modulus_mpa: float = Field(..., alias="Elastic Modulus [MPa]")
    shear_modulus_mpa: float = Field(..., alias="Shear Modulus [MPa]")
    density_kg_m3: float = Field(..., alias="Density [kg/m?]")
    ultimate_strength_mpa: float = Field(..., alias="Ultimate Strength [MPa]")
    yield_strength_mpa: float = Field(..., alias="Yield Strength [MPa]")
    shear_strength_mpa: float = Field(..., alias="Shear Strength [MPa]")
    poissons_ratio: float = Field(..., alias="Poisson's Ratio [-]")

    model_config = ConfigDict(populate_by_name=True)

    def to_external_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)

    def to_internal_dict(self) -> Dict[str, Any]:
        """
        Returns a plain dict using the original material keys from the CSV/API.
        This is useful if downstream code expects those exact names.
        """
        return {
            "Material": self.material,
            "Elastic Modulus [MPa]": self.elastic_modulus_mpa,
            "Shear Modulus [MPa]": self.shear_modulus_mpa,
            "Density [kg/m?]": self.density_kg_m3,
            "Ultimate Strength [MPa]": self.ultimate_strength_mpa,
            "Yield Strength [MPa]": self.yield_strength_mpa,
            "Shear Strength [MPa]": self.shear_strength_mpa,
            "Poisson's Ratio [-]": self.poissons_ratio,
        }


class BracketGeoItem(BaseModel):
    object_id: str = Field(..., alias="objectId")
    object_index: int = Field(..., alias="objectIndex")
    surface_id: str = Field(..., alias="surfaceId")
    geometry: GeometryPayload
    volume: Optional[float] = None
    cg: Optional[List[float]] = None
    support_type: Literal["fixed", "free"] = Field(..., alias="supportType")
    material: Optional[MaterialRecord] = None

    model_config = ConfigDict(populate_by_name=True)


class BracketRequest(BaseModel):
    format: Literal["STEP", "STL"]
    geo: List[BracketGeoItem]


# -------------------------------------------------------------------
# Bracket endpoint
# -------------------------------------------------------------------

@app.post("/run-bracket")
async def run_bracket(req: BracketRequest):
    geo = req.geo
    output_format = req.format

    if len(geo) != 2:
        raise HTTPException(
            status_code=400,
            detail="Expected exactly 2 selected surfaces in geo",
        )

    try:
        face1 = mesh_surface_to_occ_face(
            positions=geo[0].geometry.positions,
            indices=geo[0].geometry.indices,
            item_size=geo[0].geometry.item_size,
        )
        face2 = mesh_surface_to_occ_face(
            positions=geo[1].geometry.positions,
            indices=geo[1].geometry.indices,
            item_size=geo[1].geometry.item_size,
        )

        a1 = face_area(face1)
        a2 = face_area(face2)
        print(f"face1 area: {a1:.6f}")
        print(f"face2 area: {a2:.6f}")

        free_objects = []
        for item in geo:
            if item.support_type == "free":
                free_objects.append({
                    "cg": item.cg,
                    "volume": item.volume,
                    "material": item.material.to_internal_dict() if item.material else None,
                })

        free_item = next((item for item in geo if item.support_type == "free"), None)

        material_free = (
            free_item.material.to_internal_dict()
            if free_item and free_item.material
            else None
        )

        material_bracket = {
            "Material": "S235 (Structural Steel)",
            "Elastic Modulus [MPa]": 210000,
            "Shear Modulus [MPa]": 81000,
            "Density [kg/m?]": 7850,
            "Ultimate Strength [MPa]": 360,
            "Yield Strength [MPa]": 235,
            "Shear Strength [MPa]": 200,
            "Poisson's Ratio [-]": 0.3,
        }

        for item in geo:
            print(
                "selected surface:",
                {
                    "objectId": item.object_id,
                    "surfaceId": item.surface_id,
                    "volume": item.volume,
                    "supportType": item.support_type,
                    "material": item.material.to_external_dict() if item.material else None,
                },
            )

        geom = OCCBackend()

        result = main(
            geom=geom,
            srf_1=face1,
            srf_2=face2,
            free_objects=free_objects,

            min_edge_clearance=50.0,
            num_ribs=2,
            init_plate_thickness=10.0,

            bolt_spacing_1=2.0,
            bolt_spacing_2=2.0,
            init_bolt_radius=8.0,
            max_bolt_dia=20.0,

            material_free=material_free,
            material_bracket=material_bracket,
        )

        bracket_shape = result["bracket_solid"]._topods()

        outputs: Dict[str, Any] = {}

        bracket_output: Dict[str, Any] = {}
        if output_format == "STEP":
            bracket_output["value"] = geo_to_step_base64(bracket_shape)
            bracket_output["format"] = output_format
        elif output_format == "STL":
            bracket_output["value"] = geo_to_stl_base64(bracket_shape)
            bracket_output["format"] = output_format

        outputs["mesh"] = [bracket_output]

        for data_key in ["cps_bolts_1", "cps_bolts_2"]:
            if data_key in result:
                outputs[data_key] = [pt._serialize() for pt in result.get(data_key, [])]

        for data_key in ["reactions", "bolts", "log"]:
            if data_key in result:
                outputs[data_key] = result.get(data_key)

        return {
            "outputs": outputs,
            "meta": {
                "input": [
                    {
                        "objectId": item.object_id,
                        "objectIndex": item.object_index,
                        "surfaceId": item.surface_id,
                        "volume": item.volume,
                        "cg": item.cg,
                        "supportType": item.support_type,
                        "material": item.material.to_external_dict() if item.material else None,
                    }
                    for item in geo
                ],
                "freeObjects": free_objects,
                "materialFree": material_free,
                "materialBracket": material_bracket,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run bracket: {e}",
        )


# -------------------------------------------------------------------
# OLD / legacy script runner
# -------------------------------------------------------------------

class CreateObjectBody(BaseModel):
    script: str
    format: str


def create_vector(params):
    print("Generating vector with params:", params["X"], params["Y"], params["Z"])
    vec = gp_Vec(params["X"], params["Y"], params["Z"])
    return [{"name": "V", "type": "vector", "value": vec}]


def move(geo, vec):
    transform = gp_Trsf()
    transform.SetTranslation(vec)
    transformer = BRepBuilderAPI_Transform(transform)
    transformer.Perform(geo, True)
    return transformer.Shape()


def create_box(params):
    print("Generating cube with params:", params["X"], params["Y"], params["Z"])
    cube = BRepPrimAPI_MakeBox(
        params["X"] * 2,
        params["Y"] * 2,
        params["Z"] * 2,
    ).Shape()
    moved_cube = move(cube, gp_Vec(-params["X"], -params["Y"], -params["Z"]))
    return [{"name": "B", "type": "brep", "value": moved_cube}]


def create_sphere(params):
    print("Generating sphere with params:", params["R"])
    geo = BRepPrimAPI_MakeSphere(params["R"]).Shape()
    return [{"name": "S", "type": "brep", "value": geo}]


def move_geo(params):
    print("Moving geo with params:", params)
    geo = move(params["G"], params["T"])
    return [{"name": "G", "type": "brep", "value": geo}]


function_map = {
    "Vec": create_vector,
    "Box": create_box,
    "Sphere": create_sphere,
    "Move": move_geo,
}


@app.get("/items")
def get_items():
    return list(function_map)


def find_execution_order(components):
    all_components = [comp["name"] for comp in components]
    dependents_map = {comp["name"]: [] for comp in components}
    in_degree = {comp["name"]: 0 for comp in components}

    for comp in components:
        name = comp["name"]
        dependencies = comp.get("depends_on", [])
        for dep in dependencies:
            if dep not in dependents_map:
                raise ValueError(
                    f"Dependency '{dep}' of component '{name}' is not defined."
                )
            dependents_map[dep].append(name)
            in_degree[name] += 1

    from collections import deque
    queue = deque([comp_name for comp_name, deg in in_degree.items() if deg == 0])

    execution_order = []

    while queue:
        current = queue.popleft()
        execution_order.append(current)

        for dep in dependents_map[current]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

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
                        "param": link_param,
                    }

            component_def.append({
                "name": component_index,
                "depends_on": links,
            })

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
                    step_bytes = base64.b64decode(step_data_base64)

                    temp_filename = f"{uuid4()}.step"
                    with open(temp_filename, "wb") as f:
                        f.write(step_bytes)

                    occ_shape = read_step_file(temp_filename)
                    input["value"] = occ_shape
                    os.remove(temp_filename)

                if input["type"] == "link":
                    prev_outputs = results_dict[input["value"]["link_index"]]
                    output_dictionary = {
                        output["name"]: output["value"]
                        for output in prev_outputs
                    }
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
            "outputs": output_results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute script: {e}")