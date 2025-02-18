from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps

from OCC.Extend.DataExchange import write_step_file

import os, json
import boto3
from uuid import uuid4

from dotenv import load_dotenv

# Load environment variables from .env file if not in production
ENV = os.getenv("ENV", "DEV").upper()
if ENV != "PROD":
    load_dotenv()

app = FastAPI()

class CreateObjectBody(BaseModel):
    script: str

# Fetch S3 configuration from environment variables
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_REGION = os.getenv("S3_REGION")

# Validate that all necessary environment variables are set
if not all([S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_REGION]):
    raise RuntimeError("Missing one or more required AWS environment variables.")


# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=S3_REGION
)

def create_cube(params):
    print("Generating cube with params:", params["x"], params["y"], params["z"])
    return BRepPrimAPI_MakeBox(params["x"], params["y"], params["z"]).Shape()

def create_sphere(params):
    print("Generating sphere with params:", params["R"])
    return BRepPrimAPI_MakeSphere(params["R"]).Shape()

function_map = {
    "Cube": create_cube,
    "Sphere": create_sphere
}

@app.get("/items")
def get_items():
    return list(function_map)

@app.post("/run-script")
async def run_script(object_data: CreateObjectBody):
    
    script = json.loads(object_data.script)

    component = script[0]
    
    name = component.get("name")
    inputs = component.get("inputs")

    try:
        geometry = function_map[name](inputs)

        # Save file locally
        local_filename = "{}.step".format(uuid4())
        write_step_file(geometry, local_filename)

        s3_key = "geo_files/{}".format(local_filename)
        with open(local_filename, "rb") as file:
            s3_client.upload_fileobj(file, S3_BUCKET_NAME, s3_key)

        # Generate a pre-signed URL for downloading the file
        s3_file_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=3600  # URL expiration time in seconds
        )

        # Clean up local file
        os.remove(local_filename)

        return {
            "path_to_file": s3_file_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create the BREP data: {e}")

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
