from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties, brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps

from OCC.Extend.DataExchange import write_iges_file

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
    product: str
    params: str

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

def get_cube(params):
    print("Generating cube with params:", params["x"], params["y"], params["z"])
    return BRepPrimAPI_MakeBox(params["x"], params["y"], params["z"]).Shape()

def get_sphere(params):
    print("Generating sphere with params:", params["r"])
    return BRepPrimAPI_MakeSphere(params["r"]).Shape()

function_map = {
    "cube": get_cube,
    "sphere": get_sphere
}

@app.get("/items")
def get_items():
    return list(function_map)

@app.post("/create-object")
async def create_object(object_data: CreateObjectBody):
    
    product = object_data.product
    params = object_data.params

    print(product, params)

    try:
        geometry = function_map[product](json.loads(params))

        # Save IGES file locally
        local_iges_filename = "{}.iges".format(uuid4())
        write_iges_file(geometry, local_iges_filename)

        s3_key = "iges_files/{}".format(local_iges_filename)
        with open(local_iges_filename, "rb") as file:
            s3_client.upload_fileobj(file, S3_BUCKET_NAME, s3_key, ExtraArgs={"ACL": "public-read"})

        # Construct S3 file URL
        s3_file_url = "https://{}.s3.{}.amazonaws.com/{}".format(S3_BUCKET_NAME, S3_REGION, s3_key)

        # Clean up local file
        os.remove(local_iges_filename)

        return {
            "path_to_iges_file": s3_file_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create the BREP data: {e}")

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
