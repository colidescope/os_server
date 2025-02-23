# Open Shape Server

![Deploy status](https://github.com/colidescope/os_server/actions/workflows/deploy.yml/badge.svg)

## Install conda

https://www.anaconda.com/download/success

## Helpful Conda commands

```
conda config --add channels conda-forge
conda create --name=pyoccenv python=3.10
conda activate pyoccenv
conda install -c conda-forge pythonocc-core=7.8.1
conda deactivate
conda env remove --name pyoccenv
conda env list
```

## Run local

```
run_local.bat
```

## Deploy to Google Cloud

1. Install Docker CLI: https://www.docker.com/products/cli/

2. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install

3. Authenticate with Google Cloud and log in to project:

```
gcloud auth login
gcloud config set project open-shape-server
```

4. Enable Google Cloud Run service:

```
gcloud services enable run.googleapis.com
```

5. Enable Artifact Registry and create repository: https://console.cloud.google.com/artifacts

```
gcloud services enable artifactregistry.googleapis.com
gcloud artifacts repositories create os-server --repository-format=docker --location=us-central1
```

1. Build container image:

```
docker build -t os-server .
```

6. Deploy on local: http://localhost:8000/

```
docker run --env-file .env -p 8000:8000 os-server
```

7. Give build a tag to register with Google Artifact Registry

```
docker tag os-server us-central1-docker.pkg.dev/[PROJECT_NAME]/os-server/os-server:latest
```

8. Push latest image to Google Artifact Registry

```
docker push us-central1-docker.pkg.dev/[PROJECT_NAME]/os-server/os-server:latest
```

9. Deploy latest image on Google Cloud Run with these settings (keep remaining default):

```
gcloud run deploy os-server --image=[LOCATION]-docker.pkg.dev/[PROJECT_ID]/os-server/os-server:latest --platform=managed --region=us-central1 --allow-unauthenticated --port=8000
```

Once deployed, API will be accessible at the URL specified on the service details page: https://console.cloud.google.com/run/detail/us-central1/os-server/

![image](https://github.com/user-attachments/assets/549f8500-0887-4603-bb6d-790b956380d6)

To modify your deployment settings, click the `Edit & Deploy New Version` button on the service details page. Here you can modify the deployment settings including adding your environmental variables as described below.

![image](https://github.com/user-attachments/assets/ef3081a4-55b6-47ad-aef0-69678fd55c3e)

![image](https://github.com/user-attachments/assets/369adf18-5c98-4658-b979-60b97af064cb)

## Environment variables

The following variables must be set as `Repository secrets` in the Github repo for the Github Actions CI/CD to work.

```
SERVICE_NAME: os-server
REGION: us-central1
GCP_SA_KEY: [GCloud service account key]
GCP_PROJECT_ID: open-shape-server
```
