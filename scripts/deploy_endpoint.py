import os
import boto3
import botocore
from src import logger
import re

def normalize_version(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "-", value)

sm = boto3.client("sagemaker", region_name=os.environ["AWS_REGION"])

model_package_arn = os.environ["MODEL_PACKAGE_ARN"]
endpoint_name = os.environ["ENDPOINT_NAME"]
execution_role = os.environ["EXECUTION_ROLE"]
model_version = normalize_version((
    os.environ.get("MODEL_VERSION")
    or os.environ.get("GITHUB_REF_NAME")
    or os.environ.get("GITHUB_SHA", "dev")[:7]
    ))

s3_bucket = os.environ["S3_BUCKET"]

model_name = f"penguin-classifier-{model_version}"
config_name = f"penguin-config-{model_version}"

def model_exists(sm, model_name: str) -> bool:
    try:
        sm.describe_model(ModelName=model_name)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False
        raise

def endpoint_config_exists(sm, config_name: str) -> bool:
    try:
        sm.describe_endpoint_config(EndpointConfigName=config_name)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False
        raise


if model_exists(sm,model_name):
    logger.info(f"Sagemaker model already exists: {model_name}")
else:
    logger.info(f"Creating SageMaker model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=execution_role,
        Containers=[{"ModelPackageName": model_package_arn}],
    )

if endpoint_config_exists(sm, config_name):
    logger.info(f"Sagemaker endpoint config already exists: {config_name}")
else:
    logger.info(f"Creating endpoint config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": "ml.t2.medium",
            "InitialInstanceCount": 1,
        }],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": f"s3://{s3_bucket}/monitoring/captured-data",
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"},
            ],
        },
    )

try:
    sm.describe_endpoint(EndpointName=endpoint_name)
    logger.info(f"Updating existing endpoint: {endpoint_name}")
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "ValidationException":
        logger.info(f"Creating new endpoint: {endpoint_name}")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    else:
        raise

logger.info("Waiting for endpoint to be InService (this takes ~5 min)...")
sm.get_waiter("endpoint_in_service").wait(
    EndpointName=endpoint_name,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40},
)
logger.info(f"Endpoint {endpoint_name} is InService.")
