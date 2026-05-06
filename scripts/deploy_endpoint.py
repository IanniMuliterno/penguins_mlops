import os
import boto3
import botocore
from src import logger

sm = boto3.client("sagemaker", region_name=os.environ["AWS_REGION"])

model_package_arn = os.environ["MODEL_PACKAGE_ARN"]
endpoint_name = os.environ["ENDPOINT_NAME"]
execution_role = os.environ["EXECUTION_ROLE"]
model_version = os.environ["MODEL_VERSION"]
s3_bucket = os.environ["S3_BUCKET"]

model_name = f"penguin-classifier-{model_version}"
config_name = f"penguin-config-{model_version}"

logger.info(f"Creating SageMaker model: {model_name}")
sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=execution_role,
    Containers=[{"ModelPackageName": model_package_arn}],
)

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
