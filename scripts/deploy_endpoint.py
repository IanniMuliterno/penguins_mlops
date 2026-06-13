import os
import re

import boto3
import botocore

from src import logger


WAITER_CONFIG = {"Delay": 30, "MaxAttempts": 40}


def normalize_version(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "-", value)


def load_settings() -> dict[str, str]:
    model_version = normalize_version(
        os.environ.get("MODEL_VERSION")
        or os.environ.get("GITHUB_REF_NAME")
        or os.environ.get("GITHUB_SHA", "dev")[:7]
    )

    return {
        "aws_region": os.environ["AWS_REGION"],
        "model_package_arn": os.environ["MODEL_PACKAGE_ARN"],
        "endpoint_name": os.environ["ENDPOINT_NAME"],
        "execution_role": os.environ["EXECUTION_ROLE"],
        "s3_bucket": os.environ["S3_BUCKET"],
        "model_name": f"penguin-classifier-{model_version}",
        "config_name": f"penguin-config-{model_version}",
    }


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


def ensure_model(sm, *, model_name: str, execution_role: str, model_package_arn: str) -> None:
    if model_exists(sm, model_name):
        logger.info(f"Sagemaker model already exists: {model_name}")
        return

    logger.info(f"Creating SageMaker model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=execution_role,
        Containers=[{"ModelPackageName": model_package_arn}],
    )


def ensure_endpoint_config(sm, *, config_name: str, model_name: str, s3_bucket: str) -> None:
    if endpoint_config_exists(sm, config_name):
        logger.info(f"Sagemaker endpoint config already exists: {config_name}")
        return

    logger.info(f"Creating endpoint config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": "ml.t2.medium",
                "InitialInstanceCount": 1,
            }
        ],
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


def describe_endpoint(sm, endpoint_name: str) -> dict | None:
    try:
        return sm.describe_endpoint(EndpointName=endpoint_name)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return None
        raise


def create_endpoint(sm, *, endpoint_name: str, config_name: str) -> None:
    logger.info(f"Creating new endpoint: {endpoint_name}")
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)


def update_existing_endpoint(sm, *, endpoint_name: str, config_name: str, status: str) -> None:
    logger.info(f"Updating existing endpoint {endpoint_name} with current status {status}")
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)


def recreate_failed_endpoint(
    sm,
    *,
    endpoint_name: str,
    config_name: str,
    failure_reason: str | None,
) -> None:
    logger.warning(
        "Endpoint %s is in Failed status. FailureReason: %s",
        endpoint_name,
        failure_reason or "unknown",
    )
    logger.info(f"Deleting failed endpoint: {endpoint_name}")
    sm.delete_endpoint(EndpointName=endpoint_name)
    logger.info(f"Waiting for endpoint {endpoint_name} to be deleted...")
    sm.get_waiter("endpoint_deleted").wait(
        EndpointName=endpoint_name,
        WaiterConfig=WAITER_CONFIG,
    )
    create_endpoint(sm, endpoint_name=endpoint_name, config_name=config_name)


def deploy_endpoint(sm, *, endpoint_name: str, config_name: str) -> None:
    endpoint = describe_endpoint(sm, endpoint_name)

    if endpoint is None:
        create_endpoint(sm, endpoint_name=endpoint_name, config_name=config_name)
    else:
        status = endpoint["EndpointStatus"]
        if status == "Failed":
            recreate_failed_endpoint(
                sm,
                endpoint_name=endpoint_name,
                config_name=config_name,
                failure_reason=endpoint.get("FailureReason"),
            )
        else:
            update_existing_endpoint(
                sm,
                endpoint_name=endpoint_name,
                config_name=config_name,
                status=status,
            )

    logger.info("Waiting for endpoint to be InService (this takes ~5 min)...")
    sm.get_waiter("endpoint_in_service").wait(
        EndpointName=endpoint_name,
        WaiterConfig=WAITER_CONFIG,
    )
    logger.info(f"Endpoint {endpoint_name} is InService.")


def main() -> None:
    settings = load_settings()
    sm = boto3.client("sagemaker", region_name=settings["aws_region"])

    ensure_model(
        sm,
        model_name=settings["model_name"],
        execution_role=settings["execution_role"],
        model_package_arn=settings["model_package_arn"],
    )
    ensure_endpoint_config(
        sm,
        config_name=settings["config_name"],
        model_name=settings["model_name"],
        s3_bucket=settings["s3_bucket"],
    )
    deploy_endpoint(
        sm,
        endpoint_name=settings["endpoint_name"],
        config_name=settings["config_name"],
    )


if __name__ == "__main__":
    main()
