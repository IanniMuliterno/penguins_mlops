import os
import boto3
import pandas as pd
from palmerpenguins import load_penguins
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
import sagemaker

from src import logger

endpoint_name = os.environ["ENDPOINT_NAME"]
execution_role = os.environ["EXECUTION_ROLE"]
s3_bucket = os.environ["S3_BUCKET"]
aws_region = os.environ["AWS_REGION"]

baseline_data_uri = f"s3://{s3_bucket}/monitoring/baseline-data"
baseline_results_uri = f"s3://{s3_bucket}/monitoring/baseline-results"
monitoring_output_uri = f"s3://{s3_bucket}/monitoring/reports"
schedule_name = f"penguin-monitor-{endpoint_name}"

boto_session = boto3.Session(region_name=aws_region)
sm_session = sagemaker.Session(boto_session=boto_session)
sm_client = boto_session.client("sagemaker")

# Upload training features as baseline dataset (first time only)
baseline_local = "/tmp/baseline.csv"
penguins = load_penguins().dropna()
penguins.drop(columns=["species"]).to_csv(baseline_local, index=False)
sm_session.upload_data(path=baseline_local, bucket=s3_bucket, key_prefix="monitoring/baseline-data")
logger.info(f"Baseline data uploaded to {baseline_data_uri}")

monitor = DefaultModelMonitor(
    role=execution_role,
    instance_count=1,
    instance_type="ml.t3.medium",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sm_session,
)

logger.info("Running baseline suggestion job...")
monitor.suggest_baseline(
    baseline_dataset=f"{baseline_data_uri}/baseline.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_results_uri,
    wait=True,
)
logger.info(f"Baseline results at {baseline_results_uri}")

# Delete existing schedule before recreating (update path is complex)
try:
    sm_client.describe_monitoring_schedule(MonitoringScheduleName=schedule_name)
    sm_client.delete_monitoring_schedule(MonitoringScheduleName=schedule_name)
    logger.info(f"Deleted existing schedule: {schedule_name}")
except sm_client.exceptions.ResourceNotFound:
    pass

monitor.create_monitoring_schedule(
    monitor_schedule_name=schedule_name,
    endpoint_input=endpoint_name,
    output_s3_uri=monitoring_output_uri,
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_constraints(),
    schedule_cron_expression="cron(0 * ? * * *)",
)
logger.info(f"Monitoring schedule '{schedule_name}' active (runs hourly).")
