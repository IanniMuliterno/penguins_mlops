import os
import boto3
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
import sagemaker

from src import logger

endpoint_name = os.environ["ENDPOINT_NAME"]
execution_role = os.environ["EXECUTION_ROLE"]
s3_bucket = os.environ["S3_BUCKET"]
aws_region = os.environ["AWS_REGION"]
baseline_data_uri = os.environ["BASELINE_DATA_URI"]
model_version = os.environ["MODEL_VERSION"]

baseline_results_uri = f"s3://{s3_bucket}/monitoring/baseline-results/{model_version}"
monitoring_output_uri = f"s3://{s3_bucket}/monitoring/reports/{model_version}"
schedule_name = f"penguin-monitor-{endpoint_name}"

boto_session = boto3.Session(region_name=aws_region)
sm_session = sagemaker.Session(boto_session=boto_session)
sm_client = boto_session.client("sagemaker")

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
    baseline_dataset=baseline_data_uri,
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
