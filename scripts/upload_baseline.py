import os
from pathlib import Path

import boto3

from src.data_loader import load_penguins_frame
from src import logger


def main() -> None:
    bucket = os.environ["S3_BUCKET"]
    model_version = os.environ["MODEL_VERSION"]
    region = os.environ["AWS_REGION"]

    baseline_path = Path("/tmp/baseline.csv")
    baseline = load_penguins_frame().dropna().drop(columns=["species"])
    baseline.to_csv(baseline_path, index=False)

    key = f"monitoring/baselines/{model_version}/baseline.csv"
    boto3.client("s3", region_name=region).upload_file(str(baseline_path), bucket, key)
    logger.info("Uploaded monitoring baseline to s3://%s/%s", bucket, key)


if __name__ == "__main__":
    main()
