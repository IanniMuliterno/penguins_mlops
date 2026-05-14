  output "bucket_name" {
    value = aws_s3_bucket.s3_bkt.bucket
  }

  output "execution_role_arn" {
    value = aws_iam_role.sagemaker_execution.arn
  }

  output "aws_region" {
    value = var.aws_region
  }

  output "ecr_repository_name" {
    value = var.ecr_repository_name
  }

  output "ecr_repository_url" {
    value ="${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_repository_name}"
  }

  output "github_actions_role_arn" {
    value = aws_iam_role.github_actions_deploy.arn
  }

  output "model_package_group_name" {
    value = var.model_package_group_name
  }