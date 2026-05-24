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
  value = aws_ecr_repository.inference.name
}

output "erc_repository_arn" {
  value = aws_ecr_repository.inference.arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.inference.repository_url
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions_deploy.arn
}

output "model_package_group_name" {
  value = aws_sagemaker_model_package_group.registry.model_package_group_name
}

output "model_package_group_arn" {
  value = aws_sagemaker_model_package_group.registry.arn
}