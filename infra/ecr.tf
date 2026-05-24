resource "aws_ecr_repository" "inference" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = var.tags
}

resource "aws_sagemaker_model_package_group" "registry" {
  model_package_group_name        = var.model_package_group_name
  model_package_group_description = "Model registry for ${var.project_name} (${var.environment})"

  tags = var.tags
}