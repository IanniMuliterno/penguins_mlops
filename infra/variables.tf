variable "tags" {
  type = map(string)
  default = {
    environment = "production"
    project     = "penguins classifier"
  }
}

variable "assume_role_arn" {
  type = string
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type = string
  default = "penguins classifier"
}

variable "environment" {
  type = string
  default = "dev"
  
}

variable "ecr_repository_name" {
  type = string
  default = "penguins_ecr"
}

variable "model_package_group_name" {
  type = string
  default = "penguins_model_pkg_group"
}

variable "github_repository" {
  type = string
  default = "IanMuliterno/penguins_mlops"
}