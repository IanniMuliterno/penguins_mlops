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
