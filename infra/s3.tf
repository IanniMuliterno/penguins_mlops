resource "aws_s3_bucket" "s3_bkt" {
  bucket = "penguins-classifier-dev-mlops"

  tags = var.tags
}