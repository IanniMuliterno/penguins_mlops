resource "aws_s3_bucket" "s3_bkt" {
  bucket = "penguins_bucket"

  tags = var.tags
}