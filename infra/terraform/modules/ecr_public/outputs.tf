output "repository_uri" {
  description = "Full ECR Public repository URI (public.ecr.aws/{alias}/{name})"
  value       = aws_ecrpublic_repository.this.repository_uri
}

output "repository_arn" {
  description = "ECR Public repository ARN"
  value       = aws_ecrpublic_repository.this.arn
}

output "ecr_public_push_policy_arn" {
  description = "IAM policy ARN granting ECR Public push"
  value       = aws_iam_policy.ecr_public_push.arn
}
