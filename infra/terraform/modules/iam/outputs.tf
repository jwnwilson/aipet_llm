output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC"
  value       = aws_iam_role.github_actions.arn
}

output "llm_api_aws_access_key_id" {
  description = "Access key ID for the llm-api app IAM user"
  value       = aws_iam_access_key.llm_api.id
}

output "llm_api_aws_secret_access_key" {
  description = "Secret access key for the llm-api app IAM user"
  value       = aws_iam_access_key.llm_api.secret
  sensitive   = true
}

output "kaggle_training_aws_access_key_id" {
  description = "Access key ID for the Kaggle training IAM user (S3 only)"
  value       = aws_iam_access_key.kaggle_training.id
}

output "kaggle_training_aws_secret_access_key" {
  description = "Secret access key for the Kaggle training IAM user (S3 only)"
  value       = aws_iam_access_key.kaggle_training.secret
  sensitive   = true
}
