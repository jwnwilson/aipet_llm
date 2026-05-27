variable "aws_region" {
  description = "AWS region for the ECR repository"
  type        = string
  default     = "us-east-1"
}

variable "repo_name" {
  description = "ECR repository name"
  type        = string
  default     = "llm-api"
}

variable "image_retention_count" {
  description = "Number of tagged images to retain before expiring older ones"
  type        = number
  default     = 10
}

variable "github_repo" {
  description = "GitHub repository in owner/name format — scopes the OIDC trust to main-branch pushes (e.g. myorg/llm-api)"
  type        = string
}

variable "s3_bucket" {
  description = "S3 bucket name used to store models — grants the GitHub Actions role read access"
  type        = string
  default     = "aipet-jwn"
}

variable "vps_ip" {
  description = "Fallback public IP for the llm-api DNS A record — overridden by inlets_exit_node reserved IP when the module is applied"
  type        = string
  default     = "178.62.70.159"
}

variable "do_token" {
  description = "DigitalOcean personal access token — set via TF_VAR_do_token or a .tfvars file (never commit the value)"
  type        = string
  sensitive   = true
}

variable "inlets_token" {
  description = "Auth token for the inlets-pro tunnel (shared between the DO exit node and the k8s client Secret)"
  type        = string
  sensitive   = true
}

variable "inlets_license" {
  description = "inlets-pro license key (written to the DO exit node and mounted in the k8s client pod)"
  type        = string
  sensitive   = true
}
