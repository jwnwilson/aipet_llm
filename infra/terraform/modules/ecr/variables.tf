variable "repo_name" {
  description = "ECR repository name"
  type        = string
}

variable "image_retention_count" {
  description = "Number of tagged images to retain before expiring older ones"
  type        = number
  default     = 10
}

variable "untagged_retention_days" {
  description = "Days to retain untagged images before expiring them"
  type        = number
  default     = 14
}
