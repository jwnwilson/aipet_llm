# ECR Public repository — used for images that external services (e.g. RunPod)
# need to pull without authentication. ECR Public is global but its API endpoint
# lives in us-east-1; the provider region must match.
resource "aws_ecrpublic_repository" "this" {
  repository_name = var.repo_name
}

data "aws_iam_policy_document" "ecr_public_push" {
  # ECR Public requires GetAuthorizationToken + GetServiceBearerToken on "*"
  statement {
    effect = "Allow"
    actions = [
      "ecr-public:GetAuthorizationToken",
      "sts:GetServiceBearerToken",
    ]
    resources = ["*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "ecr-public:BatchCheckLayerAvailability",
      "ecr-public:CompleteLayerUpload",
      "ecr-public:InitiateLayerUpload",
      "ecr-public:PutImage",
      "ecr-public:UploadLayerPart",
    ]
    resources = [aws_ecrpublic_repository.this.arn]
  }
}

resource "aws_iam_policy" "ecr_public_push" {
  name   = "${var.repo_name}-ecr-public-push"
  policy = data.aws_iam_policy_document.ecr_public_push.json
}
