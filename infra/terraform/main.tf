provider "aws" {
  region = var.aws_region
}

provider "digitalocean" {
  token = var.do_token
}

module "inlets_exit_node" {
  source         = "./modules/inlets_exit_node"
  inlets_token   = var.inlets_token
  inlets_license = var.inlets_license
}

module "ecr" {
  source                = "./modules/ecr"
  repo_name             = var.repo_name
  image_retention_count = var.image_retention_count
}

module "ecr_temporal_ui" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-temporal-ui"
  image_retention_count = var.image_retention_count
}

module "ecr_proxy" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-proxy"
  image_retention_count = var.image_retention_count
}

module "ecr_inference" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-inference"
  image_retention_count = var.image_retention_count
}

module "ecr_training" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-training"
  image_retention_count = var.image_retention_count
}

module "ecr_export" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-export"
  image_retention_count = var.image_retention_count
}

module "acm_ui" {
  source = "./modules/acm"
  domain = "llm.jwnwilson.co.uk"
}

module "s3_ui" {
  source              = "./modules/s3_static"
  name                = "llm-api-ui"
  domain              = "llm.jwnwilson.co.uk"
  acm_certificate_arn = module.acm_ui.certificate_arn
}

module "iam" {
  source                     = "./modules/iam"
  repo_name                  = var.repo_name
  github_repo                = var.github_repo
  s3_bucket                  = var.s3_bucket
  ecr_push_policy_arn        = module.ecr.ecr_push_policy_arn
  extra_ecr_push_policy_arns = [
    module.ecr_temporal_ui.ecr_push_policy_arn,
    module.ecr_proxy.ecr_push_policy_arn,
    module.ecr_inference.ecr_push_policy_arn,
    module.ecr_training.ecr_push_policy_arn,
    module.ecr_export.ecr_push_policy_arn,
  ]
  ecr_pull_repo_arns = [
    module.ecr.repository_arn,
    module.ecr_temporal_ui.repository_arn,
    module.ecr_proxy.repository_arn,
    module.ecr_inference.repository_arn,
    module.ecr_training.repository_arn,
    module.ecr_export.repository_arn,
  ]
  ui_bucket_arn              = module.s3_ui.bucket_arn
  ui_distribution_arn        = module.s3_ui.distribution_arn
  create_ui_resources        = true
}

module "dns" {
  source           = "./modules/dns"
  vps_ip           = module.inlets_exit_node.reserved_ip
  ui_cf_domain     = module.s3_ui.cloudfront_domain
  create_ui_record = true
}
