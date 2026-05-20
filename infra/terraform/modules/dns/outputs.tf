output "fqdn" {
  description = "FQDN of the llm-api LLM API DNS record"
  value       = aws_route53_record.llm_api.fqdn
}

output "ui_fqdn" {
  description = "FQDN of the llm-ui DNS record"
  value       = length(aws_route53_record.llm_ui) > 0 ? aws_route53_record.llm_ui[0].fqdn : ""
}
