# Terraform outputs for federated learning AWS deployment

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.federated_learning_vpc.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.federated_learning_vpc.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public_subnets[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private_subnets[*].id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.federated_learning_igw.id
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = var.enable_nat_gateway ? aws_nat_gateway.nat_gw[*].id : []
}

# Load Balancer Outputs
output "coordinator_alb_dns_name" {
  description = "DNS name of the coordinator Application Load Balancer"
  value       = aws_lb.coordinator_alb.dns_name
}

output "coordinator_alb_zone_id" {
  description = "Zone ID of the coordinator Application Load Balancer"
  value       = aws_lb.coordinator_alb.zone_id
}

output "coordinator_alb_arn" {
  description = "ARN of the coordinator Application Load Balancer"
  value       = aws_lb.coordinator_alb.arn
}

output "coordinator_target_group_arn" {
  description = "ARN of the coordinator target group"
  value       = aws_lb_target_group.coordinator_tg.arn
}

# Database Outputs
output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = var.enable_rds ? aws_db_instance.federated_learning_db[0].endpoint : null
  sensitive   = true
}

output "database_port" {
  description = "RDS instance port"
  value       = var.enable_rds ? aws_db_instance.federated_learning_db[0].port : null
}

output "database_name" {
  description = "Database name"
  value       = var.db_name
}

output "database_username" {
  description = "Database username"
  value       = var.db_username
  sensitive   = true
}

# Auto Scaling Group Outputs
output "coordinator_asg_name" {
  description = "Name of the coordinator Auto Scaling Group"
  value       = aws_autoscaling_group.coordinator_asg.name
}

output "coordinator_asg_arn" {
  description = "ARN of the coordinator Auto Scaling Group"
  value       = aws_autoscaling_group.coordinator_asg.arn
}

output "client_asg_name" {
  description = "Name of the client Auto Scaling Group"
  value       = aws_autoscaling_group.client_asg.name
}

output "client_asg_arn" {
  description = "ARN of the client Auto Scaling Group"
  value       = aws_autoscaling_group.client_asg.arn
}

# Security Group Outputs
output "coordinator_security_group_id" {
  description = "ID of the coordinator security group"
  value       = aws_security_group.coordinator_sg.id
}

output "client_security_group_id" {
  description = "ID of the client security group"
  value       = aws_security_group.client_sg.id
}

output "database_security_group_id" {
  description = "ID of the database security group"
  value       = aws_security_group.database_sg.id
}

# IAM Outputs
output "coordinator_iam_role_arn" {
  description = "ARN of the coordinator IAM role"
  value       = aws_iam_role.coordinator_role.arn
}

output "client_iam_role_arn" {
  description = "ARN of the client IAM role"
  value       = aws_iam_role.client_role.arn
}

output "coordinator_instance_profile_name" {
  description = "Name of the coordinator instance profile"
  value       = aws_iam_instance_profile.coordinator_profile.name
}

output "client_instance_profile_name" {
  description = "Name of the client instance profile"
  value       = aws_iam_instance_profile.client_profile.name
}

# S3 Outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket for model storage"
  value       = aws_s3_bucket.federated_learning_bucket.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for model storage"
  value       = aws_s3_bucket.federated_learning_bucket.arn
}

output "s3_bucket_domain_name" {
  description = "Domain name of the S3 bucket"
  value       = aws_s3_bucket.federated_learning_bucket.bucket_domain_name
}

# CloudWatch Outputs
output "coordinator_log_group_name" {
  description = "Name of the coordinator CloudWatch log group"
  value       = aws_cloudwatch_log_group.coordinator_logs.name
}

output "client_log_group_name" {
  description = "Name of the client CloudWatch log group"
  value       = aws_cloudwatch_log_group.client_logs.name
}

# Launch Template Outputs
output "coordinator_launch_template_id" {
  description = "ID of the coordinator launch template"
  value       = aws_launch_template.coordinator_lt.id
}

output "client_launch_template_id" {
  description = "ID of the client launch template"
  value       = aws_launch_template.client_lt.id
}

# Auto Scaling Policy Outputs
output "coordinator_scale_up_policy_arn" {
  description = "ARN of the coordinator scale up policy"
  value       = aws_autoscaling_policy.coordinator_scale_up.arn
}

output "coordinator_scale_down_policy_arn" {
  description = "ARN of the coordinator scale down policy"
  value       = aws_autoscaling_policy.coordinator_scale_down.arn
}

output "client_scale_up_policy_arn" {
  description = "ARN of the client scale up policy"
  value       = aws_autoscaling_policy.client_scale_up.arn
}

output "client_scale_down_policy_arn" {
  description = "ARN of the client scale down policy"
  value       = aws_autoscaling_policy.client_scale_down.arn
}

# CloudWatch Alarm Outputs
output "coordinator_cpu_high_alarm_name" {
  description = "Name of the coordinator high CPU alarm"
  value       = aws_cloudwatch_metric_alarm.coordinator_cpu_high.alarm_name
}

output "coordinator_cpu_low_alarm_name" {
  description = "Name of the coordinator low CPU alarm"
  value       = aws_cloudwatch_metric_alarm.coordinator_cpu_low.alarm_name
}

output "client_cpu_high_alarm_name" {
  description = "Name of the client high CPU alarm"
  value       = aws_cloudwatch_metric_alarm.client_cpu_high.alarm_name
}

output "client_cpu_low_alarm_name" {
  description = "Name of the client low CPU alarm"
  value       = aws_cloudwatch_metric_alarm.client_cpu_low.alarm_name
}

# Application Configuration Outputs
output "coordinator_endpoint" {
  description = "Coordinator endpoint for clients to connect"
  value       = "http://${aws_lb.coordinator_alb.dns_name}"
}

output "grpc_endpoint" {
  description = "gRPC endpoint for federated learning communication"
  value       = "${aws_lb.coordinator_alb.dns_name}:50051"
}

output "rest_api_endpoint" {
  description = "REST API endpoint for monitoring and management"
  value       = "http://${aws_lb.coordinator_alb.dns_name}/api"
}

# Environment Information
output "deployment_region" {
  description = "AWS region where resources are deployed"
  value       = var.aws_region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "project_name" {
  description = "Project name"
  value       = var.project_name
}

# Connection Information
output "database_connection_string" {
  description = "Database connection string (without password)"
  value = var.enable_rds ? "postgresql://${var.db_username}@${aws_db_instance.federated_learning_db[0].endpoint}:${aws_db_instance.federated_learning_db[0].port}/${var.db_name}" : null
  sensitive = true
}

# Monitoring URLs
output "cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:"
}

output "ec2_console_url" {
  description = "EC2 console URL for managing instances"
  value = "https://${var.aws_region}.console.aws.amazon.com/ec2/v2/home?region=${var.aws_region}#Instances:"
}

output "rds_console_url" {
  description = "RDS console URL for managing database"
  value = var.enable_rds ? "https://${var.aws_region}.console.aws.amazon.com/rds/home?region=${var.aws_region}#database:id=${aws_db_instance.federated_learning_db[0].id}" : null
}

# Cost Information
output "estimated_monthly_cost" {
  description = "Estimated monthly cost (approximate)"
  value = {
    coordinator_instances = "~$${var.coordinator_desired_capacity * 25}"  # Rough estimate for t3.medium
    client_instances     = "~$${var.client_desired_capacity * 15}"       # Rough estimate for t3.small
    database            = var.enable_rds ? "~$15" : "$0"                 # Rough estimate for db.t3.micro
    load_balancer       = "~$20"
    nat_gateway         = var.enable_nat_gateway ? "~$45" : "$0"
    storage             = "~$10"
    total_estimate      = "~$${var.coordinator_desired_capacity * 25 + var.client_desired_capacity * 15 + (var.enable_rds ? 15 : 0) + 20 + (var.enable_nat_gateway ? 45 : 0) + 10}"
  }
}

# Deployment Instructions
output "deployment_instructions" {
  description = "Instructions for accessing the deployed system"
  value = {
    coordinator_url = "http://${aws_lb.coordinator_alb.dns_name}"
    health_check   = "curl http://${aws_lb.coordinator_alb.dns_name}/health"
    ssh_command    = var.key_pair_name != "" ? "Use your key pair '${var.key_pair_name}' to SSH into instances" : "Configure a key pair to enable SSH access"
    logs_command   = "aws logs tail /aws/ec2/${var.project_name}/coordinator --follow --region ${var.aws_region}"
    scale_up       = "aws autoscaling set-desired-capacity --auto-scaling-group-name ${aws_autoscaling_group.client_asg.name} --desired-capacity <number> --region ${var.aws_region}"
  }
}