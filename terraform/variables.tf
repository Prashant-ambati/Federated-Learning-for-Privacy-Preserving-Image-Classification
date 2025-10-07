# Terraform variables for federated learning AWS deployment

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "federated-learning"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

# Security Configuration
variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "ssh_allowed_cidr_blocks" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "key_pair_name" {
  description = "Name of the AWS key pair for EC2 instances"
  type        = string
  default     = ""
}

# Coordinator Configuration
variable "coordinator_instance_type" {
  description = "Instance type for coordinator servers"
  type        = string
  default     = "t3.medium"
}

variable "coordinator_min_size" {
  description = "Minimum number of coordinator instances"
  type        = number
  default     = 1
}

variable "coordinator_max_size" {
  description = "Maximum number of coordinator instances"
  type        = number
  default     = 3
}

variable "coordinator_desired_capacity" {
  description = "Desired number of coordinator instances"
  type        = number
  default     = 1
}

# Client Configuration
variable "client_instance_type" {
  description = "Instance type for client servers"
  type        = string
  default     = "t3.small"
}

variable "client_min_size" {
  description = "Minimum number of client instances"
  type        = number
  default     = 2
}

variable "client_max_size" {
  description = "Maximum number of client instances"
  type        = number
  default     = 20
}

variable "client_desired_capacity" {
  description = "Desired number of client instances"
  type        = number
  default     = 5
}

# Database Configuration
variable "enable_rds" {
  description = "Enable RDS PostgreSQL database"
  type        = bool
  default     = true
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "13.7"
}

variable "db_allocated_storage" {
  description = "Initial allocated storage for RDS (GB)"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS (GB)"
  type        = number
  default     = 100
}

variable "db_name" {
  description = "Name of the database"
  type        = string
  default     = "federated_learning"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "federated_user"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
  default     = "change_me_in_production"
}

variable "db_backup_retention_period" {
  description = "Database backup retention period (days)"
  type        = number
  default     = 7
}

variable "db_backup_window" {
  description = "Database backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "db_maintenance_window" {
  description = "Database maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "db_skip_final_snapshot" {
  description = "Skip final snapshot when deleting database"
  type        = bool
  default     = true
}

# Monitoring and Logging
variable "log_retention_days" {
  description = "CloudWatch log retention period (days)"
  type        = number
  default     = 14
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for resources"
  type        = bool
  default     = false
}

# Auto Scaling Configuration
variable "scale_up_threshold" {
  description = "CPU threshold for scaling up (%)"
  type        = number
  default     = 70
}

variable "scale_down_threshold" {
  description = "CPU threshold for scaling down (%)"
  type        = number
  default     = 20
}

variable "scale_up_cooldown" {
  description = "Cooldown period for scaling up (seconds)"
  type        = number
  default     = 300
}

variable "scale_down_cooldown" {
  description = "Cooldown period for scaling down (seconds)"
  type        = number
  default     = 300
}

# Application Configuration
variable "federated_learning_config" {
  description = "Federated learning application configuration"
  type = object({
    min_clients      = number
    max_clients      = number
    local_epochs     = number
    batch_size       = number
    learning_rate    = number
    total_rounds     = number
    round_timeout    = number
    model_type       = string
    privacy_epsilon  = number
    privacy_delta    = number
  })
  default = {
    min_clients      = 3
    max_clients      = 50
    local_epochs     = 5
    batch_size       = 32
    learning_rate    = 0.001
    total_rounds     = 100
    round_timeout    = 300
    model_type       = "simple_cnn"
    privacy_epsilon  = 1.0
    privacy_delta    = 1e-5
  }
}

# Notification Configuration
variable "notification_email" {
  description = "Email address for notifications"
  type        = string
  default     = ""
}

variable "enable_sns_notifications" {
  description = "Enable SNS notifications for alerts"
  type        = bool
  default     = false
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Use spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_types" {
  description = "Instance types for spot instances"
  type        = list(string)
  default     = ["t3.small", "t3.medium", "t3a.small", "t3a.medium"]
}

# Backup and Recovery
variable "enable_automated_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_schedule" {
  description = "Cron expression for backup schedule"
  type        = string
  default     = "cron(0 2 * * ? *)"  # Daily at 2 AM
}

# Performance Configuration
variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS"
  type        = bool
  default     = true
}

variable "monitoring_interval" {
  description = "Enhanced monitoring interval (seconds)"
  type        = number
  default     = 60
}

# Network Configuration
variable "enable_vpc_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = false
}

variable "flow_log_retention_days" {
  description = "VPC flow log retention period (days)"
  type        = number
  default     = 14
}

# Disaster Recovery
variable "enable_multi_az" {
  description = "Enable Multi-AZ deployment for RDS"
  type        = bool
  default     = false
}

variable "cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Region for cross-region backups"
  type        = string
  default     = "us-east-1"
}

# Development and Testing
variable "enable_debug_mode" {
  description = "Enable debug mode for applications"
  type        = bool
  default     = false
}

variable "enable_test_data" {
  description = "Load test data on startup"
  type        = bool
  default     = false
}

# Compliance and Security
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all storage"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

variable "compliance_mode" {
  description = "Compliance mode (none, hipaa, pci, sox)"
  type        = string
  default     = "none"
}

# Resource Tagging
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = ""
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = ""
}