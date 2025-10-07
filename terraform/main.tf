# Terraform configuration for AWS deployment of federated learning system
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "federated-learning"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# VPC Configuration
resource "aws_vpc" "federated_learning_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "federated_learning_igw" {
  vpc_id = aws_vpc.federated_learning_vpc.id
  
  tags = {
    Name = "${var.project_name}-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public_subnets" {
  count = length(var.public_subnet_cidrs)
  
  vpc_id                  = aws_vpc.federated_learning_vpc.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
    Type = "Public"
  }
}

# Private Subnets
resource "aws_subnet" "private_subnets" {
  count = length(var.private_subnet_cidrs)
  
  vpc_id            = aws_vpc.federated_learning_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
    Type = "Private"
  }
}

# Route Table for Public Subnets
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.federated_learning_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.federated_learning_igw.id
  }
  
  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

# Route Table Associations for Public Subnets
resource "aws_route_table_association" "public_rta" {
  count = length(aws_subnet.public_subnets)
  
  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

# NAT Gateway for Private Subnets
resource "aws_eip" "nat_eip" {
  count  = var.enable_nat_gateway ? length(aws_subnet.public_subnets) : 0
  domain = "vpc"
  
  tags = {
    Name = "${var.project_name}-nat-eip-${count.index + 1}"
  }
  
  depends_on = [aws_internet_gateway.federated_learning_igw]
}

resource "aws_nat_gateway" "nat_gw" {
  count = var.enable_nat_gateway ? length(aws_subnet.public_subnets) : 0
  
  allocation_id = aws_eip.nat_eip[count.index].id
  subnet_id     = aws_subnet.public_subnets[count.index].id
  
  tags = {
    Name = "${var.project_name}-nat-gw-${count.index + 1}"
  }
  
  depends_on = [aws_internet_gateway.federated_learning_igw]
}

# Route Table for Private Subnets
resource "aws_route_table" "private_rt" {
  count = var.enable_nat_gateway ? length(aws_subnet.private_subnets) : 0
  
  vpc_id = aws_vpc.federated_learning_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gw[count.index].id
  }
  
  tags = {
    Name = "${var.project_name}-private-rt-${count.index + 1}"
  }
}

# Route Table Associations for Private Subnets
resource "aws_route_table_association" "private_rta" {
  count = var.enable_nat_gateway ? length(aws_subnet.private_subnets) : 0
  
  subnet_id      = aws_subnet.private_subnets[count.index].id
  route_table_id = aws_route_table.private_rt[count.index].id
}

# Security Groups
resource "aws_security_group" "coordinator_sg" {
  name_prefix = "${var.project_name}-coordinator-"
  vpc_id      = aws_vpc.federated_learning_vpc.id
  description = "Security group for federated learning coordinator"
  
  # gRPC port
  ingress {
    from_port   = 50051
    to_port     = 50051
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "gRPC communication"
  }
  
  # REST API port
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
    description = "REST API"
  }
  
  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_cidr_blocks
    description = "SSH access"
  }
  
  # Health check port
  ingress {
    from_port   = 8081
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Health check"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }
  
  tags = {
    Name = "${var.project_name}-coordinator-sg"
  }
}

resource "aws_security_group" "client_sg" {
  name_prefix = "${var.project_name}-client-"
  vpc_id      = aws_vpc.federated_learning_vpc.id
  description = "Security group for federated learning clients"
  
  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_cidr_blocks
    description = "SSH access"
  }
  
  # Health check port
  ingress {
    from_port   = 8082
    to_port     = 8082
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Health check"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }
  
  tags = {
    Name = "${var.project_name}-client-sg"
  }
}

resource "aws_security_group" "database_sg" {
  name_prefix = "${var.project_name}-database-"
  vpc_id      = aws_vpc.federated_learning_vpc.id
  description = "Security group for database"
  
  # PostgreSQL port
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.coordinator_sg.id]
    description     = "PostgreSQL access from coordinator"
  }
  
  tags = {
    Name = "${var.project_name}-database-sg"
  }
}

# Application Load Balancer
resource "aws_lb" "coordinator_alb" {
  name               = "${var.project_name}-coordinator-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.coordinator_sg.id]
  subnets            = aws_subnet.public_subnets[*].id
  
  enable_deletion_protection = var.enable_deletion_protection
  
  tags = {
    Name = "${var.project_name}-coordinator-alb"
  }
}

# Target Group for Coordinator
resource "aws_lb_target_group" "coordinator_tg" {
  name     = "${var.project_name}-coordinator-tg"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.federated_learning_vpc.id
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
    port                = "8081"
    protocol            = "HTTP"
  }
  
  tags = {
    Name = "${var.project_name}-coordinator-tg"
  }
}

# ALB Listener
resource "aws_lb_listener" "coordinator_listener" {
  load_balancer_arn = aws_lb.coordinator_alb.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.coordinator_tg.arn
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "federated_learning_db_subnet_group" {
  count = var.enable_rds ? 1 : 0
  
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = aws_subnet.private_subnets[*].id
  
  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

# RDS Instance
resource "aws_db_instance" "federated_learning_db" {
  count = var.enable_rds ? 1 : 0
  
  identifier = "${var.project_name}-database"
  
  engine         = "postgres"
  engine_version = var.db_engine_version
  instance_class = var.db_instance_class
  
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp2"
  storage_encrypted     = true
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.database_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.federated_learning_db_subnet_group[0].name
  
  backup_retention_period = var.db_backup_retention_period
  backup_window          = var.db_backup_window
  maintenance_window     = var.db_maintenance_window
  
  skip_final_snapshot = var.db_skip_final_snapshot
  deletion_protection = var.enable_deletion_protection
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_enhanced_monitoring[0].arn
  
  tags = {
    Name = "${var.project_name}-database"
  }
}

# IAM Role for RDS Enhanced Monitoring
resource "aws_iam_role" "rds_enhanced_monitoring" {
  count = var.enable_rds ? 1 : 0
  
  name = "${var.project_name}-rds-enhanced-monitoring"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  count = var.enable_rds ? 1 : 0
  
  role       = aws_iam_role.rds_enhanced_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Launch Template for Coordinator
resource "aws_launch_template" "coordinator_lt" {
  name_prefix   = "${var.project_name}-coordinator-"
  image_id      = data.aws_ami.amazon_linux.id
  instance_type = var.coordinator_instance_type
  key_name      = var.key_pair_name
  
  vpc_security_group_ids = [aws_security_group.coordinator_sg.id]
  
  iam_instance_profile {
    name = aws_iam_instance_profile.coordinator_profile.name
  }
  
  user_data = base64encode(templatefile("${path.module}/user_data/coordinator_user_data.sh", {
    db_endpoint = var.enable_rds ? aws_db_instance.federated_learning_db[0].endpoint : ""
    db_name     = var.db_name
    db_username = var.db_username
    db_password = var.db_password
    environment = var.environment
  }))
  
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.project_name}-coordinator"
      Type = "coordinator"
    }
  }
}

# Auto Scaling Group for Coordinator
resource "aws_autoscaling_group" "coordinator_asg" {
  name                = "${var.project_name}-coordinator-asg"
  vpc_zone_identifier = aws_subnet.private_subnets[*].id
  target_group_arns   = [aws_lb_target_group.coordinator_tg.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300
  
  min_size         = var.coordinator_min_size
  max_size         = var.coordinator_max_size
  desired_capacity = var.coordinator_desired_capacity
  
  launch_template {
    id      = aws_launch_template.coordinator_lt.id
    version = "$Latest"
  }
  
  tag {
    key                 = "Name"
    value               = "${var.project_name}-coordinator-asg"
    propagate_at_launch = false
  }
}

# Launch Template for Clients
resource "aws_launch_template" "client_lt" {
  name_prefix   = "${var.project_name}-client-"
  image_id      = data.aws_ami.amazon_linux.id
  instance_type = var.client_instance_type
  key_name      = var.key_pair_name
  
  vpc_security_group_ids = [aws_security_group.client_sg.id]
  
  iam_instance_profile {
    name = aws_iam_instance_profile.client_profile.name
  }
  
  user_data = base64encode(templatefile("${path.module}/user_data/client_user_data.sh", {
    coordinator_endpoint = aws_lb.coordinator_alb.dns_name
    environment         = var.environment
  }))
  
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.project_name}-client"
      Type = "client"
    }
  }
}

# Auto Scaling Group for Clients
resource "aws_autoscaling_group" "client_asg" {
  name                = "${var.project_name}-client-asg"
  vpc_zone_identifier = aws_subnet.private_subnets[*].id
  
  min_size         = var.client_min_size
  max_size         = var.client_max_size
  desired_capacity = var.client_desired_capacity
  
  launch_template {
    id      = aws_launch_template.client_lt.id
    version = "$Latest"
  }
  
  tag {
    key                 = "Name"
    value               = "${var.project_name}-client-asg"
    propagate_at_launch = false
  }
}

# IAM Roles and Policies
resource "aws_iam_role" "coordinator_role" {
  name = "${var.project_name}-coordinator-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role" "client_role" {
  name = "${var.project_name}-client-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policies
resource "aws_iam_policy" "coordinator_policy" {
  name = "${var.project_name}-coordinator-policy"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:DescribeLogGroups"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.federated_learning_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.federated_learning_bucket.arn
        ]
      }
    ]
  })
}

resource "aws_iam_policy" "client_policy" {
  name = "${var.project_name}-client-policy"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach policies to roles
resource "aws_iam_role_policy_attachment" "coordinator_policy_attachment" {
  role       = aws_iam_role.coordinator_role.name
  policy_arn = aws_iam_policy.coordinator_policy.arn
}

resource "aws_iam_role_policy_attachment" "client_policy_attachment" {
  role       = aws_iam_role.client_role.name
  policy_arn = aws_iam_policy.client_policy.arn
}

# Instance Profiles
resource "aws_iam_instance_profile" "coordinator_profile" {
  name = "${var.project_name}-coordinator-profile"
  role = aws_iam_role.coordinator_role.name
}

resource "aws_iam_instance_profile" "client_profile" {
  name = "${var.project_name}-client-profile"
  role = aws_iam_role.client_role.name
}

# S3 Bucket for model storage
resource "aws_s3_bucket" "federated_learning_bucket" {
  bucket = "${var.project_name}-models-${random_string.bucket_suffix.result}"
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "federated_learning_bucket_versioning" {
  bucket = aws_s3_bucket.federated_learning_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "federated_learning_bucket_encryption" {
  bucket = aws_s3_bucket.federated_learning_bucket.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "federated_learning_bucket_pab" {
  bucket = aws_s3_bucket.federated_learning_bucket.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "coordinator_logs" {
  name              = "/aws/ec2/${var.project_name}/coordinator"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "client_logs" {
  name              = "/aws/ec2/${var.project_name}/client"
  retention_in_days = var.log_retention_days
}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "coordinator_scale_up" {
  name                   = "${var.project_name}-coordinator-scale-up"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = aws_autoscaling_group.coordinator_asg.name
}

resource "aws_autoscaling_policy" "coordinator_scale_down" {
  name                   = "${var.project_name}-coordinator-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = aws_autoscaling_group.coordinator_asg.name
}

resource "aws_autoscaling_policy" "client_scale_up" {
  name                   = "${var.project_name}-client-scale-up"
  scaling_adjustment     = 2
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = aws_autoscaling_group.client_asg.name
}

resource "aws_autoscaling_policy" "client_scale_down" {
  name                   = "${var.project_name}-client-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = aws_autoscaling_group.client_asg.name
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "coordinator_cpu_high" {
  alarm_name          = "${var.project_name}-coordinator-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors coordinator cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.coordinator_scale_up.arn]
  
  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.coordinator_asg.name
  }
}

resource "aws_cloudwatch_metric_alarm" "coordinator_cpu_low" {
  alarm_name          = "${var.project_name}-coordinator-cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "10"
  alarm_description   = "This metric monitors coordinator cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.coordinator_scale_down.arn]
  
  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.coordinator_asg.name
  }
}

resource "aws_cloudwatch_metric_alarm" "client_cpu_high" {
  alarm_name          = "${var.project_name}-client-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "70"
  alarm_description   = "This metric monitors client cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.client_scale_up.arn]
  
  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.client_asg.name
  }
}

resource "aws_cloudwatch_metric_alarm" "client_cpu_low" {
  alarm_name          = "${var.project_name}-client-cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "10"
  alarm_description   = "This metric monitors client cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.client_scale_down.arn]
  
  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.client_asg.name
  }
}