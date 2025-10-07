# Federated Learning AWS Deployment

This directory contains Terraform configuration for deploying the federated learning system on AWS.

## Architecture Overview

The deployment creates:

- **VPC** with public and private subnets across multiple AZs
- **Application Load Balancer** for coordinator access
- **Auto Scaling Groups** for coordinator and client instances
- **RDS PostgreSQL** database for persistence
- **S3 bucket** for model storage
- **CloudWatch** logging and monitoring
- **IAM roles** and security groups
- **Auto-scaling policies** and CloudWatch alarms

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.0 installed
3. **AWS Key Pair** created for EC2 access
4. **Domain name** (optional) for custom DNS

## Quick Start

1. **Clone and navigate to terraform directory**:
   ```bash
   cd terraform
   ```

2. **Copy and customize variables**:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your specific values
   ```

3. **Initialize Terraform**:
   ```bash
   terraform init
   ```

4. **Plan the deployment**:
   ```bash
   terraform plan
   ```

5. **Apply the configuration**:
   ```bash
   terraform apply
   ```

6. **Get deployment outputs**:
   ```bash
   terraform output
   ```

## Configuration

### Required Variables

Edit `terraform.tfvars` with these required values:

```hcl
# AWS Configuration
aws_region = "us-west-2"
key_pair_name = "your-key-pair-name"

# Database Configuration
db_password = "your-secure-password"

# Security Configuration
ssh_allowed_cidr_blocks = ["your.ip.address/32"]
```

### Important Security Settings

- **Database Password**: Change the default password in `terraform.tfvars`
- **SSH Access**: Restrict `ssh_allowed_cidr_blocks` to your IP ranges
- **API Access**: Limit `allowed_cidr_blocks` for production deployments

## Deployment Environments

### Development Environment

```hcl
environment = "dev"
coordinator_desired_capacity = 1
client_desired_capacity = 3
db_instance_class = "db.t3.micro"
enable_deletion_protection = false
```

### Production Environment

```hcl
environment = "prod"
coordinator_desired_capacity = 2
client_desired_capacity = 10
db_instance_class = "db.t3.small"
enable_deletion_protection = true
enable_multi_az = true
```

## Accessing the System

After deployment, you can access:

### Coordinator API
```bash
# Get the load balancer DNS name
COORDINATOR_URL=$(terraform output -raw coordinator_alb_dns_name)

# Health check
curl http://$COORDINATOR_URL/health

# API documentation
curl http://$COORDINATOR_URL/api/docs
```

### SSH Access
```bash
# Get instance IPs from AWS console or CLI
aws ec2 describe-instances --filters "Name=tag:Project,Values=federated-learning"

# SSH to coordinator
ssh -i your-key.pem ec2-user@<coordinator-ip>

# SSH to client
ssh -i your-key.pem ec2-user@<client-ip>
```

### Logs
```bash
# View coordinator logs
aws logs tail /aws/ec2/federated-learning/coordinator --follow

# View client logs
aws logs tail /aws/ec2/federated-learning/client --follow
```

## Monitoring

### CloudWatch Dashboards

The deployment creates custom metrics in CloudWatch:

- **FederatedLearning/Coordinator**: Coordinator-specific metrics
- **FederatedLearning/Client**: Client-specific metrics
- **AWS/EC2**: Standard EC2 metrics
- **AWS/ApplicationELB**: Load balancer metrics
- **AWS/RDS**: Database metrics

### Alarms and Auto Scaling

Auto scaling is configured based on:

- **CPU Utilization**: Scale up at 70%, scale down at 20%
- **Custom Metrics**: Training load, client connections
- **Health Checks**: Automatic replacement of unhealthy instances

## Cost Optimization

### Spot Instances

Enable spot instances for cost savings:

```hcl
enable_spot_instances = true
spot_instance_types = ["t3.small", "t3.medium", "t3a.small"]
```

### Resource Sizing

Adjust instance types based on workload:

```hcl
# For light workloads
coordinator_instance_type = "t3.small"
client_instance_type = "t3.micro"

# For heavy workloads
coordinator_instance_type = "c5.large"
client_instance_type = "c5.medium"
```

## Backup and Recovery

### Automated Backups

RDS automated backups are enabled by default:

```hcl
db_backup_retention_period = 7
db_backup_window = "03:00-04:00"
```

### Cross-Region Backup

Enable for disaster recovery:

```hcl
cross_region_backup = true
backup_region = "us-east-1"
```

## Security Best Practices

### Network Security

- Private subnets for application instances
- Security groups with minimal required access
- NAT gateways for outbound internet access
- VPC flow logs for network monitoring

### Data Security

- Encryption at rest for RDS and S3
- Encryption in transit for all communications
- IAM roles with least privilege access
- Secrets management for database credentials

### Access Control

```hcl
# Restrict SSH access
ssh_allowed_cidr_blocks = ["10.0.0.0/8"]

# Restrict API access
allowed_cidr_blocks = ["your.office.ip/32"]

# Enable deletion protection
enable_deletion_protection = true
```

## Troubleshooting

### Common Issues

1. **Instance Launch Failures**:
   ```bash
   # Check user data logs
   aws logs get-log-events --log-group-name /aws/ec2/federated-learning/coordinator --log-stream-name <instance-id>/user-data.log
   ```

2. **Database Connection Issues**:
   ```bash
   # Test database connectivity
   aws rds describe-db-instances --db-instance-identifier federated-learning-database
   ```

3. **Load Balancer Health Checks**:
   ```bash
   # Check target group health
   aws elbv2 describe-target-health --target-group-arn <target-group-arn>
   ```

### Debug Mode

Enable debug logging:

```hcl
enable_debug_mode = true
```

### Log Analysis

```bash
# Search for errors in logs
aws logs filter-log-events --log-group-name /aws/ec2/federated-learning/coordinator --filter-pattern "ERROR"

# Get recent log events
aws logs get-log-events --log-group-name /aws/ec2/federated-learning/coordinator --log-stream-name <stream-name> --start-time $(date -d '1 hour ago' +%s)000
```

## Scaling

### Manual Scaling

```bash
# Scale coordinator instances
aws autoscaling set-desired-capacity --auto-scaling-group-name federated-learning-coordinator-asg --desired-capacity 3

# Scale client instances
aws autoscaling set-desired-capacity --auto-scaling-group-name federated-learning-client-asg --desired-capacity 20
```

### Automatic Scaling

Scaling policies are configured based on:

- CPU utilization
- Memory usage
- Custom application metrics
- Training round requirements

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning**: This will delete all data including the database. Make sure to backup important data first.

## Support

For issues and questions:

1. Check CloudWatch logs for error messages
2. Review AWS console for resource status
3. Validate Terraform configuration
4. Check security group and network ACL rules

## Cost Estimation

Approximate monthly costs (us-west-2):

| Component | Configuration | Monthly Cost |
|-----------|---------------|--------------|
| Coordinator (t3.medium) | 1 instance | ~$30 |
| Clients (t3.small) | 5 instances | ~$75 |
| RDS (db.t3.micro) | Single AZ | ~$15 |
| Load Balancer | Application LB | ~$20 |
| NAT Gateway | 2 AZs | ~$90 |
| Storage & Data Transfer | Moderate usage | ~$20 |
| **Total** | | **~$250** |

Costs can be reduced by:
- Using spot instances (50-70% savings)
- Smaller instance types for development
- Single AZ deployment for non-production
- Reserved instances for long-term usage