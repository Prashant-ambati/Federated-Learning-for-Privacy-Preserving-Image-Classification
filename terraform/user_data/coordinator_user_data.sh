#!/bin/bash
# User data script for federated learning coordinator instances

set -e

# Variables from Terraform
DB_ENDPOINT="${db_endpoint}"
DB_NAME="${db_name}"
DB_USERNAME="${db_username}"
DB_PASSWORD="${db_password}"
ENVIRONMENT="${environment}"

# Log all output
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting coordinator setup at $(date)"

# Update system
yum update -y

# Install required packages
yum install -y \
    python3 \
    python3-pip \
    git \
    docker \
    postgresql \
    htop \
    curl \
    wget \
    unzip \
    awscli

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm

# Create application directory
mkdir -p /opt/federated-learning
cd /opt/federated-learning

# Clone the application (in production, you'd pull from a private repo or S3)
# For now, we'll create the necessary structure
mkdir -p {src,config,logs,data}

# Create configuration file
cat > config/coordinator.yaml << EOF
# Federated Learning Coordinator Configuration
server:
  host: "0.0.0.0"
  grpc_port: 50051
  rest_port: 8080
  health_port: 8081

database:
  host: "${DB_ENDPOINT}"
  port: 5432
  name: "${DB_NAME}"
  username: "${DB_USERNAME}"
  password: "${DB_PASSWORD}"
  pool_size: 10
  max_overflow: 20

federated_learning:
  min_clients: 3
  max_clients: 50
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.001
  total_rounds: 100
  round_timeout: 300
  model_type: "simple_cnn"

privacy:
  epsilon: 1.0
  delta: 1e-5
  max_grad_norm: 1.0
  noise_multiplier: 1.0

logging:
  level: "INFO"
  log_dir: "/opt/federated-learning/logs"
  enable_json: true
  enable_console: true
  max_file_size: 10485760  # 10MB
  backup_count: 5

monitoring:
  collection_interval: 30.0
  max_history: 1000
  enable_cloudwatch: true

storage:
  s3_bucket: "federated-learning-models-\${random_suffix}"
  model_storage_path: "/opt/federated-learning/data/models"

environment: "${ENVIRONMENT}"
EOF

# Create systemd service file
cat > /etc/systemd/system/federated-coordinator.service << 'EOF'
[Unit]
Description=Federated Learning Coordinator
After=network.target
Wants=network.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=/opt/federated-learning
Environment=PYTHONPATH=/opt/federated-learning
ExecStart=/usr/bin/python3 -m src.coordinator.main --config config/coordinator.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=federated-coordinator

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

# Create CloudWatch agent configuration
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "agent": {
        "metrics_collection_interval": 60,
        "run_as_user": "cwagent"
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/opt/federated-learning/logs/coordinator.log",
                        "log_group_name": "/aws/ec2/federated-learning/coordinator",
                        "log_stream_name": "{instance_id}/coordinator.log",
                        "timezone": "UTC"
                    },
                    {
                        "file_path": "/var/log/user-data.log",
                        "log_group_name": "/aws/ec2/federated-learning/coordinator",
                        "log_stream_name": "{instance_id}/user-data.log",
                        "timezone": "UTC"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "FederatedLearning/Coordinator",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60,
                "totalcpu": false
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "netstat": {
                "measurement": [
                    "tcp_established",
                    "tcp_time_wait"
                ],
                "metrics_collection_interval": 60
            },
            "swap": {
                "measurement": [
                    "swap_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF

# Install Python dependencies
pip3 install --upgrade pip
pip3 install \
    torch \
    torchvision \
    grpcio \
    grpcio-tools \
    flask \
    sqlalchemy \
    psycopg2-binary \
    boto3 \
    pyyaml \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    requests \
    prometheus-client \
    psutil

# Create health check script
cat > /opt/federated-learning/health_check.py << 'EOF'
#!/usr/bin/env python3
"""Health check script for coordinator."""

import requests
import sys
import time

def check_health():
    try:
        response = requests.get('http://localhost:8081/health', timeout=5)
        if response.status_code == 200:
            print("Health check passed")
            return True
        else:
            print(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
EOF

chmod +x /opt/federated-learning/health_check.py

# Create startup script
cat > /opt/federated-learning/start.sh << 'EOF'
#!/bin/bash
# Startup script for coordinator

set -e

echo "Starting federated learning coordinator..."

# Wait for database to be ready
if [ ! -z "$DB_ENDPOINT" ]; then
    echo "Waiting for database to be ready..."
    while ! pg_isready -h "$DB_ENDPOINT" -p 5432 -U "$DB_USERNAME"; do
        echo "Database not ready, waiting..."
        sleep 5
    done
    echo "Database is ready"
fi

# Initialize database if needed
python3 -c "
from src.shared.database import init_database
init_database()
print('Database initialized')
"

# Start the coordinator service
systemctl start federated-coordinator
systemctl enable federated-coordinator

echo "Coordinator started successfully"
EOF

chmod +x /opt/federated-learning/start.sh

# Set ownership
chown -R ec2-user:ec2-user /opt/federated-learning

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s

# Create cron job for health monitoring
cat > /etc/cron.d/coordinator-health << 'EOF'
# Health check every minute
* * * * * ec2-user /opt/federated-learning/health_check.py >> /var/log/health-check.log 2>&1
EOF

# Install and configure fail2ban for security
yum install -y epel-release
yum install -y fail2ban

cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
logpath = /var/log/secure
EOF

systemctl start fail2ban
systemctl enable fail2ban

# Configure automatic security updates
yum install -y yum-cron
sed -i 's/apply_updates = no/apply_updates = yes/' /etc/yum/yum-cron.conf
systemctl start yum-cron
systemctl enable yum-cron

# Set up log rotation
cat > /etc/logrotate.d/federated-learning << 'EOF'
/opt/federated-learning/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 ec2-user ec2-user
    postrotate
        systemctl reload federated-coordinator
    endscript
}
EOF

# Create monitoring script
cat > /opt/federated-learning/monitor.py << 'EOF'
#!/usr/bin/env python3
"""System monitoring script."""

import psutil
import boto3
import time
import json
from datetime import datetime

def send_metrics_to_cloudwatch():
    """Send custom metrics to CloudWatch."""
    cloudwatch = boto3.client('cloudwatch')
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Send metrics
    metrics = [
        {
            'MetricName': 'CPUUtilization',
            'Value': cpu_percent,
            'Unit': 'Percent',
            'Timestamp': datetime.utcnow()
        },
        {
            'MetricName': 'MemoryUtilization',
            'Value': memory.percent,
            'Unit': 'Percent',
            'Timestamp': datetime.utcnow()
        },
        {
            'MetricName': 'DiskUtilization',
            'Value': (disk.used / disk.total) * 100,
            'Unit': 'Percent',
            'Timestamp': datetime.utcnow()
        }
    ]
    
    try:
        cloudwatch.put_metric_data(
            Namespace='FederatedLearning/Coordinator/Custom',
            MetricData=metrics
        )
        print(f"Sent {len(metrics)} metrics to CloudWatch")
    except Exception as e:
        print(f"Failed to send metrics: {e}")

if __name__ == "__main__":
    send_metrics_to_cloudwatch()
EOF

chmod +x /opt/federated-learning/monitor.py

# Add monitoring to cron
cat >> /etc/cron.d/coordinator-health << 'EOF'
# Send custom metrics every 5 minutes
*/5 * * * * ec2-user /opt/federated-learning/monitor.py >> /var/log/monitoring.log 2>&1
EOF

# Wait for all services to be ready and start the application
sleep 30

# Start the coordinator (this will be done by the startup script)
su - ec2-user -c "cd /opt/federated-learning && ./start.sh"

echo "Coordinator setup completed at $(date)"

# Signal that the instance is ready
/opt/aws/bin/cfn-signal -e $? --stack ${AWS::StackName} --resource CoordinatorAutoScalingGroup --region ${AWS::Region} || true