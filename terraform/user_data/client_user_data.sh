#!/bin/bash
# User data script for federated learning client instances

set -e

# Variables from Terraform
COORDINATOR_ENDPOINT="${coordinator_endpoint}"
ENVIRONMENT="${environment}"

# Log all output
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting client setup at $(date)"

# Update system
yum update -y

# Install required packages
yum install -y \
    python3 \
    python3-pip \
    git \
    docker \
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

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm

# Create application directory
mkdir -p /opt/federated-learning-client
cd /opt/federated-learning-client

# Create directory structure
mkdir -p {src,config,logs,data}

# Generate unique client ID
CLIENT_ID="client_$(curl -s http://169.254.169.254/latest/meta-data/instance-id)_$(date +%s)"

# Create configuration file
cat > config/client.yaml << EOF
# Federated Learning Client Configuration
client:
  client_id: "${CLIENT_ID}"
  coordinator_host: "${COORDINATOR_ENDPOINT}"
  coordinator_port: 50051
  health_port: 8082
  
coordinator:
  grpc_endpoint: "${COORDINATOR_ENDPOINT}:50051"
  rest_endpoint: "http://${COORDINATOR_ENDPOINT}/api"
  connection_timeout: 30
  retry_attempts: 3
  retry_delay: 5

training:
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.001
  model_type: "simple_cnn"
  dataset: "mnist"
  data_path: "/opt/federated-learning-client/data"

privacy:
  epsilon: 1.0
  delta: 1e-5
  max_grad_norm: 1.0
  noise_multiplier: 1.0

capabilities:
  compute_power: "medium"
  network_bandwidth_mbps: 100
  available_samples: 1000
  has_gpu: false
  memory_gb: 4

logging:
  level: "INFO"
  log_dir: "/opt/federated-learning-client/logs"
  enable_json: true
  enable_console: true
  max_file_size: 10485760  # 10MB
  backup_count: 5

monitoring:
  collection_interval: 30.0
  max_history: 1000
  enable_cloudwatch: true

environment: "${ENVIRONMENT}"
EOF

# Create systemd service file
cat > /etc/systemd/system/federated-client.service << 'EOF'
[Unit]
Description=Federated Learning Client
After=network.target
Wants=network.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=/opt/federated-learning-client
Environment=PYTHONPATH=/opt/federated-learning-client
ExecStart=/usr/bin/python3 -m src.client.main --config config/client.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=federated-client

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
                        "file_path": "/opt/federated-learning-client/logs/client.log",
                        "log_group_name": "/aws/ec2/federated-learning/client",
                        "log_stream_name": "{instance_id}/client.log",
                        "timezone": "UTC"
                    },
                    {
                        "file_path": "/var/log/user-data.log",
                        "log_group_name": "/aws/ec2/federated-learning/client",
                        "log_stream_name": "{instance_id}/user-data.log",
                        "timezone": "UTC"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "FederatedLearning/Client",
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

# Download MNIST dataset
python3 -c "
import torchvision
import torchvision.transforms as transforms
import os

# Create data directory
os.makedirs('/opt/federated-learning-client/data', exist_ok=True)

# Download MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(
    root='/opt/federated-learning-client/data',
    train=True,
    download=True,
    transform=transform
)
testset = torchvision.datasets.MNIST(
    root='/opt/federated-learning-client/data',
    train=False,
    download=True,
    transform=transform
)
print('MNIST dataset downloaded successfully')
"

# Create health check script
cat > /opt/federated-learning-client/health_check.py << 'EOF'
#!/usr/bin/env python3
"""Health check script for client."""

import requests
import sys
import time

def check_health():
    try:
        response = requests.get('http://localhost:8082/health', timeout=5)
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

chmod +x /opt/federated-learning-client/health_check.py

# Create startup script
cat > /opt/federated-learning-client/start.sh << 'EOF'
#!/bin/bash
# Startup script for client

set -e

echo "Starting federated learning client..."

# Wait for coordinator to be ready
echo "Waiting for coordinator to be ready..."
while ! curl -f "http://${COORDINATOR_ENDPOINT}/health" >/dev/null 2>&1; do
    echo "Coordinator not ready, waiting..."
    sleep 10
done
echo "Coordinator is ready"

# Start the client service
systemctl start federated-client
systemctl enable federated-client

echo "Client started successfully"
EOF

chmod +x /opt/federated-learning-client/start.sh

# Set ownership
chown -R ec2-user:ec2-user /opt/federated-learning-client

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s

# Create cron job for health monitoring
cat > /etc/cron.d/client-health << 'EOF'
# Health check every minute
* * * * * ec2-user /opt/federated-learning-client/health_check.py >> /var/log/health-check.log 2>&1
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
cat > /etc/logrotate.d/federated-learning-client << 'EOF'
/opt/federated-learning-client/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 ec2-user ec2-user
    postrotate
        systemctl reload federated-client
    endscript
}
EOF

# Create monitoring script
cat > /opt/federated-learning-client/monitor.py << 'EOF'
#!/usr/bin/env python3
"""System monitoring script for client."""

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
            Namespace='FederatedLearning/Client/Custom',
            MetricData=metrics
        )
        print(f"Sent {len(metrics)} metrics to CloudWatch")
    except Exception as e:
        print(f"Failed to send metrics: {e}")

if __name__ == "__main__":
    send_metrics_to_cloudwatch()
EOF

chmod +x /opt/federated-learning-client/monitor.py

# Add monitoring to cron
cat >> /etc/cron.d/client-health << 'EOF'
# Send custom metrics every 5 minutes
*/5 * * * * ec2-user /opt/federated-learning-client/monitor.py >> /var/log/monitoring.log 2>&1
EOF

# Create data partitioning script for federated learning
cat > /opt/federated-learning-client/partition_data.py << 'EOF'
#!/usr/bin/env python3
"""Partition MNIST data for federated learning simulation."""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
from torch.utils.data import Subset

def partition_mnist_data(client_id, num_clients=10, data_dir='/opt/federated-learning-client/data'):
    """Partition MNIST data for federated learning."""
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )
    
    # Create non-IID partition based on client ID
    client_idx = hash(client_id) % num_clients
    
    # Sort by labels to create non-IID distribution
    targets = np.array(trainset.targets)
    indices = np.argsort(targets)
    
    # Each client gets data from 2-3 classes primarily
    classes_per_client = 2
    start_class = (client_idx * classes_per_client) % 10
    end_class = ((client_idx + 1) * classes_per_client) % 10
    
    if start_class < end_class:
        client_classes = list(range(start_class, end_class))
    else:
        client_classes = list(range(start_class, 10)) + list(range(0, end_class))
    
    # Add some samples from other classes for diversity
    client_classes.extend([(start_class + 5) % 10])
    
    # Get indices for client's classes
    client_indices = []
    for class_label in client_classes:
        class_indices = indices[targets[indices] == class_label]
        # Take a portion of the class data
        portion_size = len(class_indices) // (num_clients // len(client_classes))
        start_idx = (client_idx % (num_clients // len(client_classes))) * portion_size
        end_idx = start_idx + portion_size
        client_indices.extend(class_indices[start_idx:end_idx])
    
    # Shuffle client indices
    np.random.shuffle(client_indices)
    
    # Create subset
    client_dataset = Subset(trainset, client_indices)
    
    # Save partition info
    partition_info = {
        'client_id': client_id,
        'num_samples': len(client_indices),
        'classes': client_classes,
        'class_distribution': {
            class_label: len([i for i in client_indices if targets[i] == class_label])
            for class_label in range(10)
        }
    }
    
    with open(f'{data_dir}/partition_info_{client_id}.pkl', 'wb') as f:
        pickle.dump(partition_info, f)
    
    print(f"Partitioned data for {client_id}: {len(client_indices)} samples")
    print(f"Class distribution: {partition_info['class_distribution']}")
    
    return client_dataset, partition_info

if __name__ == "__main__":
    import sys
    client_id = sys.argv[1] if len(sys.argv) > 1 else "default_client"
    partition_mnist_data(client_id)
EOF

chmod +x /opt/federated-learning-client/partition_data.py

# Partition data for this client
python3 /opt/federated-learning-client/partition_data.py "${CLIENT_ID}"

# Wait for all services to be ready and start the application
sleep 30

# Start the client (this will be done by the startup script)
su - ec2-user -c "cd /opt/federated-learning-client && ./start.sh"

echo "Client setup completed at $(date)"

# Signal that the instance is ready
/opt/aws/bin/cfn-signal -e $? --stack ${AWS::StackName} --resource ClientAutoScalingGroup --region ${AWS::Region} || true