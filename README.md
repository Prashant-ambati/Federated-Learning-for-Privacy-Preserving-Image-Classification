# Federated Learning for Privacy-Preserving Image Classification

A production-grade distributed federated learning framework that enables privacy-preserving image classification across multiple clients without sharing raw data.

## Overview

This system implements a coordinator-client architecture where clients train CNN models locally on their private data and only share model updates (not raw data) with a central coordinator. The coordinator aggregates these updates using the FedAvg algorithm and distributes the improved global model back to clients.

## Key Features

- **Privacy-First**: Raw data never leaves client devices
- **Differential Privacy**: Configurable noise injection to protect individual data points
- **Scalable Architecture**: Support for 50+ concurrent clients with horizontal scaling
- **Production Ready**: Containerized deployment with monitoring and fault tolerance
- **High Performance**: Optimized communication protocols and model compression
- **Standard ML Frameworks**: Built on PyTorch with support for standard datasets

## Architecture

### Core Components

1. **Coordinator Service**: Manages federated learning rounds and model aggregation
2. **Client Service**: Handles local training and privacy-preserving model updates  
3. **Aggregation Service**: Implements FedAvg algorithm and model compression

### Communication

- **gRPC**: High-performance model update transmission
- **Flask REST API**: Management and monitoring endpoints
- **Redis**: Caching and session management
- **PostgreSQL**: Persistent storage for training metadata

## Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Prashant-ambati/Federated-Learning-for-Privacy-Preserving-Image-Classification.git
cd Federated-Learning-for-Privacy-Preserving-Image-Classification

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Local Development

```bash
# Start infrastructure services
docker-compose up -d redis postgres

# Run coordinator service
fl-coordinator --config config/coordinator.yaml

# Run client (in separate terminal)
fl-client --config config/client.yaml --client-id client-001
```

## Configuration

### Privacy Parameters

- **Epsilon (ε)**: Controls privacy-utility tradeoff (lower = more private)
- **Delta (δ)**: Probability of privacy breach (typically 1e-5)
- **Max Gradient Norm**: Gradient clipping threshold for bounded sensitivity

### Training Parameters

- **Local Epochs**: Number of training epochs per client per round
- **Batch Size**: Training batch size (adaptive based on client capabilities)
- **Learning Rate**: Configurable per client based on computational power

## Performance Benchmarks

- **Accuracy**: Maintains 91%+ accuracy on MNIST dataset with differential privacy
- **Latency**: 25%+ reduction compared to centralized training approaches
- **Scalability**: Tested with up to 50 concurrent clients
- **Privacy**: Configurable ε-differential privacy guarantees

## Deployment

### AWS EC2 Deployment

```bash
# Build and push Docker images
docker build -t fl-coordinator:latest -f docker/Coordinator.dockerfile .
docker build -t fl-client:latest -f docker/Client.dockerfile .

# Deploy using Terraform
cd infrastructure/aws
terraform init
terraform apply
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/coordinator/
kubectl apply -f k8s/client/
```

## Monitoring

- **Training Progress**: Real-time metrics via REST API
- **System Health**: Prometheus metrics and Grafana dashboards
- **Privacy Budget**: Tracking and alerting for privacy parameter consumption
- **Performance**: Latency, throughput, and resource utilization monitoring

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Author

**Built with ❤️ by [Prashant Ambati](https://github.com/Prashant-ambati)**

*Passionate about privacy-preserving machine learning and distributed systems. This project represents a comprehensive implementation of federated learning principles with production-grade engineering practices.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Research References

- McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
- Abadi, M., et al. "Deep Learning with Differential Privacy." CCS 2016.
- Li, T., et al. "Federated Learning: Challenges, Methods, and Future Directions." IEEE Signal Processing Magazine 2020.

## Contact

For questions and support, please open an issue on GitHub or contact the development team.