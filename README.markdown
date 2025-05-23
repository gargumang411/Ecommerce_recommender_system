# E-Commerce Recommendation System

## Project Overview
This project implements a prototype recommendation system for an e-commerce platform using a two-tower collaborative filtering model. Built with PyTorch and trained on the [Amazon Electronics 5-core dataset](https://nijianmo.github.io/amazon/index.html) (~6.7M reviews, ~72,678 users, 159,748 items), the system delivers personalized product recommendations. It supports multiple recommendation types (personalized, popularity-based, trend-based) using aggregated features to enhance training stability. The system is designed to meet non-functional requirements including scalability, low latency, high availability, security, maintainability, cost-effectiveness, efficient paging, and continuous monitoring, with robust tooling for development and deployment.

## Dataset
The Amazon Electronics 5-core dataset contains ~6.7M user reviews for electronic products, ensuring each user and item has at least 5 interactions. Preprocessing steps include:
- Encoding user and item IDs.
- Splitting data into train/validation/test sets (~88.5%/10%/1.5%).
- Assigning `interaction_status` (0.0: ignored, 0.3: interacted, 0.6: added to cart, 1.0: bought) based on rating ranks, normalized to [0,1].
- Adding aggregated features:
  - **User Activity**: Number of items bought by a user in the last 7 days.
  - **Item Popularity**: Number of times an item was bought across all users.
These features reduce sparsity and improve training stability.

## Model Architecture
The two-tower model, implemented in PyTorch, consists of:
- **User Tower**: Embeds user IDs into a 128-dimensional vector, concatenates aggregated user features (e.g., items bought in last 7 days), followed by linear layers, BatchNorm, and ReLU.
- **Item Tower**: Embeds item IDs, concatenates aggregated item features (e.g., number of times bought), followed by linear layers, BatchNorm, and ReLU.
- **Prediction**: Dot product of embeddings, scaled by sigmoid to predict interaction scores (0.0–1.0).
- **Training**: Optimized with Adam using MSE loss on a T4 GPU. Hyperparameters: learning rate = 0.01, embedding dimension = 128, batch size = 256. Planned enhancements include learning rate scheduling (e.g., ReduceLROnPlateau), gradient clipping, alternative loss functions (e.g., BPR loss), and increased regularization (e.g., dropout, L2).

## Recommendation Types
The system supports:
- **Personalized Recommendations**: Predict user-item interaction scores using the two-tower model.
- **Popularity-Based Recommendations**: Rank items by aggregated popularity for users with limited history.
- **Trend-Based Recommendations**: Prioritize items based on recent user activity.
These are blended during inference, with a re-ranking step for fairness and diversity.

## System Design
The system is architected to meet production-level non-functional requirements.

### Scalability
- **Data Storage**: Apache Cassandra for user interactions, item metadata, and aggregated features.
- **Model Training**: Apache Spark on AWS EMR for distributed training.
- **Serving**: AWS EC2 with auto-scaling for inference.

### Latency
- **Precomputation**: Precompute recommendations and cache in Redis for sub-200ms retrieval.
- **Hybrid Real-Time**: Trigger real-time inference for high-value users or events, caching results in Redis.
- **Inference Optimization**: Use model quantization and NVIDIA Triton for efficient serving.

### Throughput
- **Load Balancing**: AWS Elastic Load Balancer for traffic distribution.
- **Asynchronous Processing**: Apache Kafka for high-concurrency data ingestion, processing feedback with <1-minute latency.

### Model Update Frequency
- **Retraining Schedule**: Nightly retraining with incremental learning to capture trends.
- **Automation**: Apache Airflow to manage retraining workflows.

### Accuracy
- **Evaluation Metrics**: Monitor Precision@10 (~0.1–0.2), MAE (<0.21), and MSE.
- **Re-ranking**: Post-process with multi-objective optimization for fairness/diversity using diversity metrics (e.g., Determinantal Point Processes) and fairness metrics (e.g., demographic parity).
- **Feedback Loop**: Integrate feedback via Kafka.
- **A/B Testing**: Compare model versions to optimize engagement.

### Availability
- **Redundancy**: Deploy across AWS availability zones for >99.99% uptime.
- **Failover**: AWS Route 53 for failover management.
- **Continuous Monitoring**: Prometheus and Grafana for real-time tracking.

### Security
- **Data Encryption**: AES-256 (e.g., AWS S3) and TLS 1.3.
- **Access Control**: AWS IAM for role-based access control (RBAC).
- **Data Abstraction**: Abstract pinpoint addresses to city names or regions (e.g., "New York" instead of "123 Main St") to minimize PII exposure.
- **Compliance**: Adhere to GDPR and CCPA.

### Maintainability
- **Modular Design**: Separate data ingestion, training, and serving.
- **Version Control**: Git for source control.
- **Automated Testing**: pytest for unit and integration tests.

### Cost-Effectiveness
- **Resource Optimization**: AWS Spot Instances and serverless AWS Lambda for low-traffic inference.
- **Redis Optimization**: Eviction policies for memory efficiency.
- **Cost Monitoring**: AWS Cost Explorer to analyze retraining frequency trade-offs.

### Paging
- **Data Retrieval**: Token-based pagination in Cassandra.
- **Recommendation Delivery**: Serve in batches (e.g., 10 items per page).

### Cold-Start Handling
- **Initial Recommendations**: Show popular products (e.g., top 10 by purchases) for new users, cached in Redis.
- **Content-Based Enhancement**: Incorporate item metadata (e.g., category, price) or cluster users (e.g., k-means on demographics).

### Session-Based Recommendations
- **Embedding Management**: Precompute item embeddings (stored in Cassandra) and cache user embeddings in Redis for real-time updates.

## Tooling
- **Machine Learning**: PyTorch for model development.
- **Data Processing**: Pandas and NumPy for preprocessing.
- **Cloud Platform**: Google Colab (prototype), planned AWS deployment.
- **Monitoring**: Prometheus and Grafana for system health.
- **Logging**: ELK Stack for log analysis.
- **CI/CD**: GitHub Actions for automated deployment.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone [repository_url]
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add dataset:
   - Place `Electronics_5.json.gz` in `/content/` or mount Google Drive.
4. Preprocess data:
   ```bash
   python preprocess.py
   ```
5. Train the model:
   ```bash
   python train.py
   ```
6. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Results and Evaluation
- **Test Loss (MSE)**: 0.1792
- **Test MAE**: 0.3946
- **Precision@10**: 0.0110
These results indicate training challenges (e.g., hitting local minima), which planned enhancements aim to address.

## Future Work
- Refine preprocessing with raw ratings or binary labels.
- Enhance model with deeper layers, alternative loss functions (e.g., BPR), and regularization.
- Implement Kafka for real-time feedback and hybrid inference.
- Deploy on AWS with A/B testing, continuous monitoring, and content-based cold-start strategies.

## System Design Diagram
### ASCII Representation
```
[Users] --> [Load Balancer (AWS ELB)]
         |           |
         v           v
[API Gateway] --> [Inference Service (AWS EC2 + NVIDIA Triton)]
         |                   |
         v                   v
[Redis Cache]         [Cassandra DB]
         |                   |
         v                   v
[Prometheus/Grafana]  [Kafka (Feedback Loop)]
         |                   |
         v                   v
[Airflow (Retraining)] --> [Spark Training Cluster]
```


### System Architecture Table
| Component            | Technology                        | Purpose                          | Non-Functional Requirement       |
|---------------------|----------------------------------|----------------------------------|----------------------------------|
| Data Storage        | Apache Cassandra                 | Store data, features             | Scalability, Paging              |
| Data Ingestion      | Apache Kafka                     | Stream interactions              | Throughput, Scalability          |
| Model Training      | PyTorch, Apache Spark            | Train model                      | Scalability, Accuracy             |
| Model Serving       | AWS EC2, NVIDIA Triton           | Serve recommendations            | Latency, Throughput              |
| Caching             | Redis                            | Cache embeddings, results        | Latency                          |
| Monitoring          | Prometheus, Grafana              | Continuous tracking              | Availability, Maintainability     |
| Logging             | ELK Stack                        | Analyze logs                     | Maintainability                  |
| CI/CD               | GitHub Actions                   | Automate deployment              | Maintainability                  |
| Security            | AWS IAM, TLS 1.3, AES-256        | Protect data                     | Security                         |
| Cost Management     | AWS Cost Explorer, Spot Instances, Lambda | Optimize costs            | Cost-Effectiveness               |
