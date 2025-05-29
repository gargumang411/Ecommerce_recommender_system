# E-Commerce Recommendation System

## Project Overview
This project implements a recommendation system for an e-commerce platform's home page using a two-tower collaborative filtering model. Built with PyTorch and trained on the Amazon Electronics 5-core dataset](https://nijianmo.github.io/amazon/index.html) (~6.7M reviews, ~72,678 users, 159,748 items), the system delivers personalized product recommendations. It supports multiple recommendation types (personalized & popularity-based) using aggregated features to enhance training stability. 

Non-functional Requirements: Scalability, Low Latency, High Availability, Reliability, Security, Maintainability, Cost-Effectiveness, and Continuous Monitoring.

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
- **Training**: Optimized with Adam using MSE loss on a T4 GPU. Hyperparameters: learning rate = 0.01, embedding dimension = 128, batch size = 256. Planned enhancements include learning rate scheduling (e.g., ReduceLROnPlateau), gradient clipping, alternative loss functions (e.g., BPR loss), and increased regularization (e.g., dropout, L2) - Due to computational resource constraints, hyperparameters were assumed rather than decided using gridsearch. 

## Recommendation Types
The system supports:
- **Personalized Recommendations**: Predict user-item interaction scores using the two-tower model.
- **Popularity-Based Recommendations**: Rank items by aggregated popularity for users with limited history.
- **Trend-Based Recommendations**: Prioritize items based on recent user activity.
- **Session-Based Recommendations**: This would be a separate model created integrating sequential modelling techniques such as GRU or Transformers, it is beyond this project's scope.

- **These are blended during inference, with a re-ranking step for fairness and diversity.**

### Cold-Start Handling
- **For New Users**: Show popular products (e.g., top 10 by purchases) for new users, cached in Redis to gain initial data. 
- **For New Items**: Incorporate item metadata (e.g., category, price, product description, seller rating) to create item embeddings using a pre-trained Encoder only transformer like Sentence-BERT (SBERT) model and match with others products embeddings (created in the same way) and then recommend to similar users to whom we recommend other similar products to gain initial data before adding it to the two-tower model training.


## System Design
The system is architected to meet production-level non-functional requirements.


#### Scalability
The system scales for ~1M active users, 10M items (with images and videos), with 5% monthly growth. Capacity estimation for Year 1:
- **Storage Needs**: ~28.75 TB (3x replication), calculated as:
  - User interactions: 1M users × 1,095 interactions/user (3/day × 365) × 36 bytes ≈ 39.42 GB (~0.0385 TB).
  - Item metadata: 10M items × 536 bytes (including 500-byte descriptions) ≈ 5.36 GB (~0.0052 TB).
  - Item multimedia: 10M items × 0.5 MB (images) = 4.77 TB + 1M videos × 5 MB = 4.77 TB, totaling 9.54 TB.
  - Raw total: 0.0385 TB + 0.0052 TB + 9.54 TB ≈ 9.5837 TB × 3 (replication) = 28.75 TB, stored in Amazon S3.
- **Traffic**: ~5M daily requests (~58 RPS average), peaking at ~50M requests/day (~579 RPS), growing to ~9M requests/day (~104 RPS) and ~1,040 RPS (peak) in Year 1.
- **Data Storage**: Apache Cassandra for interactions and metadata, with S3 for multimedia.
- **Multimedia Serving**: Amazon CloudFront handles ~45.76 Gbps peak bandwidth for images and videos.
- **Model Training**: Amazon SageMaker with 20 `ml.p3.2xlarge` instances (20 V100 GPUs) for the two-tower model.
- **Serving**: SageMaker auto-scales from 10 to 50 `ml.g4dn.xlarge` instances (T4 GPUs) for ~1,040 RPS peak with sub-200ms latency.

#### Latency
The system targets sub-200ms end-to-end latency for 1M users globally, handling 1,040 RPS peak.
- **Precomputation and Caching**: Precompute recommendations for 90% of users, cached in Amazon ElastiCache for <50ms retrieval.
- **Hybrid Real-Time**: Use Amazon SageMaker for <100ms real-time inference for high-value users or events, caching in ElastiCache.
- **Paging for Fast Retrieval**: Apply token-based pagination in Apache Cassandra, serving batches of 10 items in <50ms, supporting the sub-200ms goal.
- **Multimedia Delivery**: Serve images (500 KB) via Amazon CloudFront (<50ms) and videos (5 MB for 10%) via AWS Media Services (<100ms) with edge caching.
- **Inference Efficiency**: Optimize with SageMaker Neo and T4 GPUs for <100ms inference at peak load.
- **Global Delivery**: Leverage CloudFront and AWS Global Accelerator for sub-200ms across regions.

#### Throughput
The system handles 1,040 RPS peak for 1M users, delivering recommendations and multimedia efficiently.
- **Load Balancing**: AWS Application Load Balancer distributes traffic across SageMaker endpoints, supporting 1,040 RPS.
- **Inference Throughput**: SageMaker with 50 T4 GPUs (2,500 RPS capacity) ensures peak performance.
- **Feedback Processing**: Apache Kafka ingests user feedback (e.g., clicks, purchases) at 1,040 RPS, processing in <10 seconds for real-time model updates, with data stored in S3 for batch retraining.
- **Multimedia Throughput**: Amazon CloudFront delivers ~45.76 Gbps for images (500 KB) and videos (5 MB for 10%) directly from S3, using edge caching.

#### Model Update Frequency
The system updates the two-tower model to capture trends, supporting 1M users and 10M items.
- **Retraining Schedule**: Nightly retraining with incremental learning (last 24 hours) for stability, plus hourly retraining for high-velocity items (e.g., during sales).
- **Automation**: Apache Airflow manages workflows, integrated with Amazon SageMaker Pipelines for scalable orchestration.

#### Accuracy
The system optimizes recommendation quality for 1M Daily active users with real-time feedback and long-term value.
- **Evaluation Metrics**: Monitor Precision@10, MAE, and MSE, for short-term performance.
- **Long-Term Metrics**: Assess user engagement trends, customer retention rate, revenue impact, churn reduction, and diversity maintenance over time.
- **Re-ranking**: Post-process with multi-objective optimization using Determinantal Point Processes for diversity and demographic parity for fairness.
- **Feedback Loop**: Integrate Kafka feedback at 1,040 RPS with <10-second latency for real-time updates.
- **A/B Testing**: Compare model versions or Personalize to optimize both short-term and long-term engagement.

### Availability
- **Redundancy**: Deploy across multiple geographic regions and isolated availability zones with data replication.
- **Failover**: Implement automated failover mechanisms to minimize downtime during outages.
- **Continuous Monitoring**: Provide real-time system health tracking with automated alerts and recovery triggers.
- **Disaster Recovery**: Establish a recovery plan to restore services within a defined timeframe after major disruptions.

### Security
- A flow diagram to see how data exactly flows can help identify potential leakages/data security concerns. 
- **Data Abstraction**: Abstract pinpoint addresses to city names or regions (e.g., "New York" instead of "123 Main St") to minimize PII exposure. Similarly age can be abstracted to create age groups for more abstraction. Similarly other PII features can be abstracted. 
- **Data Encryption**: AES-256 and TLS 1.3.
- **Access Control**: AWS IAM for role-based access control (RBAC).
- **Compliance**: Adhere to GDPR and CCPA.

### Maintainability
- **Modular Design**: Separate data ingestion, training, and serving.
- **Version Control**: Git for source control.
- **Automated Testing**: pytest for unit and integration tests.

#### Maintainability
The system ensures easy maintenance for 1M daily active users and 10M items, supporting updates and debugging at scale.
- **Modular Design**: Separate data ingestion, training, and serving for independent updates and scaling.
- **Version Control**: Track code and model artifacts to enable change management and rollbacks.
- **Automated Testing and Deployment**: Use automated testing and deployment pipelines using a CI/CD tool for reliability and streamlined updates.
- **Documentation**: Maintain comprehensive documentation for code, architecture, and workflows to aid onboarding and troubleshooting.
- **Logging and Debugging**: Enable detailed logging and debugging to resolve issues quickly, even at 1,040 RPS peak.

### Cost-Effectiveness
The system minimizes costs for 1M users and 10M items while maintaining performance (sub-200ms latency, 1,040 RPS peak).
- **Resource Optimization**: Good to use cost-efficient compute instances and serverless options likw AWS Spot Instances and serverless AWS Lambda for low-traffic inference.
- **Storage Efficiency**: Optimize costs for 28.75 TB of storage by archiving less frequently accessed data (e.g., older multimedia).
- **Data Transfer Optimization**: Reduce costs for 45.76 Gbps multimedia delivery by caching at edge locations.
- **Retraining Cost Management**: Balance nightly and hourly retraining to capture trends cost-effectively.
- **Cost Monitoring and Analysis**: Continuously monitor and analyze costs to identify savings opportunities using tools like AWS Cost Explorer.


### Session-Based Recommendations
- **Embedding Management**: Precompute item embeddings (stored in Cassandra) and cache user embeddings in Redis for real-time updates.


## Tooling
- **Machine Learning**: PyTorch for model development.
- **Data Processing**: Pandas and NumPy for preprocessing.
- **Cloud Platform**: AWS for production deployment, Google Colab for initial prototyping.
- **Distributed Training**: Apache Spark for large-scale model training.
- **Data Storage**: Apache Cassandra for interactions and metadata, Amazon S3 for multimedia.
- **Caching**: Amazon ElastiCache for recommendation retrieval.
- **Load Balancing**: AWS Application Load Balancer for traffic distribution.
- **Multimedia Serving**: Amazon CloudFront for images and videos, AWS Media Services for streaming.
- **Monitoring**: Prometheus and Grafana for system health.
- **Logging**: ELK Stack for log analysis.
- **Model Management**: MLflow for experiment tracking and versioning.
- **Workflow Automation**: Apache Airflow for retraining pipelines.
- **CI/CD**: GitHub Actions for automated deployment.
- **Containerization**: Docker for packaging and consistent environments.


## Clone the repository:
   ```bash
   git clone https://github.com/gargumang411/Ecommerce_recommender_system
   ```


## Results and Evaluation
- **Test Loss (MSE)**: 0.1792
- **Test MAE**: 0.3946
- **Precision@10**: 0.0110
These results indicate training challenges with limited compute resources and in finding a good dataset for the purpose since real ecommerce interaction datasets aren't open-source. Planned enhancements aim to address these.

## Future Work
- Refine preprocessing with raw ratings or binary labels.
- Enhance model with deeper layers, alternative loss functions (e.g., BPR), and regularization.
- Implement Kafka for real-time feedback and hybrid inference.
- Deploy on AWS with A/B testing, continuous monitoring, and content-based cold-start strategies.

## System Design Diagram
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
# E-Commerce Recommendation System

## Project Overview
This project implements a recommendation system for an e-commerce platform's home page using a two-tower collaborative filtering model. Built with PyTorch and trained on the Amazon Electronics 5-core dataset](https://nijianmo.github.io/amazon/index.html) (~6.7M reviews, ~72,678 users, 159,748 items), the system delivers personalized product recommendations. It supports multiple recommendation types (personalized & popularity-based) using aggregated features to enhance training stability. 

Non-functional Requirements: Scalability, Low Latency, High Availability, Reliability, Security, Maintainability, Cost-Effectiveness, and Continuous Monitoring.

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
- **Training**: Optimized with Adam using MSE loss on a T4 GPU. Hyperparameters: learning rate = 0.01, embedding dimension = 128, batch size = 256. Planned enhancements include learning rate scheduling (e.g., ReduceLROnPlateau), gradient clipping, alternative loss functions (e.g., BPR loss), and increased regularization (e.g., dropout, L2) - Due to computational resource constraints, hyperparameters were assumed rather than decided using gridsearch. 

## Recommendation Types
The system supports:
- **Personalized Recommendations**: Predict user-item interaction scores using the two-tower model.
- **Popularity-Based Recommendations**: Rank items by aggregated popularity for users with limited history.
- **Trend-Based Recommendations**: Prioritize items based on recent user activity.
- **Session-Based Recommendations**: This would be a separate model created integrating sequential modelling techniques such as GRU or Transformers, it is beyond this project's scope.

- **These are blended during inference, with a re-ranking step for fairness and diversity.**

### Cold-Start Handling
- **For New Users**: Show popular products (e.g., top 10 by purchases) for new users, cached in Redis to gain initial data. 
- **For New Items**: Incorporate item metadata (e.g., category, price, product description, seller rating) to create item embeddings using a pre-trained Encoder only transformer like Sentence-BERT (SBERT) model and match with others products embeddings (created in the same way) and then recommend to similar users to whom we recommend other similar products to gain initial data before adding it to the two-tower model training.


## System Design
The system is architected to meet production-level non-functional requirements.


#### Scalability
The system scales for ~1M active users, 10M items (with images and videos), with 5% monthly growth. Capacity estimation for Year 1:
- **Storage Needs**: ~28.75 TB (3x replication), calculated as:
  - User interactions: 1M users × 1,095 interactions/user (3/day × 365) × 36 bytes ≈ 39.42 GB (~0.0385 TB).
  - Item metadata: 10M items × 536 bytes (including 500-byte descriptions) ≈ 5.36 GB (~0.0052 TB).
  - Item multimedia: 10M items × 0.5 MB (images) = 4.77 TB + 1M videos × 5 MB = 4.77 TB, totaling 9.54 TB.
  - Raw total: 0.0385 TB + 0.0052 TB + 9.54 TB ≈ 9.5837 TB × 3 (replication) = 28.75 TB, stored in Amazon S3.
- **Traffic**: ~5M daily requests (~58 RPS average), peaking at ~50M requests/day (~579 RPS), growing to ~9M requests/day (~104 RPS) and ~1,040 RPS (peak) in Year 1.
- **Data Storage**: Apache Cassandra for interactions and metadata, with S3 for multimedia.
- **Multimedia Serving**: Amazon CloudFront handles ~45.76 Gbps peak bandwidth for images and videos.
- **Model Training**: Amazon SageMaker with 20 `ml.p3.2xlarge` instances (20 V100 GPUs) for the two-tower model.
- **Serving**: SageMaker auto-scales from 10 to 50 `ml.g4dn.xlarge` instances (T4 GPUs) for ~1,040 RPS peak with sub-200ms latency.

#### Latency
The system targets sub-200ms end-to-end latency for 1M users globally, handling 1,040 RPS peak.
- **Precomputation and Caching**: Precompute recommendations for 90% of users, cached in Amazon ElastiCache for <50ms retrieval.
- **Hybrid Real-Time**: Use Amazon SageMaker for <100ms real-time inference for high-value users or events, caching in ElastiCache.
- **Paging for Fast Retrieval**: Apply token-based pagination in Apache Cassandra, serving batches of 10 items in <50ms, supporting the sub-200ms goal.
- **Multimedia Delivery**: Serve images (500 KB) via Amazon CloudFront (<50ms) and videos (5 MB for 10%) via AWS Media Services (<100ms) with edge caching.
- **Inference Efficiency**: Optimize with SageMaker Neo and T4 GPUs for <100ms inference at peak load.
- **Global Delivery**: Leverage CloudFront and AWS Global Accelerator for sub-200ms across regions.

#### Throughput
The system handles 1,040 RPS peak for 1M users, delivering recommendations and multimedia efficiently.
- **Load Balancing**: AWS Application Load Balancer distributes traffic across SageMaker endpoints, supporting 1,040 RPS.
- **Inference Throughput**: SageMaker with 50 T4 GPUs (2,500 RPS capacity) ensures peak performance.
- **Feedback Processing**: Apache Kafka ingests user feedback (e.g., clicks, purchases) at 1,040 RPS, processing in <10 seconds for real-time model updates, with data stored in S3 for batch retraining.
- **Multimedia Throughput**: Amazon CloudFront delivers ~45.76 Gbps for images (500 KB) and videos (5 MB for 10%) directly from S3, using edge caching.

#### Model Update Frequency
The system updates the two-tower model to capture trends, supporting 1M users and 10M items.
- **Retraining Schedule**: Nightly retraining with incremental learning (last 24 hours) for stability, plus hourly retraining for high-velocity items (e.g., during sales).
- **Automation**: Apache Airflow manages workflows, integrated with Amazon SageMaker Pipelines for scalable orchestration.

#### Accuracy
The system optimizes recommendation quality for 1M Daily active users with real-time feedback and long-term value.
- **Evaluation Metrics**: Monitor Precision@10, MAE, and MSE, for short-term performance.
- **Long-Term Metrics**: Assess user engagement trends, customer retention rate, revenue impact, churn reduction, and diversity maintenance over time.
- **Re-ranking**: Post-process with multi-objective optimization using Determinantal Point Processes for diversity and demographic parity for fairness.
- **Feedback Loop**: Integrate Kafka feedback at 1,040 RPS with <10-second latency for real-time updates.
- **A/B Testing**: Compare model versions or Personalize to optimize both short-term and long-term engagement.

### Availability
- **Redundancy**: Deploy across multiple geographic regions and isolated availability zones with data replication.
- **Failover**: Implement automated failover mechanisms to minimize downtime during outages.
- **Continuous Monitoring**: Provide real-time system health tracking with automated alerts and recovery triggers.
- **Disaster Recovery**: Establish a recovery plan to restore services within a defined timeframe after major disruptions.

### Security
- A flow diagram to see how data exactly flows can help identify potential leakages/data security concerns. 
- **Data Abstraction**: Abstract pinpoint addresses to city names or regions (e.g., "New York" instead of "123 Main St") to minimize PII exposure. Similarly age can be abstracted to create age groups for more abstraction. Similarly other PII features can be abstracted. 
- **Data Encryption**: AES-256 and TLS 1.3.
- **Access Control**: AWS IAM for role-based access control (RBAC).
- **Compliance**: Adhere to GDPR and CCPA.

### Maintainability
- **Modular Design**: Separate data ingestion, training, and serving.
- **Version Control**: Git for source control.
- **Automated Testing**: pytest for unit and integration tests.

#### Maintainability
The system ensures easy maintenance for 1M daily active users and 10M items, supporting updates and debugging at scale.
- **Modular Design**: Separate data ingestion, training, and serving for independent updates and scaling.
- **Version Control**: Track code and model artifacts to enable change management and rollbacks.
- **Automated Testing and Deployment**: Use automated testing and deployment pipelines using a CI/CD tool for reliability and streamlined updates.
- **Documentation**: Maintain comprehensive documentation for code, architecture, and workflows to aid onboarding and troubleshooting.
- **Logging and Debugging**: Enable detailed logging and debugging to resolve issues quickly, even at 1,040 RPS peak.

### Cost-Effectiveness
The system minimizes costs for 1M users and 10M items while maintaining performance (sub-200ms latency, 1,040 RPS peak).
- **Resource Optimization**: Good to use cost-efficient compute instances and serverless options likw AWS Spot Instances and serverless AWS Lambda for low-traffic inference.
- **Storage Efficiency**: Optimize costs for 28.75 TB of storage by archiving less frequently accessed data (e.g., older multimedia).
- **Data Transfer Optimization**: Reduce costs for 45.76 Gbps multimedia delivery by caching at edge locations.
- **Retraining Cost Management**: Balance nightly and hourly retraining to capture trends cost-effectively.
- **Cost Monitoring and Analysis**: Continuously monitor and analyze costs to identify savings opportunities using tools like AWS Cost Explorer.


### Session-Based Recommendations
- **Embedding Management**: Precompute item embeddings (stored in Cassandra) and cache user embeddings in Redis for real-time updates.


## Tooling
- **Machine Learning**: PyTorch for model development.
- **Data Processing**: Pandas and NumPy for preprocessing.
- **Cloud Platform**: AWS for production deployment, Google Colab for initial prototyping.
- **Distributed Training**: Apache Spark for large-scale model training.
- **Data Storage**: Apache Cassandra for interactions and metadata, Amazon S3 for multimedia.
- **Caching**: Amazon ElastiCache for recommendation retrieval.
- **Load Balancing**: AWS Application Load Balancer for traffic distribution.
- **Multimedia Serving**: Amazon CloudFront for images and videos, AWS Media Services for streaming.
- **Monitoring**: Prometheus and Grafana for system health.
- **Logging**: ELK Stack for log analysis.
- **Model Management**: MLflow for experiment tracking and versioning.
- **Workflow Automation**: Apache Airflow for retraining pipelines.
- **CI/CD**: GitHub Actions for automated deployment.
- **Containerization**: Docker for packaging and consistent environments.


## Clone the repository:
   ```bash
   git clone https://github.com/gargumang411/Ecommerce_recommender_system
   ```


## Results and Evaluation
- **Test Loss (MSE)**: 0.1792
- **Test MAE**: 0.3946
- **Precision@10**: 0.0110
These results indicate training challenges with limited compute resources and in finding a good dataset for the purpose since real ecommerce interaction datasets aren't open-source. Planned enhancements aim to address these.

## Future Work
- Refine preprocessing with raw ratings or binary labels.
- Enhance model with deeper layers, alternative loss functions (e.g., BPR), and regularization.
- Implement Kafka for real-time feedback and hybrid inference.
- Deploy on AWS with A/B testing, continuous monitoring, and content-based cold-start strategies.

## System Design Diagram
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
