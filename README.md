# Real-Time Credit Card Fraud Detection using Spark & Kafka

This project is a production-ready Big Data fraud detection system utilizing **Apache Kafka** for real-time transaction ingestion, **Apache Spark (PySpark Structured Streaming)** for live ML inference, and **Streamlit** for interactive dashboarding.

## Architecture
1. **Offline ML Training (`src/pipeline.py`)**: Loads historical CSV data, undersamples to handle extreme class imbalance, trains a Random Forest pipeline, and saves the binary model.
2. **Kafka Ingestion (`src/kafka_producer.py`)**: Simulates a live payment gateway by streaming historical records into a local Kafka topic (`transactions`).
3. **Structured Streaming Inference (`src/streaming_pipeline.py`)**: A continuous PySpark job that reads from Kafka, loads the static Random Forest model, computes fraud probability, categorizes risk (NORMAL, MEDIUM, HIGH), and dumps alerts into a local SQLite database.
4. **Live Dashboard (`app.py`)**: A Streamlit interface that continuously polls the SQLite database to map fraud occurrences and probabilities in real-time.

## Prerequisites
- Python 3.8+
- Java 8 or 11 (required for PySpark)
- Docker Desktop (for Apache Kafka & Zookeeper)

## Getting Started

### 1. Start the Kafka Cluster
Open a terminal in the project root and run:
```bash
docker-compose up -d
```
This spins up a Zookeeper and Kafka broker on `localhost:9092`.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train Model
Train the models and compile the saving of the offline pipeline:
```bash
python src/pipeline.py
```
*(This generates `saved_models/rf_model` and `metrics.json`)*.

### 4. Run the Real-Time Pipeline
You need to open **three** separate terminals to run the ecosystem.

**Terminal 1: Start the Dashboard**
```bash
streamlit run app.py
```
Keep this open in your browser.

**Terminal 2: Start the PySpark Structured Streaming Job**
*Ensure you use the spark-sql-kafka package for PySpark to talk to Kafka!*
```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 src/streaming_pipeline.py
```

**Terminal 3: Start the Payment Gateway Simulator (Kafka Producer)**
```bash
python src/kafka_producer.py
```

As the producer pushes transactions into Kafka, you will see Spark process them in Terminal 2, and the Streamlit UI in Terminal 1 will live-refresh with High and Medium risk alerts!
