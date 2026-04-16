import json
import time
import pandas as pd
import os
from kafka import KafkaProducer

KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'transactions'

def create_producer():
    print(f"Connecting to Kafka at {KAFKA_BROKER}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            retries=5
        )
        print("Connected successfully!")
        return producer
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        print("Is Docker/Kafka running?")
        return None

def stream_data(producer, data_path):
    print(f"Starting to stream data from {data_path} to topic '{KAFKA_TOPIC}'...")
    
    # Read the dataset (if it's Huge, we still load it into pandas here for simplicity of simulation)
    # Dropping the 'Class' label to simulate real-world unseen data, though we might keep it 
    # to evaluate streaming accuracy later if needed. But in production, 'Class' is unknown.
    # We will actually keep 'Class' but as a separate key in the JSON so our UI can show if it was ACTUALLY fraud.
    
    df = pd.read_csv(data_path)
    
    # Let's shuffle so we get frauds mixed in randomly if using synthetic.
    # Real dataset has frauds dispersed, but let's shuffle anyway to make streaming more dynamic.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    count = 0
    for _, row in df.iterrows():
        transaction = row.to_dict()
        
        # Send to Kafka
        producer.send(KAFKA_TOPIC, value=transaction)
        count += 1
        
        # Print progress
        if count % 100 == 0:
            print(f"Sent {count} transactions...")
            producer.flush()
            
        # Simulate real-time delay (e.g., 50 milliseconds per transaction)
        # Tweak this to make the stream faster or slower
        time.sleep(0.05)

if __name__ == "__main__":
    data_paths = ["data/creditcard.csv", "data/synthetic_creditcard.csv"]
    target_path = None
    for path in data_paths:
        if os.path.exists(path):
            target_path = path
            break
            
    if not target_path:
        print("Dataset not found! Please run src/generate_synthetic.py first.")
        exit(1)
        
    producer = create_producer()
    if producer:
        try:
            stream_data(producer, target_path)
        except KeyboardInterrupt:
            print("\nStreaming stopped by user.")
        finally:
            producer.close()
