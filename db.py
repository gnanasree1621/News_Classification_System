import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, PyMongoError

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB", "news_classifier")
COLLECTION_NAME = "predictions"


def get_client(timeout_ms=3000):
    """
    Returns a MongoClient instance with specified server selection timeout.
    """
    return pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=timeout_ms)


def check_connection(timeout_ms=2000):
    """
    Verifies if MongoDB is reachable.
    Returns tuple: (is_connected: bool, message: str)
    """
    try:
        client = get_client(timeout_ms=timeout_ms)
        client.admin.command('ping')
        return True, "MongoDB Connected"
    except (ServerSelectionTimeoutError, ConnectionFailure):
        return False, f"MongoDB is not running or cannot be reached at {MONGODB_URI}."
    except Exception as e:
        return False, f"MongoDB connection error: {str(e)}"


def get_database():
    """
    Returns the news_classifier database instance.
    Raises exception if MongoDB is unavailable.
    """
    client = get_client()
    client.admin.command('ping')
    return client[MONGODB_DB_NAME]


def get_predictions_collection():
    """
    Returns the predictions collection from MongoDB.
    Ensures index on created_at for fast query ordering.
    """
    db = get_database()
    collection = db[COLLECTION_NAME]
    try:
        collection.create_index("created_at", background=True)
    except Exception:
        pass
    return collection


def save_prediction(headline: str, description: str, category: str, confidence: float, probabilities: dict):
    """
    Saves a news prediction document to MongoDB.
    
    Document schema:
    {
        "headline": str,
        "description": str,
        "category": str,
        "confidence": float,
        "probabilities": dict,
        "created_at": datetime (UTC),
        "timestamp": str
    }
    
    Returns tuple: (success: bool, document_or_error_msg)
    """
    try:
        collection = get_predictions_collection()
        now = datetime.now(timezone.utc)
        doc = {
            "headline": headline.strip(),
            "description": description.strip(),
            "category": str(category),
            "confidence": round(float(confidence), 2),
            "probabilities": {k: float(v) for k, v in probabilities.items()} if probabilities else {},
            "created_at": now,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
        }
        res = collection.insert_one(doc)
        doc["_id"] = str(res.inserted_id)
        return True, doc
    except (ServerSelectionTimeoutError, ConnectionFailure, PyMongoError):
        return False, "MongoDB connection unavailable. Please make sure MongoDB is running."
    except Exception as e:
        return False, f"Failed to save prediction to MongoDB: {str(e)}"


def get_predictions():
    """
    Retrieves prediction documents from MongoDB sorted by newest first (created_at descending).
    Returns tuple: (success: bool, list_of_docs_or_error_msg)
    """
    try:
        collection = get_predictions_collection()
        cursor = collection.find({}).sort("created_at", -1)
        docs = []
        for d in cursor:
            d["_id"] = str(d["_id"])
            if "created_at" in d and isinstance(d["created_at"], datetime):
                d["timestamp_str"] = d["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            else:
                d["timestamp_str"] = d.get("timestamp", "N/A")
            docs.append(d)
        return True, docs
    except (ServerSelectionTimeoutError, ConnectionFailure, PyMongoError):
        return False, "MongoDB connection unavailable. Please make sure MongoDB is running."
    except Exception as e:
        return False, f"Failed to retrieve predictions: {str(e)}"


def clear_predictions():
    """
    Deletes all documents from news_classifier.predictions collection.
    Returns tuple: (success: bool, deleted_count_or_error_msg)
    """
    try:
        collection = get_predictions_collection()
        res = collection.delete_many({})
        return True, res.deleted_count
    except (ServerSelectionTimeoutError, ConnectionFailure, PyMongoError):
        return False, "MongoDB connection unavailable. Please make sure MongoDB is running."
    except Exception as e:
        return False, f"Failed to clear predictions: {str(e)}"


def get_kpi_metrics():
    """
    Retrieves summary KPI metrics directly from stored MongoDB predictions.
    Returns dict with keys: total_predictions, latest_category, avg_confidence
    """
    success, result = get_predictions()
    if not success or not isinstance(result, list) or not result:
        return {
            "total_predictions": 0,
            "latest_category": "—",
            "avg_confidence": 0.0
        }
    
    total = len(result)
    latest_cat = result[0].get("category", "—")
    conf_values = [float(d.get("confidence", 0.0)) for d in result]
    avg_conf = sum(conf_values) / total if total > 0 else 0.0
    
    return {
        "total_predictions": total,
        "latest_category": latest_cat,
        "avg_confidence": round(avg_conf, 1)
    }
