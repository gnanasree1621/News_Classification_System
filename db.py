"""
db.py — MongoDB Atlas integration for the Automated News Classification System.

Design principles:
- A single MongoClient is shared across all Streamlit reruns via @st.cache_resource.
- Credentials are read exclusively from st.secrets (Streamlit Cloud) or environment
  variables (local development). Credentials and URIs are never logged or exposed.
- All public functions return (success: bool, payload) tuples. On success the payload
  is a list or dict; on failure it is a human-readable error string.
- Missing MONGODB_URI is surfaced as a clear configuration message instead of silently
  falling back to localhost when Atlas is the intended target.
"""

import logging
import os
from datetime import datetime, timezone

import pymongo
import streamlit as st
from pymongo.errors import (
    ConfigurationError,
    ConnectionFailure,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "predictions"


# ---------------------------------------------------------------------------
# Configuration helpers (never log the values they return)
# ---------------------------------------------------------------------------

def _get_uri() -> str:
    """Read MONGODB_URI from st.secrets then env. Returns empty string if missing."""
    try:
        val = st.secrets["MONGODB_URI"]
        if val:
            return str(val).strip()
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    return os.getenv("MONGODB_URI", "").strip()


def _get_db_name() -> str:
    """Read MONGODB_DB from st.secrets then env, defaulting to 'news_classifier'."""
    try:
        val = st.secrets["MONGODB_DB"]
        if val:
            return str(val).strip()
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    return os.getenv("MONGODB_DB", "news_classifier").strip() or "news_classifier"


# ---------------------------------------------------------------------------
# Shared MongoClient — one per Streamlit worker process
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_shared_client():
    """
    Create and cache a single MongoClient for the lifetime of the Streamlit
    worker process.  Returns (client, error_message); error_message is None on
    success.  The URI is never included in log output.
    """
    uri = _get_uri()
    if not uri:
        return None, (
            "MONGODB_URI is not configured. "
            "Add it to Streamlit Secrets (Streamlit Cloud) or set the "
            "MONGODB_URI environment variable for local development."
        )
    try:
        client = pymongo.MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
        )
        return client, None
    except ConfigurationError as exc:
        logger.error("MongoDB ConfigurationError: %s", exc)
        return None, (
            "The MONGODB_URI connection string is malformed. "
            "Check the value in your Streamlit Secrets."
        )
    except Exception as exc:
        logger.error("Unexpected error creating MongoClient: %s", type(exc).__name__)
        return None, f"Could not create MongoDB client: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# Public connection helpers
# ---------------------------------------------------------------------------

def check_connection(timeout_ms: int = 3000):
    """
    Verify connectivity by running the admin 'ping' command.

    Returns:
        (True,  "MongoDB Connected")              — success
        (False, <human-readable error message>)   — failure
    """
    client, init_error = _get_shared_client()
    if client is None:
        return False, init_error or "MongoDB client could not be initialised."

    try:
        client.admin.command("ping", serverSelectionTimeoutMS=timeout_ms)
        return True, "MongoDB Connected"
    except ServerSelectionTimeoutError:
        return False, (
            "MongoDB Atlas not reachable (timeout). "
            "Check that the cluster is active and the IP Access List includes "
            "Streamlit Cloud (or set 0.0.0.0/0 for testing)."
        )
    except ConnectionFailure:
        return False, "MongoDB connection failed. Check network and Atlas firewall settings."
    except OperationFailure as exc:
        logger.error("MongoDB auth failure: code=%s", getattr(exc, "code", "?"))
        return False, (
            "MongoDB authentication failed. "
            "Verify the database username and password in Streamlit Secrets."
        )
    except ConfigurationError:
        return False, (
            "The MONGODB_URI connection string is malformed. "
            "Check the value in your Streamlit Secrets."
        )
    except Exception as exc:
        logger.error("MongoDB ping error: %s", type(exc).__name__)
        return False, f"MongoDB connection error ({type(exc).__name__})"


def get_database():
    """
    Return the database handle.  Raises RuntimeError when the client has not
    been initialised (e.g. missing MONGODB_URI) so that callers can distinguish
    configuration problems from transient network errors.
    """
    client, init_error = _get_shared_client()
    if client is None:
        raise RuntimeError(init_error or "MongoDB client is not initialised.")
    return client[_get_db_name()]


def get_predictions_collection():
    """
    Return the 'predictions' collection, creating a background index on
    created_at for sort performance.  Index failures are silently ignored
    because they are non-critical.
    """
    db = get_database()
    collection = db[COLLECTION_NAME]
    try:
        collection.create_index("created_at", background=True)
    except Exception:
        pass
    return collection


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def save_prediction(
    headline: str,
    description: str,
    category: str,
    confidence: float,
    probabilities: dict,
):
    """
    Insert a prediction document into MongoDB.

    Parameters
    ----------
    confidence : float
        Confidence as a percentage (0–100), e.g. 98.90.

    Returns
    -------
    (True,  document_dict)   — insertion confirmed via inserted_id
    (False, error_message)   — insertion failed; ML result remains valid
    """
    try:
        collection = get_predictions_collection()
        now = datetime.now(timezone.utc)
        doc = {
            "headline": headline.strip(),
            "description": description.strip(),
            "category": str(category),
            "confidence": round(float(confidence), 2),   # percentage: 0–100
            "probabilities": (
                {k: float(v) for k, v in probabilities.items()} if probabilities else {}
            ),
            "created_at": now,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        result = collection.insert_one(doc)
        if not result.inserted_id:
            return False, "MongoDB insert_one returned no inserted_id."
        doc["_id"] = str(result.inserted_id)
        return True, doc

    except (ServerSelectionTimeoutError, ConnectionFailure):
        return False, (
            "Could not reach MongoDB Atlas. "
            "The prediction was NOT saved. Check your cluster and IP Access List."
        )
    except OperationFailure as exc:
        logger.error("MongoDB OperationFailure on insert: code=%s", getattr(exc, "code", "?"))
        return False, (
            "MongoDB rejected the insert (authentication or permissions error). "
            "Verify the database user has readWrite access to 'news_classifier'."
        )
    except PyMongoError as exc:
        logger.error("PyMongoError on insert: %s", type(exc).__name__)
        return False, f"Database error while saving prediction ({type(exc).__name__})."
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        logger.error("Unexpected error in save_prediction: %s", type(exc).__name__)
        return False, f"Unexpected error while saving prediction ({type(exc).__name__})."


def get_predictions():
    """
    Retrieve all prediction documents sorted newest-first.

    Returns
    -------
    (True,  list[dict])      — success; empty list when no records exist
    (False, error_message)   — failure; do NOT iterate the payload
    """
    try:
        collection = get_predictions_collection()
        docs = []
        for doc in collection.find({}).sort("created_at", -1):
            doc["_id"] = str(doc["_id"])
            if "created_at" in doc and isinstance(doc["created_at"], datetime):
                doc["timestamp_str"] = doc["created_at"].strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                doc["timestamp_str"] = doc.get("timestamp", "N/A")
            docs.append(doc)
        return True, docs

    except (ServerSelectionTimeoutError, ConnectionFailure):
        return False, (
            "Could not reach MongoDB Atlas. "
            "Check your cluster status and IP Access List."
        )
    except OperationFailure as exc:
        logger.error("MongoDB OperationFailure on find: code=%s", getattr(exc, "code", "?"))
        return False, (
            "MongoDB rejected the query (authentication or permissions error). "
            "Verify the database user has read access to 'news_classifier'."
        )
    except PyMongoError as exc:
        logger.error("PyMongoError on find: %s", type(exc).__name__)
        return False, f"Database error while retrieving predictions ({type(exc).__name__})."
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        logger.error("Unexpected error in get_predictions: %s", type(exc).__name__)
        return False, f"Unexpected error while retrieving predictions ({type(exc).__name__})."


def clear_predictions():
    """
    Delete all documents from the predictions collection.

    Returns
    -------
    (True,  deleted_count)   — success
    (False, error_message)   — failure
    """
    try:
        collection = get_predictions_collection()
        result = collection.delete_many({})
        return True, result.deleted_count

    except (ServerSelectionTimeoutError, ConnectionFailure):
        return False, "Could not reach MongoDB Atlas. History was not cleared."
    except OperationFailure as exc:
        logger.error("MongoDB OperationFailure on delete: code=%s", getattr(exc, "code", "?"))
        return False, (
            "MongoDB rejected the delete (authentication or permissions error). "
            "Verify the database user has readWrite access to 'news_classifier'."
        )
    except PyMongoError as exc:
        logger.error("PyMongoError on delete: %s", type(exc).__name__)
        return False, f"Database error while clearing predictions ({type(exc).__name__})."
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        logger.error("Unexpected error in clear_predictions: %s", type(exc).__name__)
        return False, f"Unexpected error while clearing predictions ({type(exc).__name__})."


def get_kpi_metrics() -> dict:
    """
    Return dashboard KPI metrics computed from saved predictions.

    Always returns a dict with:
        total_predictions : int
        latest_category   : str
        avg_confidence    : float  (percentage, e.g. 87.3)
        db_connected      : bool
    """
    success, result = get_predictions()
    if not success or not isinstance(result, list) or not result:
        return {
            "total_predictions": 0,
            "latest_category": "—",
            "avg_confidence": 0.0,
            "db_connected": success,
        }

    total = len(result)
    latest_cat = result[0].get("category", "—")
    conf_values = [float(doc.get("confidence", 0.0)) for doc in result]
    avg_conf = sum(conf_values) / total if total > 0 else 0.0

    return {
        "total_predictions": total,
        "latest_category": latest_cat,
        "avg_confidence": round(avg_conf, 1),
        "db_connected": True,
    }

