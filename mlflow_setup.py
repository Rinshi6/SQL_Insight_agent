import json
import os
import time
import uuid
from functools import wraps
from typing import Any
from dotenv import load_dotenv
load_dotenv(override=True) 
from pydantic import BaseModel


DEFAULT_TRACKING_URI = os.getenv("DEFAULT_TRACKING_URI")
DEFAULT_EXPERIMENT_NAME = "txt2sql-agent"

_MLFLOW_AVAILABLE = False


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def setup_mlflow() -> bool:
    """Configure MLflow to store runs in the local PostgreSQL mlflowdb database."""
    global _MLFLOW_AVAILABLE

    try:
        import mlflow
    except ImportError:
        _MLFLOW_AVAILABLE = False
        return False

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        f"{DEFAULT_EXPERIMENT_NAME}-{uuid.uuid4().hex}",
    )

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        print(f"MLflow observability disabled: could not connect to tracking store ({exc})")
        _MLFLOW_AVAILABLE = False
        return False

    try:
        import mlflow.langchain

        mlflow.langchain.autolog(silent=True)
    except Exception:
        pass

    _MLFLOW_AVAILABLE = True
    return True


def is_mlflow_enabled() -> bool:
    return _MLFLOW_AVAILABLE


def log_json_artifact(name: str, payload: Any) -> None:
    if not _MLFLOW_AVAILABLE:
        return

    import mlflow

    mlflow.log_dict(_to_jsonable(payload), name)


def log_text_artifact(name: str, text: Any) -> None:
    if not _MLFLOW_AVAILABLE:
        return

    import mlflow

    mlflow.log_text(str(text), name)


def observe_node(node_name: str):
    """Measure a graph node and attach compact inputs/outputs/errors to the active run."""

    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            if not _MLFLOW_AVAILABLE:
                return func(state, *args, **kwargs)

            import mlflow

            started_at = time.perf_counter()
            mlflow.log_metric(f"{node_name}_started", 1)
            log_json_artifact(f"nodes/{node_name}/input.json", state)

            try:
                result = func(state, *args, **kwargs)
            except Exception as exc:
                duration = time.perf_counter() - started_at
                mlflow.log_metric(f"{node_name}_duration_seconds", duration)
                mlflow.log_param(f"{node_name}_status", "error")
                log_text_artifact(f"nodes/{node_name}/error.txt", repr(exc))
                raise

            duration = time.perf_counter() - started_at
            mlflow.log_metric(f"{node_name}_duration_seconds", duration)
            mlflow.log_param(f"{node_name}_status", "success")
            log_json_artifact(f"nodes/{node_name}/output.json", result)
            return result

        return wrapper

    return decorator
