# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import tensorflow as tf
import joblib
import pickle
import numpy as np
import os
import psutil
import time
import openai
from typing import Tuple, Any, Dict

# Use torch pruning utilities properly
import torch.nn.utils.prune as torch_prune

# ptflops optional
try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None

# configure openai - expects OPENAI_API_KEY in env
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key

app = FastAPI()
os.makedirs("uploaded_models", exist_ok=True)
os.makedirs("optimized_models", exist_ok=True)


@app.get("/")
async def root():
    return {"message": "AI-Efficiency-Project"}


# -------------------------
# Helper utilities
# -------------------------
def get_model_size(filepath: str) -> float:
    """Returns file size in MB."""
    return round(os.path.getsize(filepath) / (1024 * 1024), 2)


def load_model(filepath: str) -> Tuple[Any, bool]:
    """
    Loads model and detects if it's weights-only (PyTorch state_dict).
    Returns (model_or_state_dict, is_weights_only_boolean)
    """
    ext = filepath.split(".")[-1].lower()
    is_weights_only = False

    if ext in ["pt", "pth"]:
        model = torch.load(filepath, map_location="cpu")
        if isinstance(model, dict):
            is_weights_only = True
        return model, is_weights_only

    elif ext == "pkl":
        with open(filepath, "rb") as f:
            return pickle.load(f), is_weights_only

    elif ext == "joblib":
        return joblib.load(filepath), is_weights_only

    elif ext in ["h5", "hdf5"]:
        return tf.keras.models.load_model(filepath), is_weights_only

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_num_parameters(model: Any) -> int:
    """Counts trainable parameters for supported model types, returns None if not applicable."""
    try:
        if isinstance(model, torch.nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        elif isinstance(model, tf.keras.Model):
            return model.count_params()
        elif hasattr(model, "get_params"):
            # approximate for scikit-learn: number of params in estimator.get_params()
            return len(model.get_params())
        else:
            return None
    except Exception:
        return None


def get_model_type(model: Any) -> str:
    """Return model class name or a descriptive string for state_dict."""
    try:
        if isinstance(model, dict):
            return "WeightsOnly_StateDict"
        return model.__class__.__name__
    except Exception:
        return "UnknownModel"


def measure_inference_time(model: Any, sample_input: np.ndarray = None) -> float:
    """Measures inference time for a single forward pass (approx)."""
    try:
        if sample_input is None:
            # default small random vector — user should provide real shape for accurate metrics
            sample_input = np.random.rand(1, 10)

        start = time.time()

        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                inp = torch.tensor(sample_input, dtype=torch.float32)
                _ = model(inp)
        elif isinstance(model, tf.keras.Model):
            inp = tf.convert_to_tensor(sample_input, dtype=tf.float32)
            _ = model(inp, training=False)
        elif hasattr(model, "predict"):
            _ = model.predict(sample_input)
        else:
            return None

        end = time.time()
        return round(end - start, 5)
    except Exception:
        return None


def measure_memory_usage() -> float:
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def calculate_flops(model: Any) -> float:
    """Estimate FLOPs (rough). Returns None if unavailable."""
    try:
        if isinstance(model, torch.nn.Module) and get_model_complexity_info:
            # default shape for image models; may be inaccurate for other models
            macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
            flops = macs * 2
            return round(flops / 1e6, 2)  # MFLOPs
        elif isinstance(model, tf.keras.Model):
            total_flops = 0
            for layer in model.layers:
                if hasattr(layer, "count_params"):
                    try:
                        total_flops += layer.count_params() * 2
                    except Exception:
                        pass
            return round(total_flops / 1e6, 2)
        else:
            return None
    except Exception:
        return None


# -------------------------
# Efficiency and advice
# -------------------------
def compute_efficiency_score(model_info: Dict) -> float:
    baseline = {"size_mb": 100, "inference_time": 0.1, "memory_mb": 1000, "params": 10_000_000}
    try:
        size_score = max(0, 100 - (model_info.get("model_size_MB", 0) / baseline["size_mb"]) * 100)
        time_score = max(0, 100 - (model_info.get("measure_inference_time", 0) / baseline["inference_time"]) * 100)
        mem_score = max(0, 100 - (model_info.get("measure_memory_usage", 0) / baseline["memory_mb"]) * 100)
        param_score = max(0, 100 - (model_info.get("num_parameters", 0) / baseline["params"]) * 100)

        score = round((0.3 * size_score + 0.3 * time_score + 0.2 * mem_score + 0.2 * param_score), 2)
        return min(max(score, 0), 100)
    except Exception:
        return None


def compare_with_baseline(model_info: Dict) -> Dict:
    baseline = {"size_mb": 100, "inference_time": 0.1, "memory_mb": 1000, "params": 10_000_000}
    try:
        return {
            "size_vs_baseline": f"{round((model_info.get('model_size_MB', 0)/baseline['size_mb']) * 100, 2)}% of baseline size",
            "time_vs_baseline": f"{round((model_info.get('measure_inference_time', 0)/baseline['inference_time']) * 100, 2)}% of baseline speed",
            "memory_vs_baseline": f"{round((model_info.get('measure_memory_usage', 0)/baseline['memory_mb']) * 100, 2)}% of baseline memory",
            "params_vs_baseline": f"{round((model_info.get('num_parameters', 0)/baseline['params']) * 100, 2)}% of baseline parameters"
        }
    except Exception:
        return {}


def generate_suggestions(model_info: Dict, efficiency_score: float) -> list:
    suggestions = []
    try:
        if model_info.get("model_size_MB", 0) > 100:
            suggestions.append("Consider model pruning or quantization to reduce size.")
        if model_info.get("measure_inference_time", 0) and model_info.get("measure_inference_time", 0) > 0.1:
            suggestions.append("Optimize inference speed by using TorchScript, ONNX, or TensorRT.")
        if model_info.get("measure_memory_usage", 0) > 1000:
            suggestions.append("Reduce memory footprint using mixed precision, smaller batch sizes, or model simplification.")
        if (model_info.get("num_parameters") or 0) > 10_000_000:
            suggestions.append("Try knowledge distillation or switch to lightweight architectures (MobileNet, EfficientNet).")

        if efficiency_score is not None:
            if efficiency_score >= 85:
                suggestions.append("Excellent efficiency! Model is highly optimized.")
            elif efficiency_score >= 60:
                suggestions.append("Good model, but further optimization can help in production.")
            else:
                suggestions.append("Model is inefficient. Revisit architecture, compression, or deployment pipeline.")
        return suggestions
    except Exception:
        return ["Unable to generate suggestions."]


# -------------------------
# Optimization utilities (PyTorch)
# -------------------------
def quantize_model(model: torch.nn.Module) -> Tuple[Any, str]:
    """
    Applies dynamic quantization to PyTorch model (works best for linear/embedding layers).
    Returns (quantized_model, message).
    """
    try:
        if not isinstance(model, torch.nn.Module):
            return None, "Quantization only supported for PyTorch models."

        # dynamic quantization (affects supported layers like Linear, LSTM)
        q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        return q_model, "Model quantized successfully (dynamic quantization)."
    except Exception as e:
        return None, f"Error quantizing model: {e}"


def prune_model(model: torch.nn.Module, amount: float = 0.3) -> Tuple[Any, str]:
    """
    Prunes weights (L1 unstructured) in Linear layers by default.
    Returns (pruned_model, message).
    """
    try:
        if not isinstance(model, torch.nn.Module):
            return None, "Pruning only supported for PyTorch models."

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # prune weight tensor in-place
                torch_prune.l1_unstructured(module, name="weight", amount=amount)
        return model, f"Model pruned by {amount * 100:.0f}% on applicable layers."
    except Exception as e:
        return None, f"Error pruning model: {e}"


def export_model(model: Any, export_name: str = "optimized_model.pt") -> Tuple[str, str]:
    """Exports model to optimized_models folder and returns (path, message)."""
    try:
        save_path = os.path.join("optimized_models", export_name)
        torch.save(model, save_path)
        return save_path, "Model exported successfully."
    except Exception as e:
        return None, f"Error exporting model: {e}"


# -------------------------
# OpenAI helper for smarter tips (optional)
# -------------------------
def generate_ai_tips(model_info: Dict) -> str:
    """
    Uses OpenAI to generate tailored quantization/pruning tips.
    Falls back to simple suggestions if API not configured.
    """
    if not openai_api_key:
        # fallback short instruction
        tips = "\n".join(generate_suggestions(model_info, compute_efficiency_score(model_info)))
        return f"(No OpenAI key) Quick tips:\n{tips}"

    # Compose a prompt
    prompt = (
        "You are an expert ML engineer. Given the following model metrics, provide concise, practical steps "
        "to quantize and prune the model for production (include commands or library names where helpful):\n\n"
        + "\n".join([f"{k}: {v}" for k, v in model_info.items()]) +
        "\n\nReturn a short actionable checklist."
    )

    try:
        # Use ChatCompletion-style call if available
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change if needed / available
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        text = response["choices"][0]["message"]["content"].strip()
        return text
    except Exception:
        # fallback to basic tips
        tips = "\n".join(generate_suggestions(model_info, compute_efficiency_score(model_info)))
        return f"(OpenAI call failed) Quick tips:\n{tips}"


# -------------------------
# Endpoints
# -------------------------
@app.post("/uploadmodel")
async def upload_model(file: UploadFile = File(...)):
    try:
        valid_types = ["pkl", "joblib", "pt", "pth", "h5", "hdf5"]
        ext = file.filename.split(".")[-1].lower()
        if ext not in valid_types:
            raise HTTPException(status_code=400, detail="Invalid file type")

        save_path = os.path.join("uploaded_models", file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        model, is_weights_only = load_model(save_path)

        if is_weights_only:
            # partial analysis from state_dict
            num_layers = len(model.keys())
            total_tensors = sum(v.numel() for v in model.values())
            approx_mb = round(total_tensors * 4 / (1024 * 1024), 2)
            return {
                "filename": file.filename,
                "model_size_MB": get_model_size(save_path),
                "weights_layers_detected": num_layers,
                "total_weights": total_tensors,
                "approx_memory_of_weights_MB": approx_mb,
                "measure_memory_usage": measure_memory_usage(),
                "message": "Weights-only file detected — architecture unavailable but partial info extracted."
            }

        # full model analysis
        model_info = {
            "filename": file.filename,
            "model_size_MB": get_model_size(save_path),
            "model_type": get_model_type(model),
            "num_parameters": get_num_parameters(model),
            "measure_inference_time": measure_inference_time(model),
            "measure_memory_usage": measure_memory_usage(),
            "calculate_flops": calculate_flops(model),
        }

        efficiency_score = compute_efficiency_score(model_info)
        comparison = compare_with_baseline(model_info)
        suggestions = generate_suggestions(model_info, efficiency_score)
        ai_tips = generate_ai_tips(model_info)

        model_info.update({
            "efficiency_score": efficiency_score,
            "comparison_with_baseline": comparison,
            "suggestions": suggestions,
            "ai_tips": ai_tips,
            "message": "Model analyzed successfully"
        })

        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_model(file: UploadFile = File(...)):
    """
    Accepts a PyTorch full model file (.pt/.pth saved with architecture). 
    Applies quantization + pruning and returns exported optimized model path + messages.
    """
    try:
        save_path = os.path.join("uploaded_models", file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        model, is_weights_only = load_model(save_path)
        if is_weights_only or not isinstance(model, torch.nn.Module):
            raise HTTPException(status_code=400, detail="Optimization requires a full PyTorch model (not weights-only).")

        # Quantize
        quantized_model, q_msg = quantize_model(model)

        # Prune (we run prune on a copy to avoid destroying original; prune modifies in place)
        pruned_model = None
        try:
            # attempt to deepcopy model for pruning step to allow independent operations
            import copy
            model_copy = copy.deepcopy(model)
            pruned_model, p_msg = prune_model(model_copy, amount=0.2)
        except Exception:
            # fallback: prune original
            pruned_model, p_msg = prune_model(model, amount=0.2)

        # Choose a model to export (prioritize quantized, then pruned)
        chosen = quantized_model if quantized_model is not None else pruned_model

        if chosen is None:
            raise HTTPException(status_code=500, detail="Optimization failed (no optimized model produced).")

        export_path, e_msg = export_model(chosen, export_name=f"optimized_{file.filename}")

        return {
            "quantization": q_msg,
            "pruning": p_msg,
            "export_message": e_msg,
            "export_path": export_path
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))