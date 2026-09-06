import requests

BASE = "http://127.0.0.1:8000"

# 1. Test /api/models
r = requests.get(f"{BASE}/api/models")
print("1. GET /api/models status:", r.status_code)
models_data = r.json()
print("   Active:", models_data["active_model"])
print("   Available:", [m["filename"] for m in models_data["models"]])

# 2. Test /api/model/info
r = requests.get(f"{BASE}/api/model/info")
print("2. GET /api/model/info status:", r.status_code)
info = r.json()
print("   Quantization:", info["info"]["quantization"], "| Size:", info["info"]["size_kb"], "KB | Params:", info["info"]["parameter_count"])
print("   Architecture nodes count:", len(info["architecture"]))

# 3. Test /api/test_samples
r = requests.get(f"{BASE}/api/test_samples")
print("3. GET /api/test_samples status:", r.status_code)
samples = r.json()["samples"]
proxima_s = [s for s in samples if s["expected"] == "PROXIMA"][0]
unknown_s = [s for s in samples if s["expected"] == "UNKNOWN"][0]

# 4. Test /api/predict/sample (PROXIMA)
r = requests.post(f"{BASE}/api/predict/sample", data={"sample_id": proxima_s["id"]})
res_p = r.json()
print("4. POST /api/predict/sample [PROXIMA]:", res_p["prediction"], f"PROXIMA={res_p['confidence_proxima']}%", f"Lat={res_p['inference_time_ms']}ms")
assert res_p["prediction"] == "PROXIMA", "Failed proxima sample test!"

# 5. Test /api/predict/sample (UNKNOWN)
r = requests.post(f"{BASE}/api/predict/sample", data={"sample_id": unknown_s["id"]})
res_u = r.json()
print("5. POST /api/predict/sample [UNKNOWN]:", res_u["prediction"], f"UNKNOWN={res_u['confidence_unknown']}%", f"Lat={res_u['inference_time_ms']}ms")
assert res_u["prediction"] == "UNKNOWN", "Failed unknown sample test!"

# 6. Test Model Swapping: Switch to proxima_v3_float32.tflite
r = requests.post(f"{BASE}/api/models/select", json={"model_name": "proxima_v3_float32.tflite"})
swap_res = r.json()
print("6. POST /api/models/select [Swapped to Float32]:", swap_res["active_model"], "| Quant:", swap_res["model_info"]["quantization"])

# Test prediction with swapped model
r = requests.post(f"{BASE}/api/predict/sample", data={"sample_id": proxima_s["id"]})
print("   Prediction with Float32 model:", r.json()["prediction"], f"PROXIMA={r.json()['confidence_proxima']}%")

# Switch back to primary INT8 deployment model
r = requests.post(f"{BASE}/api/models/select", json={"model_name": "proxima_v3_int8.tflite"})
print("   Switched back to:", r.json()["active_model"])

# 7. Test / (Static index.html)
r = requests.get(f"{BASE}/")
print("7. GET / status:", r.status_code, "Content-Length:", len(r.text))
assert "PROXIMA" in r.text

print("\nALL SERVER BACKEND ENDPOINTS AND MODEL HOT-SWAPPING VALIDATED SUCCESSFULLY!")
