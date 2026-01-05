import os
import pickle
import tempfile
import numpy as np
from numpy.linalg import norm

from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace

# =====================================================
# CONFIG
# =====================================================
PKL_PATH = "student_embeddings.pkl"   # must be in same folder
MODEL_NAME = "Facenet512"
THRESHOLD = 0.58

# =====================================================
# FASTAPI APP (THIS IS VERY IMPORTANT)
# =====================================================
app = FastAPI(
    title="Face Recognition API",
    description="DeepFace FaceNet512 based Student Recognition API",
    version="1.0"
)

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def l2_normalize(x):
    return x / norm(x)


# =====================================================
# LOAD EMBEDDINGS ONCE (AT STARTUP)
# =====================================================
if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(f"{PKL_PATH} not found")

with open(PKL_PATH, "rb") as f:
    db = pickle.load(f)

print(f"✅ Loaded {len(db)} student embeddings")

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/")
def home():
    return {"status": "Face Recognition API is running"}

# =====================================================
# FACE RECOGNITION ENDPOINT
# =====================================================
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(await file.read())
        img_path = temp.name

    try:
        # Extract face embedding
        result = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            enforce_detection=True
        )

        test_emb = np.array(result[0]["embedding"])
        test_emb = l2_normalize(test_emb)

        best_match = None
        best_score = -1.0

        # Compare with database
        for name, emb in db.items():
            score = cosine_similarity(emb, test_emb)
            if score > best_score:
                best_score = score
                best_match = name

        decision = "MATCH" if best_score >= THRESHOLD else "NO MATCH"

        return {
            "student": best_match,
            "similarity": round(float(best_score), 4),
            "cosine_distance": round(float(1 - best_score), 4),
            "decision": decision
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Clean up temp file
        if os.path.exists(img_path):
            os.remove(img_path)
