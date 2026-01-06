import os
import pickle
import tempfile
import numpy as np
from numpy.linalg import norm

from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace

# =====================================================
# PATH CONFIG (VERY IMPORTANT FOR DOCKER)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "student_embeddings.pkl")

MODEL_NAME = "Facenet512"
THRESHOLD = 0.58

# =====================================================
# FASTAPI APP
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
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def l2_normalize(x):
    return x / norm(x)


# =====================================================
# LOAD EMBEDDINGS ONCE (SAFE)
# =====================================================
if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(f"❌ student_embeddings.pkl not found at {PKL_PATH}")

with open(PKL_PATH, "rb") as f:
    db = pickle.load(f)

print(f"✅ Loaded {len(db)} student embeddings")

# =====================================================
# HEALTH CHECK (RENDER FRIENDLY)
# =====================================================
@app.get("/")
def health():
    return {
        "status": "running",
        "model": MODEL_NAME,
        "students_loaded": len(db)
    }

# =====================================================
# FACE RECOGNITION ENDPOINT
# =====================================================
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    img_path = None

    try:
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            img_path = tmp.name

        # Extract embedding (lazy call)
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
        for name, db_emb in db.items():
            score = cosine_similarity(db_emb, test_emb)
            if score > best_score:
                best_score = score
                best_match = name

        decision = "MATCH" if best_score >= THRESHOLD else "NO MATCH"

        return {
            "student": best_match,
            "similarity": round(best_score, 4),
            "cosine_distance": round(1 - best_score, 4),
            "decision": decision
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Always clean temp file
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
