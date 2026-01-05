import requests
import sys

# Test the health endpoint first
print("Testing health endpoint...")
response = requests.get("http://127.0.0.1:8000/")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test face recognition with an image
if len(sys.argv) > 1:
    img_path = sys.argv[1]
    print(f"\nTesting face recognition with: {img_path}")
    
    with open(img_path, "rb") as f:
        files = {"file": f}
        response = requests.post("http://127.0.0.1:8000/recognize", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
else:
    print("\nTo test face recognition, provide an image path:")
    print("python test_api.py <image_path>")
