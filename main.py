import os
from fastapi import FastAPI, UploadFile, File, Form
from Utils import register_face, verify_face_by_user, initialize_db

app = FastAPI()

initialize_db() #Init DB

@app.get("/")
def read_root():
    return {"message": "Welcome to the Face-ID API"}

@app.post("/register_face")
async def register_face_route(image: UploadFile = File(...), user_id: int = Form(...)):
    # Create a temporary file path in current directory
    image_path = os.path.join(os.getcwd(), image.filename)
    try:
        # Save the uploaded file
        with open(image_path, "wb") as buffer:
            buffer.write(await image.read())
        
        # Process the image
        new_id = register_face(image_path, user_id)
        
        return {"message": f"Face registered with ID: {new_id}"}
    finally:
        # Clean up the temporary file
        if os.path.exists(image_path):
            os.remove(image_path)


@app.post("/find_similar_faces")
async def verify_face_by_user_route(image: UploadFile = File(...), user_id: int = Form(...)):
    # Create a temporary file path in current directory
    image_path = os.path.join(os.getcwd(), image.filename)
    try:
        # Save the uploaded file
        with open(image_path, "wb") as buffer:
            buffer.write(await image.read())
        
        # Process the image
        similar_faces = verify_face_by_user(image_path, user_id)
        
        return {"similar_faces": similar_faces}
    finally:
        # Clean up the temporary file
        if os.path.exists(image_path):
            os.remove(image_path)