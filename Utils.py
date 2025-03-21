import os
from dotenv import load_dotenv
import psycopg2
from deepface import DeepFace
import ast

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_NAME = "VGG-Face"  # Pre-trained model for face recognition

# ---------------------------
# Configuration & Connection
# ---------------------------
if not DATABASE_URL:
    raise ValueError("Please set the DATABASE_URL environment variable with your Neon connection string.")

def get_db_connection():
    """Return a new connection to the Neon DB."""
    return psycopg2.connect(DATABASE_URL)

def initialize_db():
    """
    Create the pgvector extension (if not already enabled) and the face_embeddings table.
    The table stores user_id, image_path, and embedding (using pgvector with dimension 4096).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    # Enable the pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Create the table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id SERIAL PRIMARY KEY,
        user_id INT,
        image_path TEXT,
        embedding vector(4096)
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()

# ---------------------------
# Function 1: Register a Face
# ---------------------------
def register_face(image_path, user_id):
    """
    Extract a face embedding from the given image using DeepFace and insert it into the database.
    
    Args:
        image_path (str): Path to the face image.
        user_id (int): User identifier.
        
    Returns:
        int: The new record ID in the database.
    """
    # Extract embedding using DeepFace (using VGG-Face as default; outputs 4096-dimensional vector)
    embedding_objs = DeepFace.represent(img_path=image_path, model_name=MODEL_NAME)
    if not embedding_objs:
        raise ValueError("No face was detected in the image.")
    
    embedding = embedding_objs[0]["embedding"]
    # Convert the list of floats into a PostgreSQL vector literal (e.g., "[0.1, 0.2, ...]")
    embedding = "[" + ", ".join(map(str, embedding)) + "]"
    
    conn = get_db_connection()
    cur = conn.cursor()
    insert_query = """
    INSERT INTO face_embeddings (user_id, embedding)
    VALUES (%s, %s)
    RETURNING id;
    """
    cur.execute(insert_query, (user_id, embedding))
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return new_id

# ---------------------------
# Function 2: Verify Face by User ID
# ---------------------------
def verify_face_by_user(query_image_path, user_id):
    """
    Retrieve the registered face record for a given user_id, then verify whether the query face matches 
    the registered face using DeepFace.verify().
    
    Args:
        query_image_path (str): Path to the query face image.
        user_id (int): The user ID to look up in the DB.
    
    Returns:
        dict: Verification result for the candidate that matches.
              Includes: record_id, user_id, verification_distance, and threshold.
              Returns None if no match is found or an error occurs.
    """
    # Retrieve the candidate based on user_id
    conn = get_db_connection()
    cur = conn.cursor()
    select_query = """
    SELECT id, embedding
    FROM face_embeddings
    WHERE user_id = %s;
    """
    cur.execute(select_query, (user_id,))
    candidate = cur.fetchone()
    cur.close()
    conn.close()
    
    if not candidate:
        print(f"No face record found for user_id {user_id}.")
        return None
    
    record_id, candidate_embedding = candidate

    candidate_embedding = ast.literal_eval(candidate_embedding)  # Convert the string back to a list of floats
    if isinstance(candidate_embedding, list):
        candidate_embedding = [float(i) for i in candidate_embedding]

    try:
        # Verify using DeepFace.verify
        result = DeepFace.verify(
            img1_path=candidate_embedding,
            img2_path=query_image_path,
            model_name=MODEL_NAME,
            silent=False
        )
    except ValueError as e:
        print(f"Face detection failed for record {record_id}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error for record {record_id}: {str(e)}")
        return None
    
    if result.get("verified"):
        return {
            "record_id": record_id,
            "user_id": user_id,
            "verification_distance": result.get("distance"),
            "threshold": result.get("threshold")
        }
    
    return None