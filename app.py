from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import psycopg2
import pgvector
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import re  # Using regex instead of nltk

# Initialize FastAPI
app = FastAPI()

# PostgreSQL connection
DATABASE_URL = "postgresql://neondb_owner:npg_q7ZLBiDPh2sF@ep-shiny-frost-a5vcxdhs-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        filename TEXT,
        content TEXT,
        embedding VECTOR(384) -- Adjusted to 384 dimensions
    )
""")
conn.commit()

# Load local embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Serve HTML templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve the frontend
@app.get("/")
def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API Request Model
class QueryRequest(BaseModel):
    query: str

def split_into_sentences(text):
    """Split text into sentences using regex instead of nltk."""
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

@app.post("/chat/")
def chatbot_conversation(request: QueryRequest):
    """Handles user queries and fetches relevant responses from the database."""
    user_query = request.query
    print("Received query:", user_query)
    
    # Generate embedding for the query
    query_embedding = model.encode(user_query).tolist()
    print("Generated embedding:", query_embedding)
    
    # Find the closest match using pgvector similarity search
    cur.execute("""
        SELECT filename, content, embedding <=> %s::vector AS similarity 
        FROM embeddings 
        ORDER BY similarity ASC 
        LIMIT 1
    """, (query_embedding,))
    result = cur.fetchone()
    
    if result:
        filename, content, similarity = result
        
        # Tokenize content into sentences using regex
        sentences = split_into_sentences(content)
        
        # Find the most relevant sentence
        sentence_embeddings = model.encode(sentences)
        similarity_scores = [model.similarity(query_embedding, sent_emb) for sent_emb in sentence_embeddings]
        most_relevant_idx = similarity_scores.index(max(similarity_scores))
        
        # Get 2-3 sentences around the most relevant one
        start_idx = max(0, most_relevant_idx - 1)
        end_idx = min(len(sentences), most_relevant_idx + 2)
        relevant_text = " ".join(sentences[start_idx:end_idx])
        
        response_text = f"Closest match: {filename} (Similarity Score: {similarity:.4f})\nRelevant Content: {relevant_text}"
    else:
        response_text = "No relevant match found."
    
    print("Response to send:", response_text)
    return {"response": response_text}
