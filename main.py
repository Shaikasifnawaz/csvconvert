import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import numpy as np
from fastapi.responses import StreamingResponse
import asyncio
import ast
import logging
from dotenv import load_dotenv

load_dotenv()

# Setting up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='chatbot.log',
                    filemode='a')
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Safe evaluation function for embeddings
def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except Exception as e:
        logger.error(f"Error parsing embedding: {e}")
        return []

# Loading existing embedded documents from CSV
try:
    df = pd.read_csv('embedded_documents.csv')
    df['ada_embedding'] = df.ada_embedding.apply(safe_eval).apply(np.array)
    logger.info("Loaded existing embedded documents")
except FileNotFoundError:
    df = pd.DataFrame(columns=['content', 'ada_embedding'])
    logger.info("No existing embedded documents found, starting with an empty dataframe")
except Exception as e:
    logger.error(f"Error loading embedded documents: {e}")
    df = pd.DataFrame(columns=['content', 'ada_embedding'])
    logger.info("Starting with an empty dataframe due to loading error")

chat_history = []

# Function to get embeddings
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")  # Remove newlines to prevent issues
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return np.array(embedding)

# Function to compute cosine similarity
def cosine_similarity(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0  # Return zero if either embedding is empty
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Pydantic model for uploading text
class TextUpload(BaseModel):
    content: str

# Endpoint to upload text and generate embedding
@app.post("/upload")
async def upload_text(text_upload: TextUpload):
    content = text_upload.content
    embedding = get_embedding(content)
    
    new_row = pd.DataFrame({'content': [content], 'ada_embedding': [embedding]})
    global df
    df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv('embedded_documents.csv', index=False)
    
    return {"message": "Text uploaded and embedded successfully"}

# Function to generate a response based on a message and previous chat history
async def generate_response(message: str):
    global chat_history
    chat_history.append(message)
    logger.info(f"Human Question: {message}")
    print(f"Human Question: {message}")
    query_embedding = get_embedding(message)

    # Compute cosine similarities for all stored embeddings
    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, query_embedding))

    # Retrieve top 3 most similar documents
    similar_docs = df.sort_values('similarities', ascending=False).head(3)
    context = " ".join(similar_docs['content'].tolist())
    print(f"Retrieved Context: {context}")
    
    system_message = f"""
    You are an AI assistant for PBS (Proficient Business Service Ltd), Bahamas, a company that provides Total I.T. Care™ Services. Your role is to answer questions about the company's services, policies, general information, FAQ, and Contact Info.

    Greet customers, handle routine inquiries, and escalate complex issues to human support.

    Here's some important context:
    
    1. PBS specializes in comprehensive IT support and management for businesses. PBS is a technology solutions provider focusing on managed IT services.
    2. Core Business: Managed IT Services, Cybersecurity, Disaster Recovery, Network Design, AI/ML, Custom Bots, Digital Marketing, Web & Mobile App Development.
    3. Customer Focus: Strong client relationships, customized solutions, proactive management.
    4. Location: Nassau, Bahamas.

    PBS Contact Information:
    Phone: +1 242 397 3100
    Email: info@pbshope.com

    Retrieved Context: {context}
    Human's Chat History (previous questions):
    {chat_history}

    Please answer in a professional, concise, and positive tone, using the provided context.
    """

    # Stream response generation using OpenAI GPT-3.5
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ],
        stream=True
    )

    # Stream the response to the client
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield f"data: {chunk.choices[0].delta.content}\n\n"
        await asyncio.sleep(0.1)

    yield "data: [DONE]\n\n"

# Endpoint to handle chat
@app.get("/chat")
async def chat(request: Request):
    message = request.query_params.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Message parameter is required")
    
    return StreamingResponse(generate_response(message), media_type="text/event-stream")

# Running the application with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
