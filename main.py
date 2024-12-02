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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='chatbot.log',
                    filemode='a')
logger = logging.getLogger(__name__)

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

def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

try:
    df = pd.read_csv('embedded_documents.csv')
    df['ada_embedding'] = df.ada_embedding.apply(safe_eval).apply(np.array)
    print("Loaded existing embedded documents")
except FileNotFoundError:
    df = pd.DataFrame(columns=['content', 'ada_embedding'])
    print("No existing embedded documents found, starting with an empty dataframe")
except Exception as e:
    print(f"Error loading embedded documents: {e}")
    df = pd.DataFrame(columns=['content', 'ada_embedding'])
    print("Starting with an empty dataframe due to loading error")

chat_history = []

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class TextUpload(BaseModel):
    content: str

@app.post("/upload")
async def upload_text(text_upload: TextUpload):
    content = text_upload.content
    
    embedding = get_embedding(content)
    
    new_row = pd.DataFrame({'content': [content], 'ada_embedding': [embedding]})
    global df
    df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv('embedded_documents.csv', index=False)
    
    return {"message": "Text uploaded and embedded successfully"}

async def generate_response(message: str):
    global chat_history
    chat_history.append(message)
    logger.info(f"Human Question: {message}")
    print(f"Human Question: {message}")
    query_embedding = get_embedding(message)
    
    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, query_embedding))
    
    similar_docs = df.sort_values('similarities', ascending=False).head(3)
    
    context = " ".join(similar_docs['content'].tolist())
    print(f"Retrieved Context: {context}")
    system_message = """
   You are an AI assistant for (PBS)Proficient Business Service Ltd, Bahamas a company that provides Total I.T. Care™ Services. Your role is to answer questions about the company's services, policies, and general information, FAQ, Contact Info. 
 
 Greet customers, handle routine inquiries, and swiftly escalate complex issues to human support.

    Here's some important context:
    
    1. PBS specializes in comprehensive IT support and management for businesses. PBS is  a technology solutions provider with a focus on managed IT services.  

Core Business:
Managed IT Services: Their primary offering revolves around "Total I.T. Care™" which encompasses a wide range of IT needs including cybersecurity, disaster recovery, network design, and AI and ML, Custom Bots, Digital Marketing, Web & Mobile application developement services. They aim to simplify IT management for businesses so they can concentrate on their core operations.
Customer Focus: They emphasize building strong relationships with clients, understanding their needs, and providing customized solutions.
Two-Tiered Service: They offer two main service levels:
Total I.T. Care™: Remote IT management and support.
Total I.T. Care Plus™: Includes on-site IT personnel for more hands-on assistance.

Strengths:
Comprehensive Solutions: They cover a wide spectrum of IT needs, providing a one-stop shop for businesses.
Customer-Centric Approach: They prioritize understanding client requirements and tailoring solutions accordingly.
Proactive Management: They focus on preventing issues rather than just reacting to them, ensuring smooth IT operations.
Experienced Team: They highlight the expertise of their IT professionals.
Flexible Financing: They offer options to make their services more accessible to businesses.

Location:
Based in Nassau, Bahamas. This could be a strategic location for serving clients in the Caribbean and North America.

PBS Contact Information:
Phone: +1 242 397 3100 Fax: +1 242 322 1036  Email: info@pbshope.com
 
    Use the following context to answer the user's question. The context includes relevant information retrieved from the company's knowledge base, as well as the history of questions the human has asked in this conversation. All of these questions have been answered using retrieved context about the company.

    Retrieved Context: {context} you must use this context text to answer the user's question in short providing all important information talking in a smooth and natural way.

    Human's Chat History (previous questions):
    {chat_history}
    what ever the human has said so far try to analyze what they really are looking forand dont reveal it to them but influence them to ask the right question.
    Please provide accurate, helpful, and concise responses based on the given context and the user's question history. If you don't have enough information to answer a question, politely say so and offer to assist with related information you do have and ask them to email to info@pbshope.com for further details.
    Strictly make sure your response do not exceed 3-5 sentences. and make sure you respons in a natural professional way with posititvity and marketing tone.  
    """

    
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message.format(context=context, chat_history=chat_history)},
            {"role": "user", "content": message}
        ],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield f"data: {chunk.choices[0].delta.content}\n\n"
        await asyncio.sleep(0.1)  
    
    yield "data: [DONE]\n\n"

@app.get("/chat")
async def chat(request: Request):
    message = request.query_params.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Message parameter is required")
    
    return StreamingResponse(generate_response(message), media_type="text/event-stream")
