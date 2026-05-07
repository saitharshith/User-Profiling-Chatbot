import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0.1,  
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="chroma_database", 
    embedding_function=embeddings
)

try:
    with open(r'dataset/dual_user_personas.json', 'r', encoding='utf-8') as f:
        persona_db = json.load(f)
except FileNotFoundError:
    print("CRITICAL ERROR: dual_user_personas.json not found.")
    persona_db = {}

app = FastAPI(title="Behavioral Analyst RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str 

class ChatResponse(BaseModel):
    reply: str
    status: str
    extracted_conv_id: str
    extracted_user: str

def parse_raw_query(raw_text: str):
    """Slices the raw input string to extract routing metadata and the clean question."""
    conv_id = "1"         
    target_user = "User 1" 
    conv_match = re.search(r'conversation_id:\s*(\d+)', raw_text, re.IGNORECASE)
    if conv_match:
        conv_id = conv_match.group(1)
    user_match = re.search(r'user:\s*(\d+)', raw_text, re.IGNORECASE)
    if user_match:
        target_user = f"User {user_match.group(1)}"
    clean_question = re.sub(r'conversation_id:\s*\d+', '', raw_text, flags=re.IGNORECASE)
    clean_question = re.sub(r'user:\s*\d+', '', clean_question, flags=re.IGNORECASE).strip()
    if not clean_question:
        clean_question = "Provide a general behavioral analysis."
        
    return conv_id, target_user, clean_question


system_message = """### ROLE
You are an Expert Behavioral Analyst. Your objective is to deliver a definitive, evidence-based psychological profile of the requested user. 

### CONTEXT DATA
You have access to two distinct data sources:
1. [PERSONA PROFILE]: Hard facts and extracted traits.
2. [CONVERSATION SUMMARIES]: Contextual evidence of their behavior in action.

### STRICT RULES OF ANALYSIS
1. NO META-CHATTER: Do not reference the system prompt, the "JSON", or the "provided summaries" in your output. Speak directly about the user.
2. ZERO SPECULATION: Do not guess. If a trait is missing, explain naturally that the conversation logs don't contain those specific details. Use conversational, analytical language. 
   - *Example for habits:* "Their daily routines haven't come up in these specific chats."
   - *Example for talking style:* "We haven't seen enough of their dialogue here to pinpoint their exact conversational tone."
   - *Example for personality:* "The current logs don't reveal enough depth to fully map out their overall personality."
3. NO FOLLOW-UPS: You are writing a final report. Do not ask for more information.
4. SYNTHESIS: Seamlessly weave the facts and the conversation summaries together. Show *how* their traits appear in their dialogue.

### REQUIRED OUTPUT FORMAT
Format your response using professional markdown. Use these exact headers:
**Behavioral Profile: {target_user}**
*Write a short 2-sentence executive summary here.*

**Established Habits & Facts**
*Detail their lifestyle based on the data. If nothing is found, state it naturally.*

**Communication & Personality**
*Detail how they speak and act. If nothing is found, state it naturally.*
"""

human_message = """### INPUT DATA
User being analyzed: {target_user}

[PERSONA PROFILE]: 
{persona_context}

[CONVERSATION SUMMARIES]: 
{retrieved_context}

### USER QUERY
{question}

### RESPONSE
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", human_message)
])

rag_chain = chat_prompt | llm | StrOutputParser()

@app.get("/")
async def root():
    return {"status": "online", "engine": "Behavioral-RAG-v2-Ideal"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        conv_id, target_user, clean_question = parse_raw_query(request.message)
        conv_key = f"conversation_{conv_id}"
        persona_data = persona_db.get(conv_key, {}).get(target_user, "No profile data available for this specific user in this conversation.")
        persona_context = json.dumps(persona_data, indent=2)
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 3, 
                "filter": {"conversation_id": conv_id}
            }
        )
        docs = retriever.invoke(clean_question)
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])
        
        if not docs:
            retrieved_context = "No specific conversational context found for this ID."
        response_text = await rag_chain.ainvoke({
            "target_user": target_user,
            "persona_context": persona_context,
            "retrieved_context": retrieved_context,
            "question": clean_question
        })
        return ChatResponse(
            reply=response_text, 
            status="success",
            extracted_conv_id=conv_id,
            extracted_user=target_user
        )

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Generation Error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)