# Behavioral Profiler & Hybrid RAG Chatbot

## Project Overview
This project is an advanced, fully local Retrieval-Augmented Generation (RAG) system. Unlike standard chatbots that just search for text, this system deeply analyzes chat logs to extract psychological profiles (personas) and topic summaries. It then uses a hybrid approach to answer user questions by combining structured JSON data (who the person is) with unstructured vector data (what they talked about).

The entire pipeline is optimized to run locally and efficiently, using open-source models without relying on expensive, black-box APIs for data processing.

---

## How It Works: The Implementation Workflow

If you are reviewing this project, here is the plain-English breakdown of how the data flows from raw chat logs to an intelligent chatbot.

### 1. How I Detect Topic Changes (Semantic TextTiling)
Raw chat logs are messy and change topics randomly. Instead of asking a slow, expensive LLM "Did the topic change?" for every single line of text, I used a math-based approach:

* Converted every message into a mathematical vector (an "embedding") using a fast, lightweight model (`SentenceTransformer`). 
* Calculated the "Cosine Similarity" (the distance) between one message and the next. 
* If the similarity score drops below a certain threshold, it means the conversation took a sharp turn. The system marks this as a "Topic Boundary."
* Once a boundary is found, the system groups those messages and uses an open-source model (BART/Mistral) to generate a clean, third-person summary of that specific topic. 

*Why this matters:* It is extremely fast, highly accurate, and costs $0 in API fees to process massive datasets.

### 2. How the Persona is Built (Structured JSON Extraction)
To understand *who* the users are, the system needs to extract their habits, facts, traits, and communication styles. 

* Instead of feeding raw, noisy chat logs to an AI (which causes hallucinations), I feed the clean **Topic Summaries** we generated in Step 1 into an advanced LLM (like Mistral 7B).
* engineered a strict prompt that forces the AI to act as a behavioral analyst. It extracts traits and formats them directly into a rigid JSON structure.
* A major challenge in conversational data is mixing up speakers. The pipeline separates the data into `User 1` and `User 2` and permanently links them to their specific `Conversation ID` so facts never bleed into the wrong chat.

*Why this matters:* This creates a predictable, structured database of human behavior that is completely separate from the raw text.

### 3. How Retrieval & The Chatbot Works (Hybrid RAG)
When a user asks the chatbot a question (e.g., *"What are User 1's habits in conversation 10?"*), the system does not just blindly search a database. It uses a **Hybrid Context Injection** strategy:

* **Step A (Structured Data):** It instantly looks up the requested user in the Persona JSON file and grabs their exact habits and traits.
* **Step B (Unstructured Data):** It performs a semantic search on a local Vector Database (ChromaDB) to find the most relevant conversational summaries. I use **Metadata Filtering** to ensure it only searches within the requested Conversation ID, preventing data crossover.
* **Step C (Synthesis):** Both the JSON Profile and the Vector Summaries are fed into the final Chatbot (powered by LangChain and Groq). The Chatbot cross-references the hard facts from the JSON with the conversational context from the database to give a highly accurate, evidence-based answer.

*Why this matters:* Standard RAG often loses the "big picture" of who a person is. By injecting a dedicated JSON persona alongside the search results, the chatbot acts like a true behavioral analyst.

---

## Tech Stack
* **Topic Detection:** `SentenceTransformers` (all-MiniLM-L6-v2), Cosine Similarity math.
* **Summarization:** Hugging Face `pipeline`, `BART-large-samsum`.
* **Persona Extraction:** `Mistral-7B-Instruct-v0.3`, Strict JSON Prompt Engineering.
* **Vector Database:** `ChromaDB` (Local persistent storage).
* **Chatbot Framework:** `LangChain`, `Groq API` (Llama-3.1-70B for blazing-fast synthesis).
* **Environment:** Python, Pandas, PyTorch, optimized for GPU processing.

---
