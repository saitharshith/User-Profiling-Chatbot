import os
import json
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

print("Loading Model in 4-bit Quantization onto Kaggle GPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto", # Automatically distributes across Kaggle's T4 GPUs
    load_in_4bit=True, # CRITICAL: Prevents Out of Memory (OOM) errors on Kaggle
    torch_dtype=torch.float16,
    token=HF_TOKEN
)

# Initialize the text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id
)

# --- LLM INFERENCE FUNCTIONS ---

def get_local_llm_response(prompt, system_prompt="You are a helpful assistant.", max_new_tokens=10):
    """Uses the local Llama 3 pipeline with strict chat formatting."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    # Format the prompt using Llama 3's specific control tokens
    prompt_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    outputs = pipe(
        prompt_str,
        max_new_tokens=max_new_tokens,
        temperature=0.1, # Keep it deterministic
        do_sample=False,
        return_full_text=False # Only return the generated answer, not the prompt
    )
    
    return outputs[0]["generated_text"].strip()

def extract_topic_and_summary(segment_text):
    """Extracts the JSON data locally. Includes regex fallback for safety."""
    system_prompt = """You are a precise data extraction API. Your ONLY objective is to analyze a conversation segment and output a strict JSON object.
RULES:
1. `topic_label`: Must be exactly 2 to 4 words. No prefixes.
2. `summary`: Must be exactly one concise, objective sentence.
EXAMPLE OUTPUT:
{"topic_label": "Career and Travel", "summary": "User 1 shares their plans to move to Portland."}"""

    user_prompt = f"CONVERSATION SEGMENT:\n{segment_text}"
    
    raw_response = get_local_llm_response(
        prompt=user_prompt, 
        system_prompt=system_prompt, 
        max_new_tokens=150 
    )
    
    # Defensive parsing: Local LLMs might occasionally wrap JSON in markdown (```json ... ```)
    try:
        # First attempt: parse directly
        return json.loads(raw_response)
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
        except Exception:
            return {"topic_label": "Unknown", "summary": raw_response.replace('\n', ' ')}



def process_conversation_locally(conv_text, conv_id):
    messages = [msg.strip() for msg in conv_text.split('\n') if msg.strip()]
    
    checkpoints = []
    current_segment = []
    current_topic_label = "Initial Conversation"
    start_index = 1
    
    for i, msg in enumerate(messages):
        current_segment.append(msg)
        msg_index = i + 1
        
        if len(current_segment) > 1:
            monitor_prompt = f"Current Context: {current_topic_label}\nLast messages: {current_segment[-3:-1]}\nNew Message: {msg}\nDoes the 'New Message' introduce a new topic? Respond with one word: 'CHANGE' or 'CONTINUE'."
            decision = get_local_llm_response(
                prompt=monitor_prompt, 
                system_prompt="You are a strict topic detection monitor. Output exactly one word.",
                max_new_tokens=5 
            )
            
            if "CHANGE" in decision.upper():
                segment_text = " ".join(current_segment[:-1])
                extracted_data = extract_topic_and_summary(segment_text)
                
                checkpoints.append({
                    "topic_id": len(checkpoints) + 1,
                    "range": f"{start_index}-{msg_index-1}",
                    "topic_label": extracted_data.get("topic_label", "Unknown"),
                    "summary": extracted_data.get("summary", "No summary generated.")
                })
                
                start_index = msg_index
                current_segment = [msg] 
                current_topic_label = extracted_data.get("topic_label", "Unknown")

    if current_segment:
        segment_text = " ".join(current_segment)
        extracted_data = extract_topic_and_summary(segment_text)
        checkpoints.append({
            "topic_id": len(checkpoints) + 1,
            "range": f"{start_index}-{len(messages)}",
            "topic_label": extracted_data.get("topic_label", "Unknown"),
            "summary": extracted_data.get("summary", "No summary generated.")
        })

    return {"conversation_id": conv_id, "checkpoints": checkpoints}


if __name__ == "__main__":
    input_file = '/kaggle/working/dataset/conversations.csv'
    output_file = '/kaggle/working/dataset/processed_checkpoints.json'
    
    print(f"Loading dataset...")
    try:
        df = pd.read_csv(input_file, header=None)
    except FileNotFoundError:
        print("Please update the input_file path to match your Kaggle dataset directory.")
        
    all_results = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing on GPU"):
        conv_text = row[0]
        conv_id = index + 1
        
        try:
            result = process_conversation_locally(conv_text, conv_id=conv_id)
            all_results.append(result)
            
            if len(all_results) % 100 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            print(f"\nError on conv {conv_id}: {e}")
            all_results.append({"conversation_id": conv_id, "error": str(e), "checkpoints": []})

    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"Batch processing complete! Download the JSON from the Kaggle 'Output' directory.")