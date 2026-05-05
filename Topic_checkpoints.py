import os
import json
import pandas as pd
from groq import Groq
# from dotenv import load_dotenv
import tqdm

# load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY")) 
MODEL = "llama-3.1-8b-instant"

def get_llm_response(prompt, system_prompt="You are a helpful assistant."):
    """Used for simple text-based monitoring tasks."""
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model=MODEL,
        temperature=0.1, 
    )
    return chat_completion.choices[0].message.content.strip()

def extract_topic_and_summary(segment_text):
    """Uses structured prompting and JSON mode to extract data deterministically."""
    system_prompt = """You are a precise data extraction API. Your ONLY objective is to analyze a conversation segment and output a strict JSON object.

RULES:
1. `topic_label`: Must be exactly 2 to 4 words. No prefixes like "Topic:".
2. `summary`: Must be exactly one concise, objective sentence. Do not use filler like "Here is a summary".

EXAMPLE OUTPUT:
{"topic_label": "Career and Travel", "summary": "User 1 shares their plans to move to Portland for a culinary career, and User 2 recommends visiting Powell's Books."}"""

    user_prompt = f"CONVERSATION SEGMENT:\n{segment_text}"

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=MODEL,
        temperature=0.0, 
        response_format={"type": "json_object"}
    )
    
    raw_response = chat_completion.choices[0].message.content.strip()
    
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"topic_label": "Unknown Topic", "summary": raw_response}

def process_conversation(conv_text, conv_id):
    messages = [msg.strip() for msg in conv_text.split('\n') if msg.strip()]
    
    checkpoints = []
    current_segment = []
    current_topic_label = "Initial Conversation"
    start_index = 1
    
    for i, msg in enumerate(messages):
        current_segment.append(msg)
        msg_index = i + 1
        
        if len(current_segment) > 1:
            monitor_prompt = f"""
            Current Topic Context: {current_topic_label}
            Last few messages: {current_segment[-3:-1]}
            New Message: {msg}
            
            Does the 'New Message' introduce a significantly new topic or subject compared to the current context?
            Respond with only one word: 'CHANGE' or 'CONTINUE'.
            """
            
            decision = get_llm_response(monitor_prompt, "You are a topic detection monitor.")
            
            if "CHANGE" in decision.upper():
                segment_text = " ".join(current_segment[:-1])
                extracted_data = extract_topic_and_summary(segment_text)

                checkpoints.append({
                    "topic_id": len(checkpoints) + 1,
                    "range": f"{start_index}-{msg_index-1}",
                    "topic_label": extracted_data.get("topic_label", "Unknown"),
                    "summary": extracted_data.get("summary", "No summary generated.")
                })
                
                # 3. Reset for the new topic
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

    return {
        "conversation_id": conv_id,
        "checkpoints": checkpoints
    }


def process_all_conversations(input_csv_path, output_json_path):
    print(f"Loading dataset from: {input_csv_path}")
    df = pd.read_csv(input_csv_path, header=None)
    
    all_results = []
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processing Conversations"):
        conv_text = row[0]
        conv_id = index + 1 
        
        try:
            result = process_conversation(conv_text, conv_id=conv_id)
            all_results.append(result)
            if conv_id % 100 == 0:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            print(f"\nError processing conversation {conv_id}: {e}")
            all_results.append({
                "conversation_id": conv_id,
                "error": str(e),
                "checkpoints": []
            })
            continue

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"\nProcessing complete! Successfully saved {len(all_results)} conversations to {output_json_path}")

if __name__ == "__main__":
    input_file = r'/kaggle/working/User-Profiling-Chatbot/Topic_checkpoints.py'
    output_file = r'/kaggle/working/User-Profiling-Chatbot/dataset/processed_checkpoints.json'
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    process_all_conversations(input_file, output_file)
    