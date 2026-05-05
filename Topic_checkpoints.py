import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash" 

print(f"Loading {MODEL_ID} into GPU VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
)

def generate_hf_response(messages, max_new_tokens=150, temperature=0.0):
    """Handles the tokenization, generation, and decoding for the local model."""
    # Apply the specific chat template (ChatML, etc.) expected by the model
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False, # Deterministic greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )

    # Slice the output to return only the newly generated tokens (ignore the prompt)
    generated_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def clean_json_output(raw_text):
    """Regex helper to extract JSON from markdown or chatty output."""
    try:
        # Search for anything looking like a JSON object {...}
        match = re.search(r'\{[\s\S]*\}', raw_text)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return {"topic_label": "Unknown", "summary": raw_text}
    except json.JSONDecodeError:
        return {"topic_label": "Error", "summary": "Failed to parse JSON."}

def extract_topic_and_summary_hf(segment_text):
    """Few-shot engineered prompt tailored for local instruction models."""
    system_prompt = """You are a precise data extraction algorithm. Your ONLY task is to output a single, valid JSON object. Do not include markdown code blocks. Do not add conversational text.

{
  "topic_label": "2 to 4 words describing the topic",
  "summary": "One objective sentence summarizing the dialogue"
}"""

    user_prompt = f"CONVERSATION SEGMENT:\n{segment_text}\n\nOUTPUT STRICT JSON:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    raw_response = generate_hf_response(messages, max_new_tokens=150, temperature=0.0)
    return clean_json_output(raw_response)

def process_conversation_hf(conv_text, conv_id):
    messages = [msg.strip() for msg in conv_text.split('\n') if msg.strip()]
    
    checkpoints = []
    current_segment = []
    current_topic_label = "Initial Conversation"
    start_index = 1
    
    for i, msg in enumerate(messages):
        current_segment.append(msg)
        msg_index = i + 1
        
        if len(current_segment) > 1:
            monitor_sys = "You are a logical topic-tracking monitor. Reply with exactly one word: 'CHANGE' or 'CONTINUE'."
            monitor_user = f"Current Topic: {current_topic_label}\nPrevious messages: {current_segment[-3:-1]}\nNew Message: {msg}\n\nDoes the New Message start a significantly new topic? Answer CHANGE or CONTINUE."
            
            hf_messages = [
                {"role": "system", "content": monitor_sys},
                {"role": "user", "content": monitor_user}
            ]
            
            decision = generate_hf_response(hf_messages, max_new_tokens=10, temperature=0.0)
            
            if "CHANGE" in decision.upper():
                segment_text = " ".join(current_segment[:-1])
                extracted_data = extract_topic_and_summary_hf(segment_text)
                
                checkpoints.append({
                    "topic_id": len(checkpoints) + 1,
                    "range": f"{start_index}-{msg_index-1}",
                    "topic_label": extracted_data.get("topic_label", "Unknown"),
                    "summary": extracted_data.get("summary", "No summary available.")
                })
                
                start_index = msg_index
                current_segment = [msg] 
                current_topic_label = extracted_data.get("topic_label", "Unknown")

    if current_segment:
        segment_text = " ".join(current_segment)
        extracted_data = extract_topic_and_summary_hf(segment_text)
        checkpoints.append({
            "topic_id": len(checkpoints) + 1,
            "range": f"{start_index}-{len(messages)}",
            "topic_label": extracted_data.get("topic_label", "Unknown"),
            "summary": extracted_data.get("summary", "No summary available.")
        })

    return {"conversation_id": conv_id, "checkpoints": checkpoints}


if __name__ == "__main__":
    input_file = '/kaggle/working/dataset/your-dataset-name/conversations.csv'
    output_file = '/kaggle/working/dataset/hf_topic_checkpoints.json'
    
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file, header=None)
    df_test = df.head(10)
    all_results = []
    for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="Processing on Kaggle GPU"):
        conv_text = row[0]
        conv_id = index + 1
        
        result = process_conversation_hf(conv_text, conv_id)
        all_results.append(result)

        if len(all_results) % 100 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print("Batch processing complete!")