import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("Loading Gemma-4-12B...")

model_name = "google/gemma-4-12B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config={"load_in_4bit": True},
    trust_remote_code=True
)

print("✅ Loaded!\n")

def ask(question, emotion="neutral"):
    start_time = time.time()
    
    # Better prompt for Gemma
    prompt = f"""<start_of_turn>user
You are an emotionally aware NPC in a game.
User emotion: {emotion}
User said: "{question}"
<end_of_turn>
<start_of_turn>model
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up
    response = response.split("<start_of_turn>model")[-1].strip()
    
    latency = time.time() - start_time
    print(f"Q: {question} ({emotion})")
    print(f"A: {response}")
    print(f"Latency: {latency:.2f}s\n")

# Test
if __name__ == "__main__":
    tests = [
        ("I'm really frustrated with this quest, what should I do?", "frustration"),
        ("That enemy scared me! Help!", "fear"),
        ("This treasure makes me so happy!", "joy"),
        ("May I have some tea?", "polite")
    ]
    
    for q, e in tests:
        ask(q, e)