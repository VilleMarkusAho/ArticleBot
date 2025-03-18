import torch
import html
from Models.model_loader import GPT2_Model

def generate_text(prompt):
    instance = GPT2_Model.get_instance()
    
    # Tokenize input
    inputs = instance.tokenizer(prompt, return_tensors="pt").to(instance.device)
    
    with torch.no_grad():
        output_tokens = instance.model.generate(
            **inputs,
            max_length=200,  # Allow longer responses
            num_beams=5,  # Balances quality & diversity
            repetition_penalty=2.0,  # Reduce repeating patterns
            temperature=0.2,  # More deterministic response
            top_k=100,  # Allows more diverse words
            top_p=0.9,  # Keeps probability confidence
            do_sample=True,  # Sampling for variety
            no_repeat_ngram_size=3,  # Prevents excessive repetition
            num_return_sequences=1,  # Returns one best sequence
            early_stopping=True,  # Stops when response is complete
            length_penalty=1.2,  # Balances response length
            pad_token_id=instance.tokenizer.eos_token_id,  # Prevents truncation
            eos_token_id=instance.tokenizer.eos_token_id,  # Ensures completion
            return_dict_in_generate=True,  # Structured output
            output_scores=True  # Debugging purposes
    )
        
    # Decode output
    generated_response = instance.tokenizer.decode(output_tokens.sequences[0], skip_special_tokens=True)
    cleaned_response = html.unescape(generated_response).replace("#39;", "'").replace("quot;", '"')
    
    return cleaned_response