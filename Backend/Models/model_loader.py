from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"Singleton class to load GPT2 model"
class GPT2_Model:
    _instance = None
    
    @staticmethod
    def get_instance():
        print(torch.cuda.is_available())  # Should return True if CUDA is enabled
        print(torch.version.cuda)     
        
        if GPT2_Model._instance is None:
            GPT2_Model()
        return GPT2_Model._instance
    
    def __init__(self):
        if GPT2_Model._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            print("Loading GPT2 model...")
           
            model_name = "AventIQ-AI/gpt2-news-article-generation"
        
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.chat_history = {} # Store chat history
            GPT2_Model._instance = self