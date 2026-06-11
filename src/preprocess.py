import re
import string

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    #Convert to lowercase
    text = text.lower()
    
    #Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    #Remove URLs/Websites
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    #Remove Email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    #Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    #Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
