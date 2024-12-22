import re
from transformers import pipeline, AutoTokenizer



def clean_text(text):
    """
    Cleans the input text by:
    - Removing newlines and unnecessary returns.
    - Removing punctuations.
    - Removing content inside parentheses.
    """
    if not isinstance(text, str):
        return "" 
    text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses and their content
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations except spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and newlines
    words = text.split()
    if len(words) > 200:
        text = ' '.join(words[:200])
    return text

def get_emotion_vector(classifier, tokenizer, text):
    """
    Computes emotion scores for the given text. Truncates text to the model's max token limit.
    """
    # Truncate text to the model's maximum token limit



    max_length = tokenizer.model_max_length
    truncated_text = tokenizer.decode(
        tokenizer(text, truncation=True, max_length=max_length)["input_ids"], skip_special_tokens=True
    )
    
    try:
        # Perform emotion classification
        model_outputs = classifier([truncated_text])[0]
        emotion_scores = {item['label']: item['score'] for item in model_outputs}
        
        # List of all possible emotions in a fixed order
        emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
            'gratitude', 'grief', 'joy', 'love', 'neutral', 
            'nervousness', 'optimism', 'pride', 'realization', 'relief', 
            'remorse', 'sadness', 'surprise'
        ]
        
        # Create a vector of scores in the fixed order
        emotion_vector = [emotion_scores.get(emotion, 0.0) for emotion in emotions]
        return emotion_vector
    except Exception as e:
        print(f"Error during classification: {e}")
        return [0.0] * 28