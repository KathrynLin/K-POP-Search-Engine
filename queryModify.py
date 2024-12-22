import spacy
import json

def load_kpop_terms(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        terms = json.load(file)
    return terms

def query_modification_with_kpop_terms(query, kpop_terms):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)

    modified_query = []

    for token in doc:
        # Replace K-pop terms with their definitions if they exist
        if token.text in kpop_terms:
            modified_query.append(kpop_terms[token.text])
        else:
            modified_query.append(token.text)

    return ' '.join(modified_query)

