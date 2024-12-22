import json
import csv

def calculate_similarity(doc1, doc2):
    danceability_diff = doc1['danceability'] - doc2['danceability']
    energy_diff = doc1['energy'] - doc2['energy']
    return (danceability_diff**2 + energy_diff**2)**0.5

def calculate_similarities(data, list1, list2):
    doc_map = {doc['docid']: doc for doc in data}
    result = {}
    
    for docid1,_ in list1:
        if docid1 not in doc_map:
            continue
        
        doc1 = doc_map[docid1]
        similarities = []
        
        for docid2 in list2:
            if docid2 not in doc_map:
                continue
            
            doc2 = doc_map[docid2]
            similarity = calculate_similarity(doc1, doc2)
            similarities.append(similarity)
    
        result[docid1] = sum(similarities) / len(similarities) if similarities else None
    
    return result

def rerank_docids(similarity_scores):
    return sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

def read_list_from_csv(file_path):
    list2 = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            list2.extend(int(docid) for docid in row if docid.strip().isdigit())
    return list2

def get_reranked_track_list(jsonl_path, list1, csv_path):
    """
    Read data from JSONL and rerank tracks in list1 based on similarities to list2.
    """
    # Load data from JSONL
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    
    list2 = read_list_from_csv(csv_path)
    
    # Process and rerank each subset of list1
    final_results = []
    for start, end in [(0, 10), (10, 20), (20, 30)]:
        sublist = list1[start:end]
        similarity_scores = calculate_similarities(data, sublist, list2)
        reranked_docids = rerank_docids(similarity_scores)
        
        # Collect results
        doc_map = {doc['docid']: doc for doc in data}
        for docid, score in reranked_docids:
            if docid in doc_map:
                final_results.append((docid, score))
    
    return final_results
