import csv
import math
from tqdm import tqdm

"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""

def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    relevant_retrieved = 0
    precision_at_k = 0.0
    total_relevant = sum(search_result_relevances)
    if total_relevant == 0:
        return 0.0
    for idx, relevance in enumerate(search_result_relevances[:cut_off]):
        if relevance > 0:  
            relevant_retrieved += 1
            
            precision_at_k += relevant_retrieved / (idx + 1) 

    return precision_at_k / total_relevant if total_relevant > 0 else 0.0




def ndcg_score(search_result_relevances: list[float], ideal_relevance_score_ordering: list[float], cut_off: int = 10) -> float:
    """
    Calculate NDCG score given a list of relevance scores and the ideal ordering.
    Args:
        search_result_relevances: Relevance scores of retrieved documents.
        ideal_relevance_score_ordering: The ideal ordering of relevance scores (sorted in descending order).
        cut_off: Number of top results to consider in NDCG calculation (default 10).
    Returns:
        NDCG score (float).
    """
    def dcg(relevances):
        dcg_value = 0.0
        for idx, rel in enumerate(relevances[:cut_off]):
            rank = idx + 1  # Rank positions start from 1
            if rank == 1:
                dcg_value += rel
            else:
                dcg_value += rel / math.log2(rank)
        return dcg_value
    
    dcg_score = dcg(search_result_relevances)
    idcg_score = dcg(ideal_relevance_score_ordering)
    
    if idcg_score == 0.0:
        return 0.0
    else:
        ndcg = dcg_score / idcg_score
    return ndcg


def run_relevance_tests(relevance_data_filename: str, ranker, ranker_args={}) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """

    query_relevance = {}

    with open(relevance_data_filename, 'r', encoding='ISO-8859-1') as file:
        reader = csv.DictReader(file)
        for row in reader:
            query = row['query']  
            doc_id = int(row['docid'])  
            rel = int(row['rel'])
            rel = math.ceil(rel)
            if query not in query_relevance:
                query_relevance[query] = {}
            query_relevance[query][doc_id] = rel

    map_scores = []
    ndcg_scores = []
    
    for query, doc_relevances in tqdm(query_relevance.items(), desc="Evaluating"):
        retrieved_docs = ranker.query(query, **ranker_args)  
        retrieved_docids = [docid for docid, score in retrieved_docs]

        relevance_scores = [doc_relevances.get(docid, 0) for docid in retrieved_docids]
        ideal_relevance_ordering = sorted(doc_relevances.values(), reverse=True)
        
        search_result_relevances = [1 if rel > 3 else 0 for rel in relevance_scores]

        map_score_value = map_score(search_result_relevances)
        ndcg_score_value = ndcg_score(search_result_relevances, ideal_relevance_ordering)
        
        map_scores.append(map_score_value)
        ndcg_scores.append(ndcg_score_value)
    
    # Compute the average MAP and NDCG across all queries and return the scores
    return {
        'map': sum(map_scores) / len(map_scores) if map_scores else 0.0, 
        'ndcg': sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0, 
        'map_list': map_scores, 
        'ndcg_list': ndcg_scores
    }

if __name__ == '__main__':
    pass
