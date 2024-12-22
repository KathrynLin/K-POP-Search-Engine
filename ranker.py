"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from indexing import InvertedIndex
from collections import Counter
import math
from math import log
from transformers import pipeline, AutoTokenizer
from GetEmotion import get_emotion_vector

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]=None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query
        query_tokens = self.tokenize(query) or []
        if self.stopwords:
            query_tokens = [token if token.lower() not in self.stopwords else '_' for token in query_tokens]

        #print(f"Query Tokens: {query_tokens}")
        query_word_counts = Counter(query_tokens)

        # 2. Fetch a list of possible documents from the index
        candidate_docs = set()
        for token in query_word_counts:
            postings = self.index.get_postings(token)
            for docid, term_frequency, *positions in postings:
                candidate_docs.add(docid)
                #print(docid, term_frequency, positions)
        if not candidate_docs:
            return []
        
        # 2. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        doc_scores = []
        for docid in candidate_docs:
            doc_word_counts = self.index.get_doc_word_counts(docid, query_word_counts)  
            score = self.scorer.score(docid, doc_word_counts, query_word_counts)
            doc_scores.append((docid, score))
            
        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores

class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


import random

class RandomScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        """
        Initializes the RandomScorer.

        Args:
            index: The inverted index (not used in this scorer).
            parameters: Additional parameters (not used in this scorer).
        """
        self.index = index

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Assigns a random score between 1 and 5 to each document.

        Args:
            docid: The ID of the document (not used in this scorer).
            doc_word_counts: A dictionary of document word counts (not used in this scorer).
            query_word_counts: A dictionary of query word counts (not used in this scorer).

        Returns:
            A random float score between 1 and 5.
        """
        return random.uniform(1, 5)

class PopularityScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        """
        Initializes the PopularityScorer.

        Args:
            index: The inverted index.
            parameters: Additional parameters (not used in this scorer).
        """
        self.index = index

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Assigns a score based on the document's popularity.

        Args:
            docid: The ID of the document.
            doc_word_counts: A dictionary of document word counts (not used in this scorer).
            query_word_counts: A dictionary of query word counts (not used in this scorer).

        Returns:
            A score representing the popularity of the document.
        """
        # Retrieve popularity from the document metadata
        popularity = self.index.get_doc_metadata(docid).get('popularity', 0)
        return popularity


# TODO Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        dot_product = 0.0
        # doc_word_counts = {term: count for term, count in doc_word_counts.items()}
        # query_word_counts = {term: count for term, count in query_word_counts.items()}
        
        # for term, query_count in query_word_counts.items():
        #     print("doc_word_counts", doc_word_counts)
        #     doc_count = doc_word_counts.get(term, 0)
        #     print(f"Scoring Doc {docid}: Term: {term}, Query Count: {query_count}, Doc Count: {doc_count}")

        #     dot_product += query_count * doc_count
        #     #print(f"Document Term: {term}, Document Count: {doc_count}, Dot Product: {dot_product}")
        
        for term, query_count in query_word_counts.items():
            doc_count = doc_word_counts.get(term, 0)
            #print(f"Scoring Doc {docid}: Term: {term}, Query Count: {query_count}, Doc Count: {doc_count}")
            dot_product += query_count * doc_count
        if dot_product == 0:
            return 0.0
        # 2. Return the score
        # doc_magnitude = math.sqrt(sum(count ** 2 for count in doc_word_counts.values()))
        # query_magnitude = math.sqrt(sum(count ** 2 for count in query_word_counts.values()))
        # if doc_magnitude == 0 or query_magnitude == 0:
        #     return 0.0
        return dot_product

# TODO Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.mu = parameters.get('mu', 2000)


    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        # Document length |d|
        doc_length = self.index.get_doc_metadata(docid)['length']
        # |collection|
        total_token_count = self.index.get_statistics()['total_token_count']

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score
        mu = self.mu

        score = 0.0
        

        for term, q_count in query_word_counts.items():
            tf = doc_word_counts.get(term, 0)
            cf = self.index.get_term_metadata(term).get('term_count', 0)
            
            if cf == 0:
                continue
            
            probability = cf / total_token_count

            second_part = log(1 + tf / (mu * probability))
            score += q_count * second_part
            #print(f"Term: {term}, Query Count: {q_count}, TF in Document: {tf}, CF in Collection: {cf}")
            #print(f"Smoothed Probability P({term}|doc): {second_part}")

        score += sum(query_word_counts.values()) * math.log(mu / (doc_length + mu))
        #print(query_word_counts.values(), mu, doc_length)

        # 4. Return the score
        #print(f"Final Score for Document {docid}: {score}")

        return score


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        stats = self.index.get_statistics()
        doc_length = self.index.get_doc_metadata(docid)['length']  

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query
        N = len(self.index.document_metadata)
        avgdl = self.index.get_statistics()['mean_document_length']
        
        # 3. For all query parts, compute the TF and IDF to get a score 
        score = 0.0   
        
        for term, qf in query_word_counts.items():
            # Get term frequency in the document and document frequency (df) in the corpus
            tf = doc_word_counts.get(term, 0)  # Term frequency in document
            term_stats = self.index.get_term_metadata(term)
            df = term_stats['doc_frequency']  # Document frequency

            # Skip the term if it doesn't appear in the document
            if tf == 0:
                continue

            # 3. Compute IDF using the provided formula
            idf = math.log((N - df + 0.5) / (df + 0.5))

            # 4. Compute the term frequency (TF) component
            tf_component = ((self.k1 + 1) * tf) / (self.k1 * (1 - self.b + self.b * (doc_length / avgdl)) + tf)

            # 5. Compute the query frequency (QF) component (if qf > 1)
            qf_component = ((self.k3 + 1) * qf) / (self.k3 + qf)

            # 6. Calculate the term score and add to the total score
            term_score = idf * tf_component * qf_component
            score += term_score
        # 4. Return score
        return score


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        
        doc_length = self.index.get_doc_metadata(docid)['length']  
        avgdl = self.index.get_statistics()['mean_document_length']  
        N = len(self.index.document_metadata)
        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        score = 0.0
        for term, qf in query_word_counts.items():
            # Get term frequency in the document and document frequency (df) in the corpus
            tf = doc_word_counts.get(term, 0)  # Term frequency in document
            term_stats = self.index.get_term_metadata(term)
            df = term_stats['doc_frequency']  # Document frequency

            # Skip term if it doesn't appear in the document
            if tf == 0:
                continue

            # 3. Compute the logarithmic term frequency component
            tf_log = 1 + math.log(1 + math.log(tf))

            # 4. Compute the length normalization component
            length_norm = 1 / (1 - self.b + self.b * (doc_length / avgdl))

            # 5. Compute the inverse document frequency (IDF) component
            idf = math.log((N + 1) / df)

            # 6. Calculate the term score for the query and accumulate it
            term_score = qf * tf_log * length_norm * idf
            score += term_score
        # 4. Return the score
        return score
class EmotionScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex) -> None:
        self.index = index
        self.classfier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
        self.tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        doc_emotion = self.index.get_doc_metadata(docid)['emotion_vector']
        query = ' '.join(query_word_counts.keys())
        query_emotion = get_emotion_vector(self.classfier, self.tokenizer, query)
        score = sum([doc_emotion[i] * query_emotion[i] for i in range(len(doc_emotion))])
        return score

# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        N = len(self.index.document_metadata)
        # 2. Compute additional terms to use in algorithm
    

        # 3. For all query parts, compute the TF and IDF to get a score
        score = 0.0
        for term, qf in query_word_counts.items():
            tf = doc_word_counts.get(term, 0)  
            term_stats = self.index.get_term_metadata(term)
            df = term_stats['doc_frequency']  

            if tf == 0 or df == 0:
                continue

            idf = math.log(N / df) + 1
            tf_weight = math.log(tf + 1)
            term_score = tf_weight * idf

            score += term_score

        # 4. Return the score
        return score


# TODO Implement your own ranker with proper heuristics
class LengthNormalizedScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.5}) -> None:
        """
        Initializes the scorer with a length normalization parameter `b`.
        `b` is used to control the impact of length normalization.
        """
        self.index = index
        self.b = parameters['b'] 
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # Total number of documents in the corpus
        N = len(self.index.document_metadata)
        
        # Average document length
        avgdl = sum([meta['length'] for meta in self.index.document_metadata.values()]) / N
        
        # Get term statistics from the index
        term_stats = {term: self.index.get_term_metadata(term) for term in query_word_counts.keys()}

        # Get document length
        doc_length = self.index.get_doc_metadata(docid)['length']
        
        # Compute TF-IDF part
        tf_idf_score = 0.0
        for term in set(query_word_counts.keys()).intersection(doc_word_counts.keys()):
            tf = doc_word_counts[term]
            df = term_stats[term]['doc_frequency']
            idf = math.log(N / df) + 1  # IDF component
            tf_weight = math.log(tf + 1)  # Logarithmic TF
            term_score = tf_weight * idf
            tf_idf_score += term_score

        # Compute Length Normalization Penalty
        length_penalty = 1 / (1 + self.b * ((doc_length / avgdl) - 1))

        # Final score is TF-IDF score adjusted by length penalty
        final_score = tf_idf_score * length_penalty

        return final_score
    
