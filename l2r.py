import lightgbm as lgb
from document_preprocessor import Tokenizer
from indexing import InvertedIndex, BasicInvertedIndex
from ranker import *
import json
import csv
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from rerank import *
from queryModify import *

class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], scorer: RelevanceScorer,
                 feature_extractor: 'L2RFeatureExtractor', datapath: str, historypath: str) -> None:
        """
        Initializes a L2RRanker model.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The relevance scorer
            feature_extractor: The L2RFeatureExtractor object
        """
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.scorer = scorer
        self.feature_extractor = feature_extractor
        self.model = LambdaMART()
        self.datapath = datapath
        self.historypath = historypath

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of 
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            tuple: A tuple containing the training data in the form of three lists: x, y, and qgroups
                X (list): A list of feature vectors for each query-document pair
                y (list): A list of relevance scores for each query-document pair
                qgroups (list): A list of the number of documents retrieved for each query
        """

        X = []
        y = []
        qgroups = []

        for query, doc_rel_list in tqdm(query_to_document_relevance_scores.items(), desc="Processing Queries"):
            query_tokens = self.document_preprocessor.tokenize(query)
            if self.stopwords:
                query_tokens = [token for token in query_tokens if token not in self.stopwords]
            
            doc_word_counts = self.accumulate_doc_term_counts(self.document_index, query_tokens)
            title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_tokens)

            # Track the number of documents for this query
            qgroups.append(len(doc_rel_list))

            for docid, relevance in doc_rel_list:
                # Generate features for the document-query pair
                features = self.feature_extractor.generate_features(docid, doc_word_counts.get(docid, {}), 
                                                                    title_word_counts.get(docid, {}), query_tokens)
                X.append(features)
                y.append(relevance)
        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        doc_term_counts = defaultdict(dict)

        for query_token in query_parts:
            postings = index.get_postings(query_token)
            if postings:
                for docid, freq in postings:
                    doc_term_counts[docid][query_token] = freq

        return doc_term_counts

    def train(self, training_data_filename: str, dev_data_filename: str = None) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        query_to_document_relevance_scores = {}

        with open(training_data_filename, 'r', encoding='ISO-8859-1') as csvfile:
            reader = csv.DictReader(csvfile)  # Automatically reads headers from the file
            for row in reader:
                if 'query' in row and 'docid' in row and 'rel' in row:
                    query = row["query"]
                    docid = int(row["docid"])  # Convert docid to int
                    rel = float(row["rel"])    # Convert relevance to float
                    rel = math.ceil(rel)       # Round up the relevance score if it's not an integer
                    
                if query not in query_to_document_relevance_scores:
                    query_to_document_relevance_scores[query] = []
                query_to_document_relevance_scores[query].append((docid, rel))

        X_train, y_train, qgroups_train = self.prepare_training_data(query_to_document_relevance_scores)

        # If dev dataset is provided, use it for validation
        if dev_data_filename:
            query_to_document_relevance_scores_dev = {}
            with open(dev_data_filename, 'r', encoding='ISO-8859-1') as csvfile:
                reader = csv.DictReader(csvfile)  # Automatically reads headers from the file
                for row in reader:
                    if 'query' in row and 'docid' in row and 'rel' in row:
                        query = row["query"]
                        docid = int(row["docid"])  # Convert docid to int
                        rel = float(row["rel"])    # Convert relevance to float
                        rel = math.ceil(rel)       # Round up the relevance score if it's not an integer

                    if query not in query_to_document_relevance_scores_dev:
                        query_to_document_relevance_scores_dev[query] = []
                    query_to_document_relevance_scores_dev[query].append((docid, rel))

            X_dev, y_dev, qgroups_dev = self.prepare_training_data(query_to_document_relevance_scores_dev)
        else:
            X_dev, y_dev, qgroups_dev = None, None, None

        # Define parameters for the model
        evals_result = {}  # This will store the results of the evaluation metrics
        if X_dev is not None and y_dev is not None and qgroups_dev is not None:
            # Train with validation data and track progress
            self.model.fit(
                X_train, y_train, qgroups_train,
                eval_set=[(X_dev, y_dev)],  # Validation set
                eval_names=["dev"],         # Name of the evaluation set
                eval_group=[qgroups_dev],   # Query groups for validation data
                early_stopping_rounds=10,   # Early stopping criteria
                evals_result=evals_result   # Store evaluation metrics in this dictionary
            )
        else:
            # Train without validation data
            self.model.fit(
                X_train, y_train, qgroups_train
            )
    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.predict(X)

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        kpop_terms_file = "data/new_kpop_terms.json"  
        kpop_terms = load_kpop_terms(kpop_terms_file)
        modified_query = query_modification_with_kpop_terms(query, kpop_terms)

        query_tokens = self.document_preprocessor.tokenize(modified_query)
        if self.stopwords:
            query_tokens = [token for token in query_tokens if token not in self.stopwords]
        query_word_counts = dict(Counter(query_tokens))
          
        doc_word_counts = self.accumulate_doc_term_counts(self.document_index, query_tokens)
        title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_tokens)
        
        initial_scores = {}
        for docid in doc_word_counts.keys():
            # Use the RelevanceScorer to get initial scores (e.g., BM25)
            score = self.scorer.score(docid, doc_word_counts[docid], query_word_counts)
            initial_scores[docid] = score
        sorted_initial_scores = sorted(initial_scores.items(), key=lambda item: item[1], reverse=True)
        top_100_docs = sorted_initial_scores[:100]  
              
        X = []
        doc_ids = []
        for docid, _ in top_100_docs:
            # Get word counts for the title and body of the document
            doc_body_word_counts = doc_word_counts.get(docid, {})
            doc_title_word_counts = title_word_counts.get(docid, {})

            features = self.feature_extractor.generate_features(docid, doc_body_word_counts, doc_title_word_counts, query_tokens)
            X.append(features)
            doc_ids.append(docid)
        
        if len(X) > 0:
            scores = self.model.predict(X)
        else:
            scores = []
        ranked_docs = [(doc_ids[i], scores[i]) for i in range(len(doc_ids))]
        ranked_docs = sorted(ranked_docs, key=lambda item: item[1], reverse=True)
    
        remaining_docs = sorted_initial_scores[100:]
        for docid, score in remaining_docs:
            ranked_docs.append((docid, score))
        
        ranked_docs = get_reranked_track_list(self.datapath, ranked_docs, self.historypath)
        return ranked_docs        

class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str]) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
        """
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.bm25 = BM25(document_index)
        self.pivoted_normalization = PivotedNormalization(document_index)
        self.emotion_socrer = EmotionScorer(document_index)

    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        if docid not in self.document_index.document_metadata:
            return 0 
        return self.document_index.document_metadata[docid].get('length', '')

    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        if docid not in self.document_index.document_metadata:
            return 0 
        return self.title_index.document_metadata[docid].get('length', '')


    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        query_word_counts = Counter(query_parts)

        if not word_counts:
            return 0.0
           
        tf_score = 0.0
        
        for term, qf in query_word_counts.items():
            if index.statistics['index_type'] == 'BasicInvertedIndex':
                tf = index.index.get(term, {}).get(docid, 0)
            else:
                tf = len(index.index.get(term, {}).get(docid, []))
                
            if tf > 0:
                tf_score += math.log(tf + 1)  
        return tf_score
    
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        query_word_counts = Counter(query_parts)

        N = len(index.document_metadata)
        score = 0.0
        
        for term, qf in query_word_counts.items():

            if index.statistics['index_type'] == 'BasicInvertedIndex':
                tf = index.index.get(term, {}).get(docid, 0)
            else:
                tf = len(index.index.get(term, {}).get(docid, []))
            term_stats = index.get_term_metadata(term)
            df = term_stats['doc_frequency']  

            if tf == 0 or df == 0:
                continue

            idf = math.log(N / df) + 1
            tf_weight = math.log(tf + 1)
            term_score = tf_weight * idf
            score += term_score
            
        return score
    
    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        query_word_counts = Counter(query_parts)
        return self.bm25.score(docid, doc_word_counts, query_word_counts)

    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        query_word_counts = Counter(query_parts)
        return self.pivoted_normalization.score(docid, doc_word_counts, query_word_counts)

    def get_title_keyword_density(self, title_index: InvertedIndex, docid: int, title_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the Title Keyword Density, which is the proportion of query terms 
        that appear in the document's title relative to the total number of words in the title.

        Args:
            title_index: An inverted index for the title to use for calculating the statistics
            docid: The id of the document
            title_word_counts: The words in the title mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The Title Keyword Density score, a value between 0 and 1.
        """
        # If the title is empty, return 0 as the density score
        if not title_word_counts or sum(title_word_counts.values()) == 0:
            return 0.0
        
        # Total number of words in the title
        total_title_words = sum(title_word_counts.values())

        # Count how many query terms appear in the title
        matching_terms = 0
        for term in query_parts:
            if term in title_word_counts:
                matching_terms += title_word_counts[term]

        # Calculate the density as the proportion of matching query terms in the title
        return matching_terms / total_title_words if total_title_words > 0 else 0.0

    def get_emotion_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        query_word_counts = Counter(query_parts)
        return self.emotion_socrer.score(docid, doc_word_counts, query_word_counts)

        
    
    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str]) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """

        feature_vector = []

        title_length = self.get_title_length(docid)
        feature_vector.append(title_length)
        
        query_length = len(query_parts)
        feature_vector.append(query_length)
        
        tf_doc = self.get_tf(self.document_index, docid, doc_word_counts, query_parts)
        feature_vector.append(tf_doc)
        
        tf_idf_doc = self.get_tf_idf(self.document_index, docid, doc_word_counts, query_parts)
        feature_vector.append(tf_idf_doc)
        
        tf_title = self.get_tf(self.title_index, docid, title_word_counts, query_parts)
        feature_vector.append(tf_title)
        
        tf_idf_title = self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts)
        feature_vector.append(tf_idf_title)
        
        bm25_score = self.get_BM25_score(docid, doc_word_counts, query_parts)
        feature_vector.append(bm25_score)
        
        pivoted_normalization_score = self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts)
        feature_vector.append(pivoted_normalization_score)
        
        title_keyword_density = self.get_title_keyword_density(self.title_index, docid, title_word_counts, query_parts)
        feature_vector.append(title_keyword_density)

        doc_emotion_score = self.get_emotion_score(docid, doc_word_counts, query_parts)
        feature_vector.append(doc_emotion_score)
        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 10,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.04,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": 1,
        }

        if params:
            default_params.update(params)

        self.model = lgb.LGBMRanker(**default_params)

    def fit(self,  X_train, y_train, qgroups_train, X_dev=None, y_dev=None, qgroups_dev=None):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """

        if X_dev is not None and y_dev is not None and qgroups_dev is not None:
            # Train with validation data
            self.model.fit(
                X_train, y_train,
                group=qgroups_train,  # Evaluation set
                eval_names=["dev"],
                eval_group=[qgroups_dev],   # Query group sizes for validation data
                early_stopping_rounds=10
            )
        else:
            # Train without validation data
            self.model.fit(
                X_train, y_train,
                group=qgroups_train        
            )

        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        return self.model.predict(featurized_docs)
