'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
from document_preprocessor import Tokenizer
from collections import Counter, defaultdict
import os
import json
import gzip
from tqdm import tqdm
from GetEmotion import get_emotion_vector

class IndexType(Enum):
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    InvertedIndex = 'BasicInvertedIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter() # token count
        self.vocabulary = set()  # the vocabulary of the collection
        self.document_metadata = {} # metadata like length, number of unique tokens of the documents

        self.index = defaultdict(list)  # the index 

    
    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str], danceability: int, energy: int) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError

    def get_doc_word_counts(self, docid: int, query_word_counts: dict[str, int]) -> dict[str, int]:
        """
        Retrieves the word counts for a document, limiting to terms that appear in the query.
        """
        doc_word_counts = {}
        for term in query_word_counts:
            postings = self.get_postings(term)  # Retrieve the list of postings for the term
            for posting in postings:
                if isinstance(posting, tuple) and len(posting) >= 2:
                    doc, term_frequency = posting[0], posting[1]
                else:
                    doc, term_frequency = posting, 1  # If posting is an int, treat it as the frequency or doc_id.
                
                if doc == docid:
                    doc_word_counts[term] = term_frequency
        return doc_word_counts
    
class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.index = defaultdict(lambda: defaultdict(int))
        #self.total_token_with_filtered = 0 # number of tokens before filtered
        
    def add_doc(self, docid: int, tokens: list[str], danceability: int, energy: int) -> None:
        doc_length = len(tokens)
        filtered_tokens = [token for token in tokens if token is not None]
        term_count = Counter(filtered_tokens)
        unique_tokens = len(term_count)

        filtered_text =  ' '.join([token for token in tokens if token is not None])
        emotion_vector = get_emotion_vector(filtered_text)
        self.document_metadata[docid] = {
            'unique_tokens': unique_tokens,
            'length': doc_length,
            'emotion_vector': emotion_vector,
            'danceability': danceability,
            'energy': energy
        }

        for term, count in term_count.items():
            if term:
                self.index[term][docid] += count
                self.statistics['vocab'][term] += count 
                self.vocabulary.add(term)
        
    
    def remove_doc(self, docid: int) -> None:
        if docid in self.document_metadata:
            doc_metadata = self.document_metadata[docid]
            doc_length = doc_metadata['length']
            
            del self.document_metadata[docid]
            
            terms_to_delete = []
            
            for term in list(self.index.keys()):
                if docid in self.index[term]:
                    term_count = self.index[term][docid]
                    
                    self.statistics['vocab'][term] -= term_count
                    del self.index[term][docid]
                    
                    if not self.index[term]:
                        terms_to_delete.append(term)
            
            for term in terms_to_delete:
                del self.index[term]
                self.vocabulary.remove(term)
            
    
    def get_postings(self, term: str) -> list:
        if term in self.index:
            return list(self.index[term].items())  # returns (docid, term frequency)
        else:
            return []
    
    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        return self.document_metadata.get(doc_id, {})
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        return {
            'term_count': self.statistics['vocab'].get(term, 0),
            'doc_frequency': len(self.index.get(term, []))
        }
    
    def get_statistics(self) -> dict[str, int]:
        total_token_count = sum([doc_metadata['length'] for doc_metadata in self.document_metadata.values()])
        num_documents = len(self.document_metadata)
        mean_document_length = total_token_count / num_documents if num_documents > 0 else 0
        
        stored_total_token_count = self.statistics.get('stored_total_token_count', 0)
        #print(f"in the get_statistics function, the total token count is {total_token_count}, the doc_metadata is {self.document_metadata}")
        return {
            'unique_token_count': len(self.vocabulary),
            'total_token_count': total_token_count,
            'stored_total_token_count': stored_total_token_count,
            'number_of_documents': num_documents,
            'mean_document_length': mean_document_length
        }
    
    def save(self, index_directory_name: str) -> None:
        if not os.path.exists(index_directory_name):
            os.makedirs(index_directory_name)
        
        with open(f"{index_directory_name}/index.txt", "w") as f:
            for term, postings in self.index.items():
                postings_str = ",".join(f"{docid}:{count}" for docid, count in postings.items())
                f.write(f"{term} -> {postings_str}\n")
        
        with open(f"{index_directory_name}/vocabulary.txt", "w") as f:
            for word in self.vocabulary:
                f.write(f"{word}\n")
        
        with open(f"{index_directory_name}/document_metadata.txt", "w") as f:
            f.write(str(self.document_metadata))
        
        #self.statistics['total_token_with_filtered'] = self.total_token_with_filtered
        
        with open(f"{index_directory_name}/statistics.txt", "w") as f:
            f.write(str(self.statistics))
        
    def load(self, index_directory_name: str) -> None:
        with open(f"{index_directory_name}/index.txt", "r") as f:
            for line in f:
                term, postings_str = line.split(" -> ")
                postings = postings_str.split(",")
                self.index[term] = {int(docid): int(count) for docid, count in (posting.split(":") for posting in postings)}
        
        with open(f"{index_directory_name}/vocabulary.txt", "r") as f:
            self.vocabulary = set(f.read().splitlines())
        
        with open(f"{index_directory_name}/document_metadata.txt", "r") as f:
            self.document_metadata = eval(f.read())
        
        with open(f"{index_directory_name}/statistics.txt", "r") as f:
            self.statistics = eval(f.read())
        
        #self.total_token_with_filtered = self.statistics.get('total_token_with_filtered', 0)

class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        self.statistics['index_type'] = 'PositionalInvertedIndex'
        self.index = defaultdict(lambda: defaultdict(list))  
        
    def add_doc(self, docid: int, tokens: list[str], danceability: int, energy: int) -> None:
        #term_count = Counter(tokens)
        #self.total_token_with_filtered += sum(term_count.values())
        doc_length = len(tokens)

        term_positions = defaultdict(list)  # Store positions for each valid term
        for position, token in enumerate(tokens):
            if token is not None:
                term_positions[token].append(position)
        

        unique_tokens = len(term_positions)   
        filtered_text =  ' '.join([token for token in tokens if token is not None])
        emotion_vector = get_emotion_vector(filtered_text)
        
        self.document_metadata[docid] = {
            'unique_tokens': unique_tokens,
            'length': doc_length,
            'emotion_vector': emotion_vector,
            'danceability': danceability,
            'energy': energy
        }

        for term, positions in term_positions.items():
            self.index[term][docid].extend(positions)  
            self.statistics['vocab'][term] += len(positions)  
            self.vocabulary.add(term)
        
    def remove_doc(self, docid: int) -> None:
        if docid in self.document_metadata:
            del self.document_metadata[docid]

            terms_to_delete = []
        
        for term in list(self.index.keys()):
            if docid in self.index[term]:
                positions = self.index[term][docid]
                self.statistics['vocab'][term] -= len(positions)  
                del self.index[term][docid]
                
                if not self.index[term]:
                    terms_to_delete.append(term)
        
        for term in terms_to_delete:
            del self.index[term]
            self.vocabulary.remove(term)  
    
    def get_postings(self, term: str) -> list:
        if term in self.index:
            # Return the sorted list of postings (docid, term frequency, positions)
            return sorted(
                [(docid, len(positions), positions) for docid, positions in self.index[term].items()],
                key=lambda x: x[0]
            )  # Sort by docid
        else:
            return []
    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        return self.document_metadata.get(doc_id, {})
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        return {
            'term_count': self.statistics['vocab'].get(term, 0),
            'doc_frequency': len(self.index.get(term, []))
        }
    
    def get_statistics(self) -> dict[str, int]:
        total_token_count = sum([doc_metadata['length'] for doc_metadata in self.document_metadata.values()])
        num_documents = len(self.document_metadata)
        
        mean_document_length = total_token_count / num_documents if num_documents > 0 else 0
        
        stored_total_token_count = self.statistics.get('stored_total_token_count', 0)
        return {
            'unique_token_count': len(self.vocabulary),
            'total_token_count': total_token_count,
            'stored_total_token_count': stored_total_token_count,
            'number_of_documents': num_documents,
            'mean_document_length': mean_document_length
        }
    def save(self, index_directory_name: str) -> None:
    
        if not os.path.exists(index_directory_name):
            os.makedirs(index_directory_name)
        
        with open(f"{index_directory_name}/index.txt", "w") as f:
            for term, postings in self.index.items():
                postings_str = ",".join(f"{docid}:{','.join(map(str, positions))}" for docid, positions in postings.items())
                f.write(f"{term} -> {postings_str}\n")
        
        with open(f"{index_directory_name}/vocabulary.txt", "w") as f:
            for word in self.vocabulary:
                f.write(f"{word}\n")
        
        with open(f"{index_directory_name}/document_metadata.txt", "w") as f:
            f.write(str(self.document_metadata))
        
        with open(f"{index_directory_name}/statistics.txt", "w") as f:
            f.write(str(self.statistics))

    def load(self, index_directory_name: str) -> None:
        
        with open(f"{index_directory_name}/index.txt", "r") as f:
            for line in f:
                term, postings_str = line.split(" -> ")
                postings = postings_str.split(",")
                self.index[term] = {int(docid): list(map(int, positions.split(','))) for docid, positions in (posting.split(":") for posting in postings)}
        
        with open(f"{index_directory_name}/vocabulary.txt", "r") as f:
            self.vocabulary = set(f.read().splitlines())
        
        with open(f"{index_directory_name}/document_metadata.txt", "r") as f:
            self.document_metadata = eval(f.read())
        
        with open(f"{index_directory_name}/statistics.txt", "r") as f:
            self.statistics = eval(f.read())


class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                        document_preprocessor: Tokenizer, stopwords: set[str],
                        minimum_word_frequency: int, text_key = "text",
                        max_docs: int = -1, load_file_dir: str = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.

        Returns:
            An inverted index
        
        '''
        index = None
        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            raise ValueError(f"Invalid index type: {index_type}")
        
        
        def filters(tokens, global_term_count, minimum_word_frequency, stopwords):
            filtered_tokens = [term if (term not in stopwords and global_term_count[term] >= minimum_word_frequency) else None for term in tokens]
            return filtered_tokens
        
        def open_file(dataset_path):
            if dataset_path.endswith('.gz'):
                return gzip.open(dataset_path, 'rt', encoding='utf-8')  # Open as text using gzip
            else:
                return open(dataset_path, 'r', encoding='utf-8')  # Open as a regular text file

        global_term_count = Counter()
        doc_count = 0
        # index_ls = {}
        if load_file_dir: 
            index.load(load_file_dir)
        else:
            with open_file(dataset_path) as f:
                for line in tqdm(f, desc="Indexing"):
                    if max_docs > 0 and doc_count >= max_docs:
                        break 
                    doc = json.loads(line)
                    docid = int(doc['docid'])
                    text = doc.get(text_key, '')
                    danceability = doc.get('danceability', 0)
                    energy = doc.get('energy', 0)
                    if not isinstance(text, str):  # 确保 text 是字符串
                        text = ""
                    tokens = document_preprocessor.tokenize(text)
                    global_term_count.update(tokens)
                    doc_count += 1
                    # index_ls[doc["docid"]] = tokens
                
                # for docid, tokens in tqdm(index_ls.items(), desc="Processing documents"):
                #     filtered_tokens = filters(tokens, global_term_count, minimum_word_frequency, stopwords)
                #     index.add_doc(docid, filtered_tokens)   
                    
            doc_count = 0  # Reset document count for the second pass
            with open_file(dataset_path) as f:
                for line in tqdm(f, desc="Indexing"):
                    if max_docs > 0 and doc_count >= max_docs:
                        break
                    doc = json.loads(line)
                    docid = int(doc['docid'])
                    text = doc.get(text_key, '')
                    danceability = doc.get('danceability', 0)
                    energy = doc.get('energy', 0)
                    if not isinstance(text, str):  # 确保 text 是字符串
                        text = ""
                    tokens = document_preprocessor.tokenize(text)
                    
                    # Filter tokens and add to the index
                    filtered_tokens = filters(tokens, global_term_count, minimum_word_frequency, stopwords)
                    index.add_doc(docid, filtered_tokens, danceability, energy)

                    # Delete tokens and filtered tokens to free memory
                    del tokens, filtered_tokens
                    doc_count += 1
                        
        return index
            
            
                
