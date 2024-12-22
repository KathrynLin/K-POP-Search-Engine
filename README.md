# K-pop Music Search Engine - Interactive Usage Guide


## **Prerequisites**

### **Required Libraries**

Before running the pipeline, ensure you have the following Python libraries installed. You can install most of these using `pip`.


---
## **Data Preprocessors**

Change the API keys in `config.json` to your own keys.

Run `Data_Preprocessor/web_scraper.py` to get data from the Spotify API and scrape the lyrics from Genius. 

Run `Data_Preprocessor/translate_lyrics.py` to translate the lyrics to English.


## **Pipeline Components**

The pipeline consists of the following steps:

### 1. **Index Creation**
Indexes are created for efficient document retrieval. Run:
```python
from indexing import Indexer, IndexType
from document_preprocessor import RegexTokenizer

stopwords = set()
with open('tests/stopwords.txt', 'r', encoding='utf-8') as file:
    for stopword in file:
        stopwords.add(stopword)

preprocessor = RegexTokenizer('\w+')

main_index = Indexer.create_index(
    IndexType.BasicInvertedIndex, 'data/kpop_tracks.jsonl.gz', preprocessor
)
main_index.save('data/main_index')

title_index = Indexer.create_index(
    IndexType.BasicInvertedIndex, 'data/kpop_tracks.jsonl.gz', preprocessor
)
title_index.save('data/title_index')
```

### 2. **Ranker Setup**
Set up the baseline models and advanced rankers:
- **BM25** for baseline ranking:
  ```python
  from ranker import BM25
  bm25 = BM25(main_index)
  ```
- **Learning-to-Rank (L2R)** with or without user history:
  ```python
  from l2r_baseline import L2RFeatureExtractor, L2RRanker

  l2r_feature_extractor = L2RFeatureExtractor(
      document_index=main_index,
      title_index=title_index,
      stopwords=stopwords
  )
  
  l2r_ranker = L2RRanker(
      document_index=main_index,
      title_index=title_index,
      stopwords=stopwords,
      scorer=BM25(main_index),
      feature_extractor=l2r_feature_extractor
  )
  l2r_ranker.train('data/train.csv')
  ```

### 3. **Run Queries**
Execute queries interactively:
```python
initial_ranking = l2r_ranker.query("Songs about love")
```

### 4. **Apply Filters**
Narrow down results using filters:
```python
from filter import filter_tracks

filtered_tracks = filter_tracks(
    jsonl_path='data/data.jsonl',
    artist_name='Stray Kids',
    danceability_range=(0.5, 0.9),
    energy_range=(0.7, 1.0),
    release_year_range=(2019, 2023),
    docid_list=[item[0] for item in initial_ranking]
)

for track in filtered_tracks:
    print(track)e
```

---

## **Evaluating the Engine**

Run the evaluation script to compare models:
```python
from relevance import run_relevance_tests

bm25_results = run_relevance_tests('data/test.csv', bm25)
l2r_results = run_relevance_tests('data/test.csv', l2r_ranker)

print("BM25 - MAP:", bm25_results['map'])
print("L2R - MAP:", l2r_results['map'])
```


---

## **Interactive Testing**
To test specific queries:
1. Replace `"Stray Kids energetic songs"` in the query function with your desired search prompt.
2. Modify filter parameters (artist name, danceability, etc.) to refine results. Note that danceability and energy are on a scale from 0 to 1.

