import requests
import json

ACCESS_TOKEN = 'DZnqcYCoW3lR8e-8B0GobBzwarl8Kv0EAuQ1SXJICit8TWsukexaEIEX9o3a5xbL'

def get_lyrics(artist, song):
    params = {"q": song}
    url = f"https://api.genius.com/search?"
    response = requests.get(url, params=params, headers={'Authorization': 'Bearer ' + ACCESS_TOKEN, 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'})
    if response.status_code == 200:
        return response.json()['response']['hits']
    else:
        return response.text
    
def get_lyrics_url(data, song, artist):
    original_url, translation_url = None, None
    for hit in data:
        if song.lower() in hit['result']['full_title'].lower() and artist.lower() in hit['result']['artist_names'].lower() and not original_url:
            original_url = hit['result']['url']
        if song.lower() in hit['result']['full_title'].lower() and "English Translation" in hit['result']['full_title']:
            translation_url = hit['result']['url']
    return (original_url, translation_url)

    
if __name__ == "__main__":
    data = get_lyrics("stray kids", "social path")
    url = get_lyrics_url(data, "social path", "stray kids")
    print(url)