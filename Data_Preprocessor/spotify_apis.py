import requests
import json
import gzip
from bs4 import BeautifulSoup

# 从配置文件中加载配置
with open('config.json') as f:
    config = json.load(f)

SPOTIFY_CLIENT_ID = config["SPOTIFY_CLIENT_ID"]
SPOTIFY_CLIENT_SECRET = config["SPOTIFY_CLIENT_SECRET"]
GENIUS_ACCESS_TOKEN = config["GENIUS_ACCESS_TOKEN"]

# 获取 Spotify 访问令牌
auth_url = 'https://accounts.spotify.com/api/token'
auth_response = requests.post(auth_url, {
    'grant_type': 'client_credentials',
    'client_id': SPOTIFY_CLIENT_ID,
    'client_secret': SPOTIFY_CLIENT_SECRET,
})
access_token = auth_response.json()['access_token']
headers = {'Authorization': f'Bearer {access_token}'}

# Genius API 访问令牌和 User-Agent
genius_headers = {
    'Authorization': f'Bearer {GENIUS_ACCESS_TOKEN}',
    'User-Agent': 'Mozilla/5.0'
}

# Genius API 搜索歌词
def get_lyrics_links(artist, song):
    params = {"q": song}
    url = "https://api.genius.com/search"
    response = requests.get(url, params=params, headers=genius_headers)
    if response.status_code == 200:
        data = response.json()['response']['hits']
        original_url, translation_url = None, None
        for hit in data:
            if song.lower() in hit['result']['full_title'].lower() and artist.lower() in hit['result']['artist_names'].lower() and not original_url:                
                original_url = hit['result']['url']
            if song.lower() in hit['result']['full_title'].lower() and "English Translation" in hit['result']['full_title']:
                translation_url = hit['result']['url']
        return original_url, translation_url
    else:
        print("Error accessing Genius API:", response.text)
        return None, None

# 爬取 Genius 歌词页面
def scrape_lyrics(url):
    if url:
        page = requests.get(url)
        html = BeautifulSoup(page.text, 'html.parser')
        lyrics_divs = html.find_all("div", {"data-lyrics-container": "true"})
    
        lyrics = ""
        for div in lyrics_divs:
            lyrics += div.get_text(separator="\n") + "\n"
    
        return lyrics.strip()
    else:
        return None


# Spotify 搜索并获取歌曲信息
artist_name = "Stray Kids"
search_url = 'https://api.spotify.com/v1/search'
params = {'q': f'artist:{artist_name}', 'type': 'track', 'limit': 5}
response = requests.get(search_url, headers=headers, params=params)
tracks = response.json()['tracks']['items']

# 将数据写入 jsonl.gz 文件
with gzip.open('kpop_tracks.jsonl.gz', 'at', encoding='utf-8') as file:
    for track in tracks:
        track_id = track['id']
        track_name = track['name']
        artists = ', '.join([artist['name'] for artist in track['artists']])
        release_date = track['album']['release_date']
        
        # 获取音频特征
        audio_features_url = f'https://api.spotify.com/v1/audio-features/{track_id}'
        audio_features_response = requests.get(audio_features_url, headers=headers)
        audio_features = audio_features_response.json()

        # 获取 Genius 歌词链接
        original_url, translation_url = get_lyrics_links(artist_name, track_name)

        # 爬取歌词内容
        original_lyrics = scrape_lyrics(original_url)
        english_lyrics = scrape_lyrics(translation_url)
        #print(f"Scraped lyrics for {track_name}:{english_lyrics}")
        # 构建 JSON 数据
        track_data = {
            'track_id': track_id,
            'title': track_name,
            'artists': artists,
            'release_date': release_date,
            'danceability': audio_features.get('danceability'),
            'energy': audio_features.get('energy'),
            'original_lyrics': original_lyrics,
            'english_lyrics': english_lyrics
        }

        # 写入 JSONL 文件
        file.write(json.dumps(track_data) + '\n')



print(scrape_lyrics("https://genius.com/Stray-kids-social-path-lyrics"))