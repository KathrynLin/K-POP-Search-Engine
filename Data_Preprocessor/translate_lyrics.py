import gzip
import json
from tqdm import tqdm
from googletrans import Translator

def translate_to_english(text):
    """
    使用 Google Translate 将文本翻译为英文
    """
    translator = Translator()
    try:
        translated = translator.translate(text, src='ko', dest='en')
        return translated.text
    except Exception as e:
        print(f"翻译失败: {e}")
        return None

# 输入和输出文件路径
input_file = "kpop_tracks.jsonl.gz"
output_file = "translated_kpop_tracks.jsonl.gz"

translator = Translator()

# 处理 .jsonl.gz 文件
with gzip.open(input_file, 'rt', encoding='utf-8') as gz_file:
    with gzip.open(output_file, 'wt', encoding='utf-8') as modified_gz_file:
        for line in tqdm(gz_file, desc="Translating lyrics"):
            # 解析每一行 JSON 数据
            data = json.loads(line)
            
            # 检查 text 是否为 null，且 original_lyrics 存在
            if data.get("text") is None and data.get("original_lyrics"):
                original_lyrics = data["original_lyrics"]
                
                # 翻译歌词
                translated_lyrics = translate_to_english(original_lyrics)
                if translated_lyrics:
                    data["text"] = translated_lyrics  # 将翻译后的文本存入 text 字段
            
            # 写入修改后的数据到新的 .jsonl 文件
            modified_gz_file.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"数据处理完成，已保存到 {output_file}")
