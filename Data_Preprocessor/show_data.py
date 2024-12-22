import gzip
import json

def read_first_jsonl_from_gz(file_path):
    # 打开并解压缩文件
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        # 读取第一行
        first_line = f.readline()
        # 将JSON字符串解析为Python字典
        first_json = json.loads(first_line)
        # 展示内容
        print("第一个jsonl的内容:")
        print(json.dumps(first_json, indent=4, ensure_ascii=False))

# 示例用法
file_path = 'translated_kpop_tracks.jsonl.gz'  
read_first_jsonl_from_gz(file_path)
