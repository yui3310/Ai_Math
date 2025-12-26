# ollama_module.py (支援流式處理 STREAMING)

import requests
import json
from typing import Generator

# 設置 Ollama 服務的基礎 URL
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHECK_URL = "http://localhost:11434/" 



def get_ollama_response_stream(prompt: str, model_name: str = "qwen3:1.7b") -> Generator[str, None, None]:
    """
    將文字提示發送給 Ollama API，並以流式 (Streaming) 方式獲取回覆。
    
    Returns:
        Generator[str]: 逐塊 (chunk) 生成的文字回覆。
    """
    headers = {
        'Content-Type': 'application/json',
    }
    

    #full_prompt = f"{character_setting}\n\n{prompt}\n："
    # *** 關鍵修改：設置 "stream": True ***
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "max_tokens": 256,
        "options": {
            "keep_alive": "10m", # 保持模型在記憶體中 10 分鐘
            "temperature": 0.85, 
            "top_p": 0.95,
            # 懲罰重複 (避免它一直跳針)
            "repeat_penalty": 1.1
        }
    }
    
    try:
        print(f"🧠 [OLLAMA] 正在請求模型 ({model_name})，開始流式接收...")
        
        # 進行連線測試 (可選，但建議保留)
        requests.get(OLLAMA_CHECK_URL, timeout=5).raise_for_status()
        
        # 發送 POST 請求，設置 stream=True 讓 requests 模組返回一個迭代響應
        response = requests.post(
            OLLAMA_BASE_URL, 
            headers=headers, 
            data=json.dumps(data), 
            timeout=120, # 將超時時間設長一點，以防萬一
            stream=True # 啟用 requests 的流式讀取
        )
        
        # 處理 HTTP 錯誤（例如模型不存在 404）
        if response.status_code != 200:
             error_msg = f"❌ [OLLAMA] 請求失敗，狀態碼: {response.status_code}. 詳細錯誤: {response.text}"
             yield error_msg
             return
             
        # 逐行讀取流式響應
        for line in response.iter_lines():
            if line:
                try:
                    # 每行是一個 JSON 對象
                    data_chunk = json.loads(line.decode('utf-8'))
                    
                    # 提取文本部分
                    chunk_text = data_chunk.get("response", "")
                    
                    # 如果不是結束標記，則產生 (yield) 文本
                    if not data_chunk.get("done"):
                        yield chunk_text
                        
                except json.JSONDecodeError:
                    # 處理可能損壞的 JSON 行
                    continue

    except requests.exceptions.ConnectionError:
        yield "❌ [OLLAMA] 無法連接到 Ollama 服務。請確認服務是否正在運行。"
    except requests.exceptions.RequestException as e:
        yield f"❌ [OLLAMA] 請求 Ollama 時發生錯誤: {e}"
