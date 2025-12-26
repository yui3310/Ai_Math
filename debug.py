import requests
import json

# 1. 設定模型 (請確保您有 pull qwen2.5:7b)
MODEL = "qwen2.5:7b"

print(f"🔍 正在測試模型: {MODEL}")

# 2. 定義工具 (這是標準 Ollama 格式)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# 3. 定義對話
messages = [
    {
        "role": "system", 
        "content": "You are a helpful assistant. If asked about time, you MUST use the get_current_time tool."
    },
    {
        "role": "user", 
        "content": "現在幾點？"
    }
]

# 4. 發送請求 (注意 stream 必須是 False 才能看到 tool_calls)
payload = {
    "model": MODEL,
    "messages": messages,
    "tools": tools,
    "stream": False, 
    "options": {"temperature": 0.1} # 溫度調低，讓它變笨但聽話
}

try:
    print("🚀 發送請求給 Ollama...")
    response = requests.post("http://127.0.0.1:11434/api/chat", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        msg = result.get("message", {})
        
        print("\n=== 🟢 Ollama 回傳的原始資料 ===")
        print(json.dumps(msg, indent=2, ensure_ascii=False))
        print("==============================\n")

        if msg.get("tool_calls"):
            print("✅ 成功！模型回傳了 tool_calls！")
            print(f"   工具名稱: {msg['tool_calls'][0]['function']['name']}")
        else:
            print("❌ 失敗！模型直接回傳了 content (文字)，沒有用工具。")
            print(f"   AI 說: {msg.get('content')}")
            
    else:
        print(f"❌ API 錯誤: {response.status_code} - {response.text}")

except Exception as e:
    print(f"❌ 連線失敗: {e}")