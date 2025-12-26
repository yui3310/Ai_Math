# vllm_module.py (專用於 vLLM Qwen2.5-3B)

from openai import OpenAI
from typing import Generator
import os

# --- 🔧 設定區 ---
# vLLM 伺服器地址 (WSL2 的 localhost 通常可以互通)
VLLM_API_URL = "http://localhost:8000/v1"

# 模型名稱 (必須跟 WSL 啟動指令的一模一樣)
MODEL_NAME = "Qwen/Qwen3-4B-AWQ"

# 初始化 OpenAI 客戶端 (vLLM 相容 OpenAI API)
client = OpenAI(
    base_url=VLLM_API_URL,
    api_key="EMPTY" # 本地端不需要 Key
)

def get_llm_response_stream(prompt: str) -> Generator[str, None, None]:
    """
    發送 Prompt 給 vLLM 並流式接收回覆
    """
    try:
        print(f"🚀 [vLLM] 發送請求給 Qwen 3B...")
        
        # 發送聊天請求
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                # vLLM 支援標準 Chat 格式，這裡直接把整包 prompt 塞給 user
                # 因為您的 main_app 已經把 System Prompt 組合進去了
                {"role": "user", "content": prompt}
            ],
            stream=True,
            
            # --- 🎭 參數調校 (針對 Qwen 3B 優化) ---
            temperature=0.85, # 創意度 (0.7~0.9)
            top_p=0.95,       # 多樣性
            max_tokens=512,   # 限制回答長度 (避免長篇大論)
            frequency_penalty=0.1, # 減少重複
            presence_penalty=0.1
        )

        # 流式接收
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content

    except Exception as e:
        error_msg = f"❌ [vLLM] 連線失敗: {e}"
        print(error_msg)
        yield error_msg

# 測試區塊
if __name__ == "__main__":
    print("正在測試 vLLM 連線...")
    for text in get_llm_response_stream("你好，請自我介紹。"):
        print(text, end="", flush=True)