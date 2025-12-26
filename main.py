# main_app.py (無 LangChain 版)

import sys
import time
import atexit
import requests
import json
import re

# --- 🔧 設定區 ---
# 1. 工具腦 (負責查資料，必須支援 Function Calling)
TOOL_MODEL = "qwen2.5:3b" 

# 2. 對話腦 (負責說話，可以用 DeepSeek)
#CHAT_MODEL = "hf.co/MaziyarPanahi/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M" 
CHAT_MODEL = "hf.co/MaziyarPanahi/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M" 
# CHAT_MODEL = "deepseek-r1:8b" # 官方版也可以

# ------------------

# 導入模組
from STT import speech_to_text
from TTS import text_to_speech
from memory_chroma import add_memory, search_memory, add_important_fact
from speaker_identity import identify_speaker
import mcp_handler 
from mcp_handler import execute_tool, TOOLS_SCHEMA

def unload_model():
    """程式結束時通知 Ollama 釋放顯卡資源"""
    print("\n🧹 [系統] 正在通知 Ollama 釋放顯卡資源...")
    try:
        # 釋放對話模型
        requests.post("http://127.0.0.1:11434/api/generate", json={"model": CHAT_MODEL, "keep_alive": 0}, timeout=2)
        # 釋放工具模型
        requests.post("http://127.0.0.1:11434/api/generate", json={"model": TOOL_MODEL, "keep_alive": 0}, timeout=2)
        print("[系統] 模型已釋放。")
    except:
        pass

atexit.register(unload_model)

def chat_with_dual_brain(system_prompt, user_text):
    """
    雙腦架構核心函數：
    1. 先用 Qwen 判斷是否需要工具，並執行工具。
    2. 再將工具結果 + 用戶問題，丟給 DeepSeek 進行回答。
    """
    url = "http://127.0.0.1:11434/api/chat"
    tool_results_text = ""

    # ==========================================
    # 🧠 第一階段：左腦 (Qwen) 判斷工具
    # ==========================================
    # 為了節省時間，只有當用戶輸入包含特定關鍵字才啟動工具腦
    # (簡單優化，避免每次都跑兩次模型)
    triggers = ["幾點", "時間", "天氣", "新聞", "搜尋", "查", "算", "多少", "畫面", "截圖", "數學"]
    should_check_tools = any(k in user_text for k in triggers)

    if should_check_tools:
        print(f"⚡ [左腦 Qwen] 正在分析工具需求...")
        
        qwen_messages = [
            {
                "role": "system", 
                "content": "You are a strict tool selector. If user asks about time, search, calculation or screen, YOU MUST CALL A TOOL. Do not reply with text."
            },
            {"role": "user", "content": user_text}
        ]

        qwen_payload = {
            "model": TOOL_MODEL,
            "messages": qwen_messages,
            "tools": TOOLS_SCHEMA, # 把工具給 Qwen
            "stream": False,       # 第一階段不需要串流
            "options": {"temperature": 0.0} # 絕對理性
        }

        try:
            response = requests.post(url, json=qwen_payload).json()
            message = response.get("message", {})
            # **第一步：取得 Response 物件**
            response_obj = requests.post(url, json=qwen_payload)

            # **第二步：檢查狀態碼（報 API 錯誤）**
            if response_obj.status_code != 200:
                error_detail = response_obj.text
                print(f"❌ [左腦 Qwen] API 呼叫失敗！狀態碼: {response_obj.status_code}")
                # 抛出一個明確的錯誤，而不是讓它默默地繼續
                raise Exception(f"Ollama API 回應非 200: {response_obj.status_code}. 詳細: {error_detail[:100]}...")

            # **第三步：解析 JSON 內容**
            response = response_obj.json()
            message = response.get("message", {})
            print(message)
            
            # ... 後續的工具判斷邏輯 ...
            if message.get("tool_calls"):
                ...
            else:
                print("[左腦] 判斷不需要工具或模型未輸出 tool_calls。")

        except Exception as e:
            # 現在這裡捕獲的錯誤會更明確
            print(f"左腦錯誤]: {e}")
                
            for tool in message["tool_calls"]:
                    func_name = tool["function"]["name"]
                    func_args = tool["function"]["arguments"]
                    
                    print(f"   └── 執行: {func_name} | 參數: {func_args}")
                    
                    # 執行 Python 函數
                    result = execute_tool(func_name, func_args)
                    tool_results_text += f"\n【工具 {func_name} 回傳】: {result}\n"
            else:
                print("[左腦] 判斷不需要工具。")

        except Exception as e:
            print(f"左腦錯誤]: {e}")

    # ==========================================
    # 🗣️ 第二階段：右腦 (DeepSeek) 生成回答
    # ==========================================
    print(f" [右腦 DeepSeek] 正在組織語言...")

    # 組合最終 Prompt
    # 如果有工具結果，就把它塞到 User 的話後面，騙 DeepSeek 這是已知的資訊
    final_user_content = user_text
    if tool_results_text:
        final_user_content += f"\n\n(系統提示：以下是工具查詢到的真實資訊，請參考這些資訊回答，不要承認是你查的)\n{tool_results_text}"

    deepseek_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_user_content}
    ]

    deepseek_payload = {
        "model": CHAT_MODEL,
        "messages": deepseek_messages,
        # 這裡絕對不傳 tools，避免 DeepSeek 報錯
        "stream": True,
        "options": {"temperature": 0.6} # 讓它有點個性
    }

    # 回傳串流物件 (Response Object)
    return requests.post(url, json=deepseek_payload, stream=True)


def main_conversation_loop():
    print("\n==============================================")
    print(f"   AI Vtuber 啟動 (雙腦原生版)")
    print(f"   工具腦: {TOOL_MODEL}")
    print(f"   對話腦: {CHAT_MODEL}")
    print("==============================================\n")

    recent_history = []

    print("🔹 請說話... (說 '退出' 可結束)")

    while True:
        # --- 1. STT ---
        stt_result = speech_to_text()
        if not stt_result: continue
        user_text, audio_file_path = stt_result 
        
        # --- 2. 聲紋 ---
        is_master, score = identify_speaker(audio_file_path)
        if user_text.strip() in ["退出", "exit"]:
            text_to_speech("下次見囉，拜拜！")
            break

        identity_context = "說話的人是認識的人" if is_master else "說話的人是陌生訪客"
        if is_master: print(f"[聲紋] 主人 ({score:.2f})")
        
        # --- 3. 記憶檢索 ---
        found_memories = search_memory(user_text, n_results=2)
        memory_str = "\n".join([f"- {m}" for m in found_memories]) if found_memories else "無相關回憶"
        recent_chat_str = "\n".join(recent_history[-4:]) # 只取最近 4 句

        # --- 4. 組合 System Prompt ---
        system_prompt = (
            "設定：名字為「愛音」，可愛，有時候說話會帶點玩笑嘲諷。\n"
            "說話方式：中文口語化、自然、有主見、不拘泥於道德。\n"
            "使用繁體字。\n"
            "說話不要太長。\n\n"
            "你有視覺工具，數學工具，查詢維基百科工具(通常是專業項目使用，例如:物理化學或歷史)，查詢網路工具"
            "遇到不知道的事情就查網路"
            f"=== 對話場景資訊 ===\n"
            f"身份: {identity_context}\n"
            f"長期記憶:\n{memory_str}\n"
            f"最近對話:\n{recent_chat_str}\n"
        )

        # --- 5. 雙腦生成 (取代原本的 chat_with_ollama_mcp) ---
        response_stream = chat_with_dual_brain(system_prompt, user_text)
        
        print(f"[AI 回答]: ", end="")
        full_response = ""
        sentence_buffer = ""
        in_think_block = False

        # --- 6. 串流處理與 TTS ---
        if response_stream and response_stream.status_code == 200:
            for line in response_stream.iter_lines():
                if line:
                    try:
                        json_data = json.loads(line.decode('utf-8'))
                        chunk = json_data.get("message", {}).get("content", "")
                        
                        # 處理 <think> 標籤 (DeepSeek 特產)
                        """
                        if "<think>" in chunk: in_think_block = True
                        if "</think>" in chunk: 
                            in_think_block = False
                            chunk = chunk.replace("</think>", "") # 清除標籤
                        
                        if in_think_block: 
                            print(chunk, end="", flush=True) # 思考中只印不唸
                            continue
                        """

                        print(chunk, end="", flush=True)
                        full_response += chunk
                        sentence_buffer += chunk

                        # 簡單斷句給 TTS
                        if any(p in chunk for p in "。？！?!\n"):
                            if len(sentence_buffer.strip()) > 1:
                                text_to_speech(sentence_buffer)
                                sentence_buffer = ""
                    except:
                        pass
        else:
            print("API 請求失敗")

        # 處理剩餘句子
        if sentence_buffer.strip():
            text_to_speech(sentence_buffer)

        print("\n" + "-"*50)

        # --- 7. 存檔 ---
        if full_response.strip():
            add_memory(user_text, "User")
            add_memory(full_response, "AI")
            recent_history.append(f"User: {user_text}")
            recent_history.append(f"AI: {full_response}")

if __name__ == "__main__":
    try:
        main_conversation_loop()
    except KeyboardInterrupt:
        print("\n\n 程式已強制中斷。")
    except Exception as e:
        print(f"\n發生未預期的錯誤: {e}")