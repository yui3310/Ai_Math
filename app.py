from flask import Flask, render_template, request, Response, stream_with_context, jsonify
import json
import atexit
import sys
import os # 記得導入 os

# --- 導入模組 ---
# 假設 main_app 中包含了所有核心邏輯和模型配置
from main_app import (
    chat_with_dual_brain, 
    unload_model,
    text_to_speech,    
    speech_to_text,    
    identify_speaker,  
    search_memory,     
    add_memory         
)

# 導入圖片與PDF處理函數
from mcp_handler import process_uploaded_image, process_pdf_pipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key' 

# ==============================================================================
# 🚨【終極解法】暴力解除大小限制 (確保能上傳大型檔案/Base64)
# ==============================================================================
# 將總上傳限制設為一個極大值 (16GB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024 
# 將表單記憶體限制 (處理 Base64 字串) 設為 1GB
app.config['MAX_FORM_MEMORY_SIZE'] = 1024 * 1024 * 1024 
app.config['MAX_HEADERS_LIST'] = 1024 * 1024 

# ------------------------------------------------------------------------------

chat_history = [] 
atexit.register(unload_model) 

@app.route("/")
def index():
    return render_template("index.html", history=chat_history)

@app.route("/chat", methods=["POST"])
def chat():
    # 1. 獲取輸入
    try:
        user_text = request.form.get("user_input", "").strip()
        image_base64 = request.form.get("image_base64", "").strip()
        pdf_file = request.files.get("pdf_file")
        pdf_page = request.form.get("pdf_page", "1")
    except Exception as e:
        print(f"❌ 接收資料失敗: {e}")
        return Response(json.dumps({"text": f"❌ 資料傳輸失敗: {e}", "done": True}) + "\n", mimetype='application/jsonlines')
    
    audio_file_path = None
    identity_context = "使用者正在使用文字介面與你交談" 
    
    # --- 混合輸入邏輯 ---
    if user_text and not image_base64 and not pdf_file:
        print(f" [Web文字輸入]: {user_text}")
    elif not user_text and not image_base64 and not pdf_file: 
        print(" [Web語音模式]...")
        stt_result = speech_to_text() 
        if not stt_result:
            return Response(json.dumps({"text": "❌ (未偵測到語音)", "done": True}) + "\n", mimetype='application/jsonlines')
        user_text, audio_file_path = stt_result
        if audio_file_path:
            is_master, score = identify_speaker(audio_file_path)
            identity_context = "認識的人" if is_master else "陌生訪客"

    # --- 圖片處理 ---
    if image_base64:
        vision_analysis = process_uploaded_image(image_base64, user_text)
        user_text = vision_analysis
        print(" [Web圖片] 已轉換為文字描述")

    # --- PDF 處理 ---
    if pdf_file:
        try:
            page_num = int(pdf_page)
            pdf_bytes = pdf_file.read() 
            print(f" 📄 [Web PDF] 接收到檔案，大小: {len(pdf_bytes)/1024/1024:.2f} MB")
            pdf_analysis = process_pdf_pipeline(pdf_bytes, page_num, user_text)
            user_text = pdf_analysis
        except Exception as e:
            print(f"PDF 錯誤: {e}")
            user_text = f"PDF 處理發生錯誤: {str(e)}"

    if "退出" in user_text:
        return Response(json.dumps({"text": "掰掰！", "done": True}) + "\n", mimetype='application/jsonlines')

    # 2. Prompt
    found_memories = search_memory(user_text, n_results=2)
    memory_str = "\n".join([f"- {m}" for m in found_memories]) if found_memories else "無相關回憶"
    recent_msgs = chat_history[-100:] 
    recent_chat_str = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in recent_msgs])

    system_prompt = (
        "你喜歡解數學題目，看到題目會喜歡推導，並擅長使用 WolframAlpha\n"
        f"=== 對話場景資訊 ===\n"
        f"身份: {identity_context}\n"
        f"長期記憶:\n{memory_str}\n"
        f"最近對話:\n{recent_chat_str}\n"
    )

    history_log = "[使用者上傳檔案]" if (image_base64 or pdf_file) else user_text
    chat_history.append({"speaker": "user", "text": history_log})

    # 3. 雙腦生成
    try:
        response_stream = chat_with_dual_brain(system_prompt, user_text)
    except Exception as e:
        return Response(json.dumps({"text": f"❌ Error: {e}", "done": True}) + "\n", status=500, mimetype='application/jsonlines')

    # 4. 串流回應
    def generate_response(stream):
        full_ai_response = ""
        in_think_block = False 
        
        if stream and hasattr(stream, 'iter_lines'):
            for line in stream.iter_lines():
                if line:
                    try:
                        json_data = json.loads(line.decode('utf-8'))
                        chunk = json_data.get("message", {}).get("content", "")
                        
                        if chunk:
                            # --- <think> 過濾演算法 ---
                            content_to_yield = ""
                            temp_chunk = chunk
                            
                            while len(temp_chunk) > 0:
                                if not in_think_block:
                                    start_idx = temp_chunk.find("<think>")
                                    if start_idx != -1:
                                        content_to_yield += temp_chunk[:start_idx]
                                        in_think_block = True
                                        temp_chunk = temp_chunk[start_idx + 7:]
                                    else:
                                        content_to_yield += temp_chunk
                                        temp_chunk = ""
                                else:
                                    end_idx = temp_chunk.find("</think>")
                                    if end_idx != -1:
                                        in_think_block = False
                                        temp_chunk = temp_chunk[end_idx + 8:]
                                    else:
                                        temp_chunk = ""
                            
                            if content_to_yield:
                                full_ai_response += content_to_yield
                                yield json.dumps({"text": content_to_yield, "done": False}) + "\n"
                        if json_data.get("done"): break
                    except: break
        
        if full_ai_response.strip():
            chat_history.append({"speaker": "ai", "text": full_ai_response})
            add_memory(user_text, "User")
            add_memory(full_ai_response, "AI")
            yield json.dumps({"text": "", "done": True, "full_text": full_ai_response}) + "\n"
        else:
            yield json.dumps({"text": "(AI 無回應)", "done": True}) + "\n"

    return Response(stream_with_context(generate_response(response_stream)), mimetype='application/jsonlines')

@app.route("/tts", methods=["POST"])
def generate_audio():
    """生成音訊檔並傳回 (含防呆檢查)"""
    data = request.json
    text_to_speak = data.get("text", "")
    
    if not text_to_speak:
        return jsonify({"error": "No text provided"}), 400

    try:
        # 呼叫 TTS
        audio_file_path = text_to_speech(text_to_speak)
        
        # 檢查是否真的有回傳路徑，以及檔案是否存在
        if not audio_file_path or not isinstance(audio_file_path, str):
            print(f"⚠️ TTS 生成失敗: text_to_speech 回傳了 {type(audio_file_path)}")
            return jsonify({"error": "TTS generation failed (Internal Error)"}), 500
            
        if not os.path.exists(audio_file_path):
            print(f"⚠️ TTS 檔案找不到: {audio_file_path}")
            return jsonify({"error": "TTS file not found"}), 500

        # 讀取檔案
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()
            
        return Response(audio_data, mimetype="audio/wav")
        
    except Exception as e:
        print(f"❌ TTS 路由發生錯誤: {e}")
        return jsonify({"error": f"TTS exception: {str(e)}"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "檔案太大", "detail": "Server rejected payload (413)"}), 413

if __name__ == "__main__":
    print("="*50)
    print(f"🚀 Flask 伺服器啟動中...")
    print(f"📂 MAX_CONTENT_LENGTH 設定為: {app.config['MAX_CONTENT_LENGTH']}")
    print(f"🧠 MAX_FORM_MEMORY_SIZE 設定為: {app.config['MAX_FORM_MEMORY_SIZE'] / (1024*1024):.2f} MB")
    print("="*50)
    
    app.run(debug=True, port=5000, threaded=False, use_reloader=False)