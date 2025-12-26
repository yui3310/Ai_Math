# tts_module.py (Session加速 + 詳細Debug版)

import random
import requests
import pygame
import os
import re
import time

# ==========================================
# 🔧 配置區域
# ==========================================

API_URL = "http://127.0.0.1:9880"
LANGUAGE = 'zh'
TTS_VOLUME = 0.6

# ❗ 請確認路徑
REF_AUDIO_PATH = r"D:\ai_vtuber\GPT_SoVITS\GPT-SoVITS_MyGO-\参考音频\Anon干声素材\参考音频\サンキュー、あの頃の私なんだかこの辺.wav" 

EMOTION_SAMPLES = {
    "normal": [
        {"path": r"D:\ai_vtuber\GPT_SoVITS\GPT-SoVITS_MyGO-\参考音频\Anon干声素材\参考音频\サンキュー、あの頃の私なんだかこの辺.wav", "text": "こんにちは、今日はいい天気ですね。", "lang": LANGUAGE},
        # ... (請保留您原本完整的字典內容，這裡省略以節省篇幅) ...
    ]
}

# 補上預設值，避免 KeyError
if "normal" not in EMOTION_SAMPLES:
    EMOTION_SAMPLES["normal"] = [{"path": REF_AUDIO_PATH, "text": "你好", "lang": "zh"}]

DEFAULT_EMOTION = "normal"
GPT_MODEL_PATH = r"D:\ai_vtuber\GPT_SoVITS\GPT-SoVITS_MyGO-\SoVITS_weights\anon1_e8_s2184.pth"
SOVITS_MODEL_PATH = r"D:\ai_vtuber\GPT_SoVITS\GPT-SoVITS_MyGO-\SoVITS_weights\anon1_e8_s2184.pth"
TTS_TEMP_FILE = "tts_sovits_output.wav"

# 🔥 優化 1: 使用 Session，保持 HTTP 連線，減少延遲
session = requests.Session()

def load_character_model():
    if not os.path.exists(GPT_MODEL_PATH) or not os.path.exists(SOVITS_MODEL_PATH):
        return

    print(f"⏳ [SoVITS] 請求切換模型...")
    url = f"{API_URL}/set_model"
    params = {"gpt_model_path": GPT_MODEL_PATH, "sovits_model_path": SOVITS_MODEL_PATH}
    
    try:
        resp = session.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            print("✅ [SoVITS] 模型就緒")
    except Exception as e:
        print(f"❌ [SoVITS] API 未啟動或連線失敗: {e}")

load_character_model()

def text_to_speech(text: str, emotion: str = None, lang: str = LANGUAGE):
    if not text: return
    
    # 簡單過濾
    text = text.replace("，", ",")
    if not any(c.isalnum() for c in text): return

    # 選擇情感音訊
    target_list = EMOTION_SAMPLES.get(emotion, EMOTION_SAMPLES[DEFAULT_EMOTION])
    try:
        target_sample = random.choice(target_list)
    except:
        # 如果選不到，用預設的第一個
        target_sample = EMOTION_SAMPLES["normal"][0]

    payload = {
        "text": text,
        "text_language": LANGUAGE,
        "refer_wav_path": target_sample["path"],
        "prompt_text": target_sample["text"],
        "prompt_language": 'ja',
        "text_split_method": "cut0", 
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False,
        "top_k": 5, 
        "top_p": 0.8,
        "temperature": 0.8
    }

    url = f"{API_URL}/"

    #print(f"🔄 [TTS] 正在發送請求給 SoVITS... (Text: {text[:10]}...)")
    start_time = time.time()

    try:
        # 🔥 優化 2: 使用 session 發送，並加入超時保護 (120s)
        response = session.post(url, json=payload, timeout=120)
        
        duration = time.time() - start_time
        #print(f"✅ [TTS] 生成完畢! 耗時: {duration:.2f}秒")

        if response.status_code == 200:
            with open(TTS_TEMP_FILE, "wb") as f:
                f.write(response.content)
            
            # 只有檔案大於 1KB 才播放
            if os.path.getsize(TTS_TEMP_FILE) > 1000:
                _play_audio(TTS_TEMP_FILE)
            else:
                print("⚠️ [TTS] 生成的音訊檔案太小 (可能失敗)")
        
        elif response.status_code == 400:
            print(f"❌ [TTS] 參數錯誤 (400)。請檢查參考音訊路徑是否正確。")
            print(f"   路徑: {target_sample['path']}")
        else:
            print(f"❌ [TTS] 伺服器錯誤: {response.status_code}")

    except requests.exceptions.ReadTimeout:
        print("❌ [TTS] 逾時 (Timeout)! GPT-SoVITS 兩分鐘內沒有回應。")
        print("💡 建議: 請檢查您的顯卡 VRAM 是否已滿，或 GPT-SoVITS視窗是否被凍結。")
    except Exception as e:
        print(f"❌ [TTS] 連線錯誤: {e}")

def _play_audio(file_path):
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.set_volume(TTS_VOLUME) 
        pygame.mixer.music.play()
        
        # 這裡會卡住主程式直到播放完畢，這是正常的
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        time.sleep(0.3)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"❌ 播放失敗: {e}")

def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'[。？！;；]', text)
    return [s.strip() for s in sentences if s.strip()]