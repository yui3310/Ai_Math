# speaker_identity.py (繞過 torchaudio 讀取版)

import torch
import torchaudio
import os
import soundfile as sf  # 👈 直接使用 soundfile 讀取，不透過 torchaudio

# =========================================================
# 🚑 熱修復 1：解決 speechbrain 依賴問題
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
# =========================================================

from speechbrain.inference.speaker import EncoderClassifier

# 💾 設定你的聲音樣本路徑
MASTER_VOICE_FILE = "master_voice.wav" 

print("⏳ [聲紋] 正在載入 SpeechBrain 模型...")
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
except Exception as e:
    print(f"❌ [聲紋] 模型載入失敗: {e}")
    classifier = None

def get_embedding(wav_path):
    """將聲音檔案轉成聲紋向量"""
    if not os.path.exists(wav_path):
        print(f"⚠️ [聲紋] 找不到檔案: {wav_path}")
        return None
        
    if classifier is None:
        return None

    try:
        # 🚀 核彈級修復：完全繞過 torchaudio.load
        # 1. 使用 soundfile 直接讀取 (它回傳 numpy array)
        audio_array, sample_rate = sf.read(wav_path)
        
        # 2. 轉成 PyTorch Tensor
        signal = torch.from_numpy(audio_array).float()
        
        # 3. 處理維度 (Soundfile 是 [時間, 聲道], PyTorch 需要 [聲道, 時間])
        if len(signal.shape) == 1:
            # 單聲道: [T] -> [1, T]
            signal = signal.unsqueeze(0)
        else:
            # 多聲道: [T, C] -> [C, T] -> 取平均變單聲道 [1, T]
            signal = signal.transpose(0, 1)
            signal = signal.mean(dim=0, keepdim=True)
            
        # 計算聲紋
        with torch.no_grad():
            embeddings = classifier.encode_batch(signal)
            
        return embeddings
        
    except Exception as e:
        print(f"❌ [聲紋] 分析音訊失敗: {e}")
        # 印出更詳細錯誤以便除錯
        import traceback
        traceback.print_exc()
        return None

# 快取的聲紋
master_embedding = None

def load_master_voice():
    """程式啟動時，先記住的聲音"""
    global master_embedding
    if os.path.exists(MASTER_VOICE_FILE):
        print(f"✅ [聲紋] 讀取聲音樣本: {MASTER_VOICE_FILE}")
        master_embedding = get_embedding(MASTER_VOICE_FILE)
        
        if master_embedding is not None:
            print(f"✅ [聲紋] 聲紋註冊成功！")
        else:
            print(f"❌ [聲紋] 聲紋讀取失敗，請檢查 wav 檔案格式。")
            
    else:
        print(f"⚠️ [聲紋] 找不到樣本 ({MASTER_VOICE_FILE})")

def identify_speaker(current_audio_path, threshold=0.45):
    """比對當前的錄音"""
    if master_embedding is None or classifier is None:
        return False, 0.0 

    current_emb = get_embedding(current_audio_path)
    if current_emb is None:
        return False, 0.0

    # 計算餘弦相似度
    score = torch.nn.functional.cosine_similarity(master_embedding, current_emb, dim=-1)
    score_val = score.mean().item()
    
    # print(f"🔍 [聲紋] 相似度得分: {score_val:.4f}") 
    
    if score_val > threshold:
        return True, score_val
    else:
        return False, score_val

# 啟動時自動載入
load_master_voice()