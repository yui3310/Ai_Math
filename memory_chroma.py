# memory_chroma.py (Pro 加強版)

import chromadb
from chromadb.utils import embedding_functions
import datetime
import uuid
import os

# 💾 資料庫設定
DB_PATH = "./chroma_db"

# 建立資料夾 (如果不存在)
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

client = chromadb.PersistentClient(path=DB_PATH)

# 🔥 升級 1: 使用支援多語言更強的模型
# 如果第一次跑會下載稍微久一點 (約 400MB)，但中文效果好很多
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

print(f"[記憶系統] 載入嵌入模型: {MODEL_NAME}...")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME
)

# 建立兩個集合：一個存流水帳 (Chat)，一個存重要事實 (Facts)
collection_chat = client.get_or_create_collection(
    name="chat_history",
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"} # 使用餘弦相似度，0~1 之間，越小越相似
)

collection_facts = client.get_or_create_collection(
    name="core_facts",
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"}
)

def _get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_memory(text: str, speaker: str):
    """一般對話記憶 (流水帳)"""
    timestamp = _get_timestamp()
    full_text = f"[{timestamp}] {speaker}: {text}"
    
    collection_chat.add(
        documents=[full_text],
        metadatas=[{"speaker": speaker, "timestamp": timestamp, "type": "chat"}],
        ids=[str(uuid.uuid4())]
    )

def add_important_fact(text: str):
    """
    🔥 升級 2: 重要事實記憶 (例如：使用者名字、居住地、喜好)
    這些記憶在搜尋時會有較高的優先級
    """
    timestamp = _get_timestamp()
    # 事實不需要加 Speaker，直接存內容
    full_text = f"[{timestamp}] 重要情報: {text}"
    
    collection_facts.add(
        documents=[full_text],
        metadatas=[{"timestamp": timestamp, "type": "fact"}],
        ids=[str(uuid.uuid4())]
    )
    print(f"⭐ [記憶] 已寫入重要事實: {text}")

def search_memory(query_text: str, n_results: int = 3, threshold: float = 0.4):
    """
    🔥 升級 3: 混合搜尋 + 品質過濾
    threshold: 相似度門檻 (0~1)，距離大於此值(越不相關)則丟棄。
    建議值 0.3~0.5。如果 AI 常常瞎掰無關的回憶，把這個值調低 (e.g. 0.3)。
    """
    
    # 1. 先搜「重要事實」(Facts) - 權重高
    fact_results = collection_facts.query(
        query_texts=[query_text],
        n_results=2 # 拿 2 個事實
    )
    
    # 2. 再搜「對話歷史」(Chat)
    chat_results = collection_chat.query(
        query_texts=[query_text],
        n_results=n_results + 2 # 多拿一點來過濾
    )
    
    final_memories = []

    # --- 處理事實 (Facts) ---
    if fact_results['documents']:
        for doc, dist in zip(fact_results['documents'][0], fact_results['distances'][0]):
            # Chroma 的 cosine distance: 0 (完全一樣) ~ 1 (完全不同)
            # 我們只要距離夠近的
            if dist < threshold: 
                final_memories.append(f"【重要設定】{doc}")

    # --- 處理對話 (Chat) ---
    # 我們需要把結果拿出來做「時間排序」，讓最近的對話優先級稍微高一點
    temp_chats = []
    if chat_results['documents']:
        for doc, meta, dist in zip(chat_results['documents'][0], chat_results['metadatas'][0], chat_results['distances'][0]):
            if dist < threshold:
                temp_chats.append({
                    "text": doc,
                    "date": meta["timestamp"],
                    "distance": dist
                })
    
    # 🔥 升級 4: 簡單的時間加權邏輯
    # 如果兩者相似度差不多，優先選時間比較近的
    # 這裡簡單處理：直接按照相似度排序 (Chroma 預設已排好)，但我們只取前 N 個有效結果
    
    for item in temp_chats:
        final_memories.append(item["text"])

    # 限制回傳數量
    final_memories = final_memories[:n_results]
    
    if not final_memories:
        return [] # 沒相關記憶就回傳空，不要硬塞
        
    return final_memories