from langchain_core.documents import Document
from datetime import datetime
import json
import os

DEBUG = True
LOG_DIR = "logs_auditoria"
LOG_FILE = os.path.join(LOG_DIR, "severino_debug.log")

# Garante que a pasta existe (Windows, Mac, Linux)
os.makedirs(LOG_DIR, exist_ok=True)

class ServerinoLogging:
    def __init__(self):
        pass

    def log_debug(self, titulo, data):
        if not DEBUG: return
        def convert(o): 
            if hasattr(o, "item"): return float(o)
            if isinstance(o, Document): return {"content": o.page_content[:200], "metadata": o.metadata}
            return str(o)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n{'='*80}\n[{timestamp}] {titulo}\n"
        log_entry += json.dumps(data, indent=2, ensure_ascii=False, default=convert)
        log_entry += f"\n{'='*80}\n"
        
        with open(LOG_FILE, "a", encoding="utf-8") as f: 
            f.write(log_entry)