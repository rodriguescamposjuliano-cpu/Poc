import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import warnings

os.environ["HF_HUB_OFFLINE"] = "1" # Força modo offline
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Evita outro warning comum

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning) # Ignora avisos de versão

def gerar_arquivos_validacao():
    # 1. Configuração do Banco (Use o mesmo modelo do seu indexador!)
    print("Conectando ao banco de dados para extração de conferência...")
    # Ajustado para o modelo multilingual que você usou no indexador original
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    if not os.path.exists("chroma_db"):
        print("Erro: Pasta 'chroma_db' não encontrada.")
        return

    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    # 2. Recupera todos os dados brutos do banco
    dados = vectorstore.get()
    documentos = dados['documents']
    metadatas = dados['metadatas']

    if not documentos:
        print("O banco de dados está vazio.")
        return

    # 3. Organiza os textos por Tenant
    textos_por_tenant = {}
    for doc, meta in zip(documentos, metadatas):
        tid = meta.get('tenant_id', 'SEM_ID')
        if tid not in textos_por_tenant:
            textos_por_tenant[tid] = []
        
        # Capturamos o ID do chunk e a Referência (Artigo/Título)
        chunk_id = meta.get('chunk_id', 'N/A')
        referencia = meta.get('referencia', 'N/A')
        
        # Guardamos os metadados junto com o conteúdo
        textos_por_tenant[tid].append({
            "chunk": chunk_id,
            "ref": referencia,
            "conteudo": doc
        })

    # 4. Gera os arquivos físicos
    pasta_saida = "validacao_textos"
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    print(f"📂 Gerando arquivos em '{pasta_saida}'...")

    for tid, blocos in textos_por_tenant.items():
        # Ordena pelos IDs de chunk para manter a sequência lógica
        blocos.sort(key=lambda x: x['chunk'])
        
        nome_arquivo = os.path.join(pasta_saida, f"CONFERENCIA_TENANT_{tid}.txt")
        
        with open(nome_arquivo, "w", encoding="utf-8") as f:
            f.write(f"=== RELATÓRIO DE AUDITORIA DE INDEXAÇÃO ===\n")
            f.write(f"CONDOMÍNIO (TENANT): {tid}\n")
            f.write(f"TOTAL DE CHUNKS: {len(blocos)}\n")
            f.write("="*60 + "\n\n")

            for item in blocos:
                f.write(f"📍 ID DO CHUNK: {item['chunk']}\n")
                f.write(f"📜 REFERÊNCIA: {item['ref']}\n")
                f.write(f"📝 CONTEÚDO:\n{item['conteudo']}\n")
                f.write("-" * 60 + "\n\n")
        
        print(f"✅ Arquivo gerado: {nome_arquivo}")

if __name__ == "__main__":
    gerar_arquivos_validacao()
    print("\nAuditoria concluída! Verifique a pasta 'validacao_textos'.")