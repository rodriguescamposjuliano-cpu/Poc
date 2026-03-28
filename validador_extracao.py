import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def gerar_arquivos_validacao():
    # 1. Configuração do Banco
    print("Conectando ao banco de dados para extração de conferência...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists("chroma_db"):
        print("Erro: Pasta 'chroma_db' não encontrada.")
        return

    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    # 2. Recupera todos os dados brutos do banco
    # O método .get() retorna dicionários com 'documents' e 'metadatas'
    dados = vectorstore.get()
    documentos = dados['documents']
    metadatas = dados['metadatas']

    if not documentos:
        print("O banco de dados está vazio.")
        return

    # 3. Organiza os textos por Tenant
    # Estrutura: { '001': ["texto pag 1", "texto pag 2"], '002': [...] }
    textos_por_tenant = {}
    for doc, meta in zip(documentos, metadatas):
        tid = meta.get('tenant_id', 'SEM_ID')
        if tid not in textos_por_tenant:
            textos_por_tenant[tid] = []
        
        # Guardamos uma tupla (página, conteúdo) para ordenar depois
        pag = meta.get('page', 0)
        textos_por_tenant[tid].append((pag, doc))

    # 4. Gera os arquivos físicos
    pasta_saida = "validacao_textos"
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    print(f"📂 Gerando arquivos em '{pasta_saida}'...")

    for tid, blocos in textos_por_tenant.items():
        # Ordena por número de página para facilitar a leitura
        blocos.sort(key=lambda x: x[0])
        
        nome_arquivo = f"{pasta_saida}/CONFERENCIA_TENANT_{tid}.txt"
        
        with open(nome_arquivo, "w", encoding="utf-8") as f:
            f.write(f"=== RELATÓRIO DE EXTRAÇÃO PARA AUDITORIA ===\n")
            f.write(f"TENANT ID: {tid}\n")
            f.write(f"TOTAL DE BLOCOS ENCONTRADOS: {len(blocos)}\n")
            f.write("="*50 + "\n\n")

            for pag, conteúdo in blocos:
                f.write(f"--- [PÁGINA {pag}] ---\n")
                f.write(f"{conteúdo}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"Arquivo gerado: {nome_arquivo}")

if __name__ == "__main__":
    gerar_arquivos_validacao()
    print("\nPronto! Abra a pasta 'validacao_textos' e compare com seus PDFs.")