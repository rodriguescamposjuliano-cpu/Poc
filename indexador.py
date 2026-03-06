import os
import shutil
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURAÇÕES DO AMBIENTE ---
# Ajuste os caminhos conforme seu sistema (macOS/Linux/Windows)
POPPLER_PATH = "/opt/homebrew/bin"
TESSDATA_DIR = "/opt/homebrew/share/tessdata"
PDF_PATH = "regimento.pdf"
CHROMA_DIR = "chroma_db"

def limpar_banco_antigo():
    """Remove o banco anterior para evitar duplicidade ou lixo de versões falhas."""
    if os.path.exists(CHROMA_DIR):
        print(f"♻️ Removendo banco antigo em '{CHROMA_DIR}'...")
        shutil.rmtree(CHROMA_DIR)

def load_pdf_with_ocr(pdf_path: str):
    """Converte PDF em texto usando OCR de alta resolução (300 DPI)."""
    print(f"🔍 Iniciando OCR de alta fidelidade: {pdf_path}")
    
    # DPI=300 é o padrão ouro para documentos jurídicos com fontes pequenas
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)

    documents = []
    for i, page in enumerate(pages):
        num_pagina = i + 1
        # Extração de texto em Português
        text = pytesseract.image_to_string(
            page, 
            lang="por", 
            config=f"--tessdata-dir {TESSDATA_DIR}"
        )

        if text.strip():
            # Mantemos a referência da página nos metadados
            documents.append(
                Document(
                    page_content=text, 
                    metadata={"page": num_pagina}
                )
            )
            print(f"📄 Página {num_pagina} processada.")
    return documents

# --- EXECUÇÃO DO FLUXO ---

# 1. Reset
limpar_banco_antigo()

# 2. Captura de Dados
docs = load_pdf_with_ocr(PDF_PATH)

print("\n✂️ Fatiando o regimento com foco em artigos longos...")

# 3. Divisão Robusta (Chunking)
# Aumentamos o tamanho do bloco para 3500 para o Artigo 31 caber inteiro
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3500,       
    chunk_overlap=500,     # 500 caracteres de sobra para manter o contexto entre blocos
    separators=[
        "\nArt. ",          # Prioridade máxima: Tentar começar um bloco sempre em um Artigo
        "\nParágrafo único", 
        "\n§",              
        "\n\n",             # Quebras de parágrafo duplo
        "\n",               # Quebras de linha simples
        ". "                # Final de frases
    ]
)

splits = text_splitter.split_documents(docs)
print(f"📦 Total de blocos gerados: {len(splits)}")

# 4. Embeddings e Persistência
print(f"🧠 Gerando inteligência vetorial (Embeddings)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print(f"\n✅ SUCESSO! O banco '{CHROMA_DIR}' está pronto para ser usado pelo Severino.")
print(f"DICA: Rode agora o seu script de debug para ver o Artigo 31 completo.")