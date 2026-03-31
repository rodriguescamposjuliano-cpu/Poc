import os
import shutil
import re
import fitz
from pdf2image import convert_from_path
import pytesseract

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# CONFIGURAÇÕES
POPPLER_PATH = "/opt/homebrew/bin"  # Ajuste conforme seu sistema
CHROMA_DIR = "chroma_db"
PASTA_REGIMENTOS = "./regimentos_input"

# 1. Correção de Erros Comuns de OCR
mapa_correcao_ocr = {
    r"(?i)intemo": "Interno", r"(?i)intimo": "Interno",
    r"(?i)iniração": "infração", r"(?i)muita": "multa",
    r"(?i)peia": "pela", r"(?<=[\s\n])[8S]\s+(?=\d+º)": "§ ",
    r"(?i)regimento": "Regimento"
}

# -------------------------------
# LIMPEZA E NORMALIZAÇÃO
# -------------------------------
def limpar_texto(texto):
    if not texto: return ""
    
    # Remove ruídos de cabeçalho e rodapé comuns (Página X, Nome do Condomínio)
    texto = re.sub(r'(?i)Regimento Interno.*|Residencial Wish.*|Pág\w*\.?\s*\d+', '', texto)
    
    # Corrige erros de OCR em símbolos jurídicos
    texto = re.sub(r'\$(\d+º)', r'§\1', texto)
    texto = re.sub(r'\b8(\d+º)', r'§\1', texto)
    
    # Remove quebras de linha no meio de frases (hifenização)
    texto = re.sub(r'(\w)-\n(\w)', r'\1\2', texto)
    
    # Normaliza espaços múltiplos e quebras excessivas
    texto = re.sub(r'[ \t]+', ' ', texto)
    return texto.strip()

# -------------------------------
# DIVISOR INTELIGENTE HÍBRIDO
# -------------------------------
def dividir_documento(texto):
    blocos = []
    
    for erro, correcao in mapa_correcao_ocr.items():
        texto = re.sub(erro, correcao, texto)
    
    # PADRÃO 1: Seções Numeradas (ex: 5. DA ADMINISTRAÇÃO ou 22. ANIMAIS)
    # Procura: Início de linha + Número + Ponto + Espaço + Título em Maiúsculas
    padrao_topicos = r'(?:\n|^)(\d+\.\s+[A-ZÀ-Ú\s]{5,})'
    
    # PADRÃO 2: Artigos (ex: Art. 1º ou Artigo 22)
    padrao_artigos = r'(?:\n|^)(Art\.?\s*\d+º?)'

    # Verifica qual padrão é predominante no documento
    if len(re.findall(padrao_topicos, texto)) > 5:
        # Divisão por Tópicos/Assuntos
        partes = re.split(padrao_topicos, texto)
        i = 1
        while i < len(partes):
            titulo = partes[i].strip()
            conteudo = partes[i+1].strip() if (i+1) < len(partes) else ""
            
            texto_bloco = f"--- {titulo} ---\n{conteudo}"
            if len(texto_bloco) > 50:
                blocos.append({"tipo": "topico", "id": titulo, "conteudo": texto_bloco})
            i += 2
            
    elif len(re.findall(padrao_artigos, texto)) > 3:
        # Divisão por Artigos
        partes = re.split(padrao_artigos, texto)
        i = 1
        while i < len(partes):
            num_art = partes[i].strip()
            conteudo = partes[i+1].strip() if (i+1) < len(partes) else ""
            
            texto_bloco = f"{num_art} {conteudo}"
            if len(texto_bloco) > 30:
                blocos.append({"tipo": "artigo", "id": num_art, "conteudo": texto_bloco})
            i += 2
    
    # Fallback se não encontrar estrutura clara
    if not blocos:
        # Divide por tamanho fixo com sobreposição (overlap)
        tamanho = 1000
        for i in range(0, len(texto), tamanho - 200):
            chunk = texto[i:i+tamanho]
            blocos.append({"tipo": "chunk", "id": str(i), "conteudo": chunk})
            
    return blocos

# -------------------------------
# PROCESSAMENTO DE PDF (TEXTO + OCR)
# -------------------------------
def extrair_texto_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texto_completo = ""
    
    for page in doc:
        # Tenta extrair texto nativo
        texto_pag = page.get_text().strip()
        
        # Se a página estiver vazia ou for imagem (muito pouco texto), usa OCR
        if len(texto_pag) < 100:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_path = "temp.png"
            pix.save(img_path)
            texto_pag = pytesseract.image_to_string(img_path, lang="por")
            if os.path.exists(img_path): os.remove(img_path)
            
        texto_completo += "\n" + texto_pag
        
    doc.close()
    return limpar_texto(texto_completo)

# -------------------------------
# CORE: INDEXAÇÃO NO CHROMA
# -------------------------------
def main():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"Limpando banco anterior em {CHROMA_DIR}...")

    # Modelo excelente para busca semântica em Português
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    documentos_para_indexar = []

    if not os.path.exists(PASTA_REGIMENTOS):
        print(f"Erro: Pasta {PASTA_REGIMENTOS} não encontrada."); return

    for arquivo in os.listdir(PASTA_REGIMENTOS):
        if arquivo.lower().endswith(".pdf"):
            path_completo = os.path.join(PASTA_REGIMENTOS, arquivo)
            print(f"Lendo: {arquivo}...")
            
            # 1. Extrai ID do arquivo (ex: 001_wish.pdf -> 001)
            tenant_id = arquivo.split("_")[0]
            
            # 2. Extrai e limpa texto
            texto_bruto = extrair_texto_pdf(path_completo)
            
            # 3. Divide em blocos lógicos (Artigos ou Tópicos)
            blocos = dividir_documento(texto_bruto)
            
            # 4. Converte para objetos LangChain Document
            for indice, b in enumerate(blocos):

                chunk_id = f"{indice+1:02d}"

                documentos_para_indexar.append(
                    Document(
                        page_content=b["conteudo"],
                        metadata={
                            "tenant_id": tenant_id,
                            "tipo": b["tipo"],
                            "origem": arquivo,
                            "referencia": b["id"],
                            "chunk_id": f"chunk_{chunk_id}"
                        }
                    )
                )

    if documentos_para_indexar:
        Chroma.from_documents(
            documents=documentos_para_indexar,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        print(f"✅ Sucesso! {len(documentos_para_indexar)} blocos indexados.")
    else:
        print("⚠️ Nenhum documento PDF processado.")

if __name__ == "__main__":
    main()