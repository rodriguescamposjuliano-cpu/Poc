import os
import shutil
import re
import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Silenciar warnings técnicos
os.environ["HF_HUB_OFFLINE"] = "1"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# CONFIGURAÇÕES
CHROMA_DIR = "chroma_db"
PASTA_REGIMENTOS = "./regimentos_input"

# 1. Correção de Erros Comuns de OCR
mapa_correcao_ocr = {
    r"(?i)intemo": "Interno", 
    r"(?i)intimo": "Interno",
    r"(?i)iniração": "infração", 
    r"(?i)muita": "multa",
    r"(?i)peia": "pela", 
    r"(?i)\bArm\b\.?": "Art.",
    r"(?i)\bAr\b\.?": "Art.",
    r"(?i)\bAnt\b\.?": "Art.",
    r"(?i)\bAst\b\.?": "Art.",
    r"(?i)\bAt\b\.?": "Art."
    
     
}

# -------------------------------
# 1. TRATAMENTO DE IMAGEM (ANTI-ASSINATURA)
# -------------------------------
def tratar_imagem_para_ocr(pix):
    """Remove assinaturas leves e carimbos usando técnicas de OpenCV."""
    # Converte o pixmap para array numpy (formato OpenCV)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding: Transforma o que é claro (assinaturas azuis/carimbos) em branco 
    # e o que é escuro (texto impresso) em preto puro.
    _, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

# -------------------------------
# 2. FILTRO DE SANIDADE (ANTI-LIXO OCR)
# -------------------------------
def linha_e_valida(linha):
    """Verifica se a linha extraída faz sentido gramatical ou é ruído visual."""
    texto = linha.strip()
    if len(texto) < 4: return False
    
    # Se a linha tiver muitos símbolos e poucos caracteres alfanuméricos, é ruído
    letras = re.findall(r'[a-zA-ZÀ-Úà-ú]', texto)
    if len(texto) > 0:
        proporcao = len(letras) / len(texto)
        if proporcao < 0.45: # Menos de 45% de letras? Provavelmente um rabisco.
            return False

    # Filtro de palavras sem vogais (OCR de assinaturas gera coisas como 'SbrOLTT')
    palavras = texto.split()
    for p in palavras:
        if len(p) > 5 and not re.search(r'[aeiouAEIOUÀ-Úà-ú]', p):
            return False
            
    return True

# -------------------------------
# 3. LIMPEZA E NORMALIZAÇÃO
# -------------------------------
def limpar_texto(texto):
    if not texto: 
        return ""
    
    linhas = texto.split('\n')
    linhas_validas = []
    
    for linha in linhas:
        l = linha.strip()
        
        # --- PROTEÇÃO PARA ARTIGOS ---
        # Se a linha começa com "Art", ela é SAGRADA. Não aplicamos filtros de data/ruído nela.
        if re.match(r'(?i)^Art\b', l):
            linhas_validas.append(l)
            continue

        # Filtro de Sanidade padrão
        if not l or not linha_e_valida(l):
            continue

        # Filtro de Cabeçalho/Rodapé
        if re.fullmatch(r'(?i)\s*(Regimento Interno|Pág\w*\.?\s*\d+|Página\s*\d+)\s*', l):            
            continue
        
        # Filtro de Protocolo: Refinado para não confundir horários (HH:MM) com datas (DD/MM)
        # Verificamos se tem a barra '/' especificamente.
        if "/" in l and re.search(r'\d{2}/\d{2}/\d{2,4}', l) and len(l) < 35:
            continue

        linhas_validas.append(l)
    
    texto_final = '\n'.join(linhas_validas)
    
    # --- CORREÇÕES GLOBAIS ---

    # A) Parágrafo Único
    texto_final = re.sub(r'(?i)\b(8|S|B)\s+Único\b', 'Parágrafo Único', texto_final)
    
    # B) Símbolos de parágrafo
    texto_final = re.sub(r'(?m)^[8\$S]\s*(\d+)', r'§ \1', texto_final)

    # C) União de palavras (Letra + Hífen + Quebra + Letra)
    texto_final = re.sub(r'([a-zA-ZÀ-Úà-ú])-\s*\n([a-zA-ZÀ-Úà-ú])', r'\1\2', texto_final)
    
    # D) Normalização de Travessões e Hifens de Artigos
    # Agora incluímos o travessão longo '—' que aparece no seu Art. 28
    texto_final = re.sub(r'(\d+)\s*[\-–—]\s*', r'\1 - ', texto_final)
    
    # E) Espaços duplos (preservando quebras de linha importantes)
    texto_final = re.sub(r'[ \t]+', ' ', texto_final)
    
    return texto_final.strip()

# -------------------------------
# 4. DIVISÃO E EXTRAÇÃO
# -------------------------------
def dividir_documento(texto):
    blocos = []
    
    # Este regex ignora se tem 1 espaço, 5 espaços, ponto ou hífen após o número
    padrao_artigos = r'(?i)(?:^|\n)\s*((?:Art(?:igo)?\.?)\s+\d+[\s.\-–—]*[º°ª]?)'

    for erro, correcao in mapa_correcao_ocr.items():
        texto = re.sub(erro, correcao, texto)

    matches = list(re.finditer(padrao_artigos, texto))
    
    if len(matches) > 0:
        for i in range(len(matches)):
            inicio = matches[i].start()
            fim = matches[i+1].start() if i+1 < len(matches) else len(texto)
            
            trecho = texto[inicio:fim].strip()
            # Captura o ID limpo (ex: "Art. 32")
            id_ref = matches[i].group(1).strip()
            # Limpa hifens ou pontos extras que sobraram no ID
            id_ref = re.sub(r'[\s.\-–—]+$', '', id_ref)
            
            if len(trecho) > 20:
                blocos.append({
                    "tipo": "artigo",
                    "id": id_ref,
                    "conteudo": trecho
                })
    
    # Se não achar nada, faz o corte padrão por tamanho
    if not blocos:
        tamanho = 1200
        for i in range(0, len(texto), tamanho - 200):
            blocos.append({"tipo": "chunk", "id": "bloco", "conteudo": texto[i:i+tamanho]})
            
    return blocos

def extrair_texto_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texto_completo = ""
    
    for page in doc:
        texto_pag = page.get_text().strip()
        
        # Se for imagem ou tiver pouco texto, aplica OCR com tratamento de imagem
        if len(texto_pag) < 150:
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) # Zoom de 3x
            img_limpa = tratar_imagem_para_ocr(pix)
            pil_img = Image.fromarray(img_limpa)
            texto_pag = pytesseract.image_to_string(pil_img, lang="por", config='--psm 3')
            
        texto_completo += "\n" + texto_pag
        
    doc.close()
    return limpar_texto(texto_completo)

# -------------------------------
# 5. MAIN
# -------------------------------
def main():
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("Limpando banco anterior...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    documentos_para_indexar = []

    if not os.path.exists(PASTA_REGIMENTOS):
        os.makedirs(PASTA_REGIMENTOS); return

    for arquivo in os.listdir(PASTA_REGIMENTOS):
        if arquivo.lower().endswith(".pdf"):
            print(f"Processando: {arquivo}...")
            path_completo = os.path.join(PASTA_REGIMENTOS, arquivo)
            tenant_id = arquivo.split("_")[0]
            
            texto_bruto = extrair_texto_pdf(path_completo)
            blocos = dividir_documento(texto_bruto)
            
            for indice, b in enumerate(blocos):
                documentos_para_indexar.append(
                    Document(
                        page_content=b["conteudo"],
                        metadata={
                            "tenant_id": tenant_id,
                            "tipo": b["tipo"],
                            "referencia": b["id"],
                            "doc_id": f"doc_{tenant_id}",
                            "chunk_id": f"chunk_{indice+1:02d}"
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

if __name__ == "__main__":
    main()