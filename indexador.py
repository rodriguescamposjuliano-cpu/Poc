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
    if not texto: return ""
    
    # Filtro de rodapé específico para o Wish Coimbra e similares
    padrao_rodape = r'(?i)Regimento Interno\s*[-–—]\s*Residencial.*|^\d+$'
    
    linhas = texto.split('\n')
    linhas_validas = []
    
    for linha in linhas:
        l = linha.strip()
        
        # Ignora rodapés e números de página isolados
        if re.fullmatch(padrao_rodape, l):
            continue
            
        # Mantém a linha se for um marcador de tópico ou sub-tópico
        if re.match(r'^\d+(\.\d+)*\.?$', l):
            linhas_validas.append(l)
            continue

        if not l or not linha_e_valida(l):
            continue
            
        # Filtra Sumário e outros metadados
        if re.fullmatch(r'(?i)\s*(Sumário|Pág\w*\.?\s*\d+)\s*', l):            
            continue
        
        linhas_validas.append(l)
    
    # COSTURA DE PARÁGRAFOS: Une linhas que foram quebradas injustamente
    texto_final = ""
    for i in range(len(linhas_validas)):
        l_atual = linhas_validas[i]
        if i < len(linhas_validas) - 1:
            l_prox = linhas_validas[i+1]
            # Se a linha atual não termina com pontuação e a próxima não é um novo tópico
            if not l_atual.endswith(('.', ':', '!', '?')) and \
               not re.match(r'^(\d+\.|Art|§)', l_prox):
                texto_final += l_atual + " "
            else:
                texto_final += l_atual + "\n"
        else:
            texto_final += l_atual

    return texto_final.strip()

# -------------------------------
# 4. DIVISÃO E EXTRAÇÃO
# -------------------------------
def dividir_documento(texto):
    blocos = []
    # Limite de caracteres para decidir se um "assunto" deve ser fatiado
    LIMITE_CHUNK = 2500 
    OVERLAP = 300 # Sobreposição para não cortar frases no meio

    for erro, correcao in mapa_correcao_ocr.items():
        texto = re.sub(erro, correcao, texto)

    # 1. Tenta padrão de ARTIGOS primeiro
    padrao_artigos = r'(?i)(?:^|\n)\s*((?:Art(?:igo)?\.?)\s+\d+[\s.\-–—]*[º°ª]?)'
    matches_art = list(re.finditer(padrao_artigos, texto))
    
    if len(matches_art) > 5:
        for i in range(len(matches_art)):
            inicio = matches_art[i].start()
            fim = matches_art[i+1].start() if i+1 < len(matches_art) else len(texto)
            trecho = texto[inicio:fim].strip()
            id_ref = matches_art[i].group(1).strip()
            id_ref = re.sub(r'[\s.\-–—]+$', '', id_ref)
            
            # Se um artigo for absurdamente grande, podemos fatiar aqui também, 
            # mas geralmente artigos são curtos.
            if len(trecho) > 20:
                blocos.append({"tipo": "artigo", "id": id_ref, "conteudo": trecho})
    
    # 2. PADRÃO ASSUNTO (WISH COIMBRA) com Fatiamento Inteligente
    else:
        padrao_topicos = r'(?m)^\s*(\d+\.\s+[A-ZÀ-Ú\s]{4,})$'
        matches_top = list(re.finditer(padrao_topicos, texto))
        
        if len(matches_top) > 0:
            for i in range(len(matches_top)):
                inicio = matches_top[i].start()
                fim = matches_top[i+1].start() if i+1 < len(matches_top) else len(texto)
                
                assunto_titulo = matches_top[i].group(1).strip()
                conteudo_completo = texto[inicio:fim].strip()
                
                # ESTRATÉGIA: Se o assunto for maior que o limite, fatiamos
                if len(conteudo_completo) > LIMITE_CHUNK:
                    # Quebra o conteúdo em pedaços mantendo o título no início de cada um
                    corpo_texto = conteudo_completo[len(assunto_titulo):].strip()
                    
                    for j in range(0, len(corpo_texto), LIMITE_CHUNK - OVERLAP):
                        fim_fatia = j + LIMITE_CHUNK
                        fatia = corpo_texto[j:fim_fatia]
                        
                        # Injeção de cabeçalho: O segredo para a IA não se perder
                        conteudo_final = f"[{assunto_titulo} - CONTINUAÇÃO]\n{fatia}"
                        if j == 0:
                            conteudo_final = f"{assunto_titulo}\n{fatia}"
                            
                        blocos.append({
                            "tipo": "assunto_fatiado",
                            "id": assunto_titulo,
                            "conteudo": conteudo_final
                        })
                else:
                    if len(conteudo_completo) > 20:
                        blocos.append({
                            "tipo": "assunto_completo",
                            "id": assunto_titulo,
                            "conteudo": conteudo_completo
                        })
        else:
            # Fallback para documentos sem estrutura clara
            for i in range(0, len(texto), 1500 - 200):
                blocos.append({"tipo": "chunk_geral", "id": "geral", "conteudo": texto[i:i+1500]})
            
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
                # O ID de referência agora é o que aparecerá na citação
                ref_final = b["id"]
                
                documentos_para_indexar.append(
                    Document(
                        page_content=b["conteudo"],
                        metadata={
                            "tenant_id": tenant_id,
                            "tipo": b["tipo"],
                            "referencia": ref_final, # Ex: "3. DEVERES DOS CONDÔMINOS"
                            "doc_id": f"doc_{tenant_id}",
                            "chunk_id": f"{indice+1:02d}"
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