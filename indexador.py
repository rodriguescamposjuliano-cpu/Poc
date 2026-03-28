import os
import shutil
import re
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from spellchecker import SpellChecker

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURAÇÕES ---
POPPLER_PATH = "/opt/homebrew/bin"
CHROMA_DIR = "chroma_db"
PASTA_REGIMENTOS = "./regimentos_input"

spell = SpellChecker(language='pt')

def extrair_titulo_real(pdf_path, doc_fitz):
    """
    Identifica o título de forma genérica baseada na estrutura do topo da página,
    parando assim que encontra um bloco denso de texto ou múltiplas quebras.
    """
    texto_capa = ""
    try:
        # 1. Tenta digital, se falhar vai para OCR
        texto_capa = doc_fitz[0].get_text().strip()
        if not texto_capa or len(texto_capa) < 20:
            paginas_imagem = convert_from_path(pdf_path, first_page=1, last_page=1, poppler_path=POPPLER_PATH)
            if paginas_imagem:
                texto_capa = pytesseract.image_to_string(paginas_imagem[0], lang="por")

        # 2. LIMPEZA INICIAL: Remove sujeira de borda de OCR
        linhas = [l.strip() for l in texto_capa.split('\n') if len(l.strip()) > 2]
        
        titulo_acumulado = []
        
        # 3. LÓGICA DE CAPTURA POR BLOCO:
        # Vamos pegar as primeiras linhas até encontrarmos algo que pareça "texto real" 
        # (linhas muito longas ou palavras que indicam início de artigos)
        for linha in linhas[:10]: # Analisa apenas o topo (primeiras 10 linhas úteis)
            # Se a linha contiver números de Artigo ou for muito longa (> 60 caracteres),
            # provavelmente entramos no corpo do texto. Paramos aqui.
            if re.search(r'(?i)\b(Art|Artigo|Cap|Capítulo|Parágrafo|§)\b', linha) or len(linha) > 65:
                break
            
            # Se a linha for curta e estiver no topo, faz parte do título
            titulo_acumulado.append(linha)
            
            # Se já pegamos o nome do condomínio (geralmente nas primeiras 3-4 linhas), 
            # não precisamos de 10 linhas.
            if len(titulo_acumulado) >= 5: 
                break

        # 4. SANITIZAÇÃO FINAL
        resultado = " ".join(titulo_acumulado)
        # Remove qualquer caractere que não seja letra, número ou espaço
        resultado = re.sub(r'[^\w\sÀ-ÿ]', '', resultado)
        # Remove espaços duplos
        resultado = re.sub(r'\s+', ' ', resultado).strip().upper()

        return resultado if resultado else "REGIMENTO INTERNO - DESCONHECIDO"

    except Exception as e:
        print(f"Erro na extração genérica: {e}")
    return "REGIMENTO INTERNO - ERRO NA LEITURA"



def limpar_e_corrigir_texto(texto):
    if not texto: return ""
    
    # 1. Correção de Erros Comuns de OCR
    mapa_correcao_ocr = {
        r"(?i)intemo": "Interno", r"(?i)intimo": "Interno",
        r"(?i)iniração": "infração", r"(?i)muita": "multa",
        r"(?i)peia": "pela", r"(?<=[\s\n])[8S]\s+(?=\d+º)": "§ ",
        r"(?i)regimento": "Regimento"
    }
    for erro, correcao in mapa_correcao_ocr.items():
        texto = re.sub(erro, correcao, texto)
    
    # 2. Normalização
    texto = re.sub(r'(\w)-\n(\w)', r'\1\2', texto) 
    texto = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', texto)

    # 3. Corretor Ortográfico Dinâmico (Variável palavras_corrigidas restaurada)
    palavras = texto.split()
    palavras_corrigidas = [] # <--- Variável aqui
    excecoes = {"art", "artigo", "caput", "condomínio", "§", "id"}

    for palavra in palavras:
        p_limpa = re.sub(r'[^\w]', '', palavra.lower())
        
        if len(p_limpa) > 3 and p_limpa not in excecoes:
            if p_limpa not in spell:
                corr = spell.correction(p_limpa)
                # Se houver correção, substitui mantendo a estrutura da palavra original
                palavras_corrigidas.append(palavra.lower().replace(p_limpa, corr) if corr else palavra)
            else:
                palavras_corrigidas.append(palavra)
        else:
            palavras_corrigidas.append(palavra)

    # Reconstroi o texto e ajusta quebras de linha para o Splitter
    texto_final = " ".join(palavras_corrigidas)
    texto_final = texto_final.replace("artigo", "\nArtigo").replace("art ", "\nArt ")
    return re.sub(r'\s+', ' ', texto_final).strip()

def processar_arquivo(pdf_path, tenant_id):
    print(f"--- Processando: {os.path.basename(pdf_path)} ---")
    doc_digital = fitz.open(pdf_path)
    
    titulo_real = extrair_titulo_real(pdf_path, doc_digital)
    
    print(f"Título Identificado: {titulo_real}")

    # Metadados exigidos pelo Item 4.1.1 do Projeto Final
    metadados_base = {
        "doc_id": f"doc_{tenant_id}",
        "titulo": titulo_real,
        "fonte": f"Arquivo: {os.path.basename(pdf_path)}",
        "tipo": "regulamento_condominial",
        "tenant_id": tenant_id
    }

    full_text_bruto = ""
    docs_temporarios = []

    for i, page in enumerate(doc_digital):
        text = page.get_text().strip()
        if text:
            text_limpo = limpar_e_corrigir_texto(text)
            docs_temporarios.append(Document(
                page_content=text_limpo, 
                metadata={**metadados_base, "page": i + 1, "type": "digital"}
            ))
            full_text_bruto += text_limpo

    if len(full_text_bruto) < 200:
        print("📸 Iniciando OCR para PDF de imagem...")
        pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)
        docs_temporarios = []
        for i, page in enumerate(pages):
            raw_ocr = pytesseract.image_to_string(page, lang="por")
            text_limpo = limpar_e_corrigir_texto(raw_ocr)
            docs_temporarios.append(Document(
                page_content=text_limpo, 
                metadata={**metadados_base, "page": i + 1, "type": "ocr"}
            ))
    
    doc_digital.close()
    return docs_temporarios

if __name__ == "__main__":
    if os.path.exists(CHROMA_DIR):
        print("Limpando banco antigo...")
        shutil.rmtree(CHROMA_DIR)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    todos_docs = []
    for arquivo in os.listdir(PASTA_REGIMENTOS):
        if arquivo.lower().endswith(".pdf"):
            tid = arquivo.split("_")[0]
            todos_docs.extend(processar_arquivo(os.path.join(PASTA_REGIMENTOS, arquivo), tid))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, 
        separators=["\nArt ", "\nArtigo ", "\n§", "\n\n"]
    )
    splits = splitter.split_documents(todos_docs)

    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DIR)
    print(f"\n✅ Indexação finalizada: {len(splits)} chunks criados com metadados.")