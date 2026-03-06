import os
import logging

# --- 1. SUPRESSÃO DE LOGS TÉCNICOS (Deve vir antes de qualquer import de IA) ---
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import time
import csv
from datetime import datetime
from dotenv import load_dotenv

# --- 2. LANGCHAIN E INTEGRAÇÕES ATUALIZADAS ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Migrado para o pacote atualizado
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- 3. IMPORT DO CLASSIFICADOR LOCAL ---
from localclassifier import LocalClassifier 

# Carrega variáveis do .env
load_dotenv()

class SeverinoIA:
    def __init__(self):
        # Configurações de Identidade e Infraestrutura
        self.config = {
            "nome_ia": "Severino",
            "nome_condominio": "Monte Fuji",
            "db_dir": "chroma_db",
            "model_llm": "llama-3.1-8b-instant" # O modelo mais potente para RAG
        }
        
        # A. Inicializa o Classificador Local (Machine Learning)
        self.classifier = LocalClassifier()
        
        # B. Inicializa Embeddings (Silencioso)
        print("🧠 Carregando inteligência vetorial...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # C. Conecta ao Banco de Dados (Padrão Novo)
        if not os.path.exists(self.config["db_dir"]):
            print(f"❌ ERRO: Banco '{self.config['db_dir']}' não encontrado. Execute o indexador.py primeiro.")
            exit()
            
        self.vectorstore = Chroma(
            persist_directory=self.config["db_dir"], 
            embedding_function=self.embeddings
        )
        
        # D. Configura o LLM com Temperature Zero (Precisão Jurídica)
        self.llm = ChatGroq(
            model=self.config["model_llm"],
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def recuperar_contexto(self, pergunta: str) -> str:
        """Busca no regimento os trechos mais relevantes para o morador."""
        # K=10 para garantir que pegamos o contexto completo de artigos longos
        docs = self.vectorstore.similarity_search(pergunta, k=10)
        contexto = "\n\n".join([
            f"[Página {d.metadata.get('page', 'S/N')}]: {d.page_content}" 
            for d in docs
        ])
        return contexto

    def registrar_log_sindico(self, pergunta: str):
        """Classifica a intenção via ML local e salva no CSV para o Dashboard."""
        categoria = self.classifier.classificar(pergunta)
        log_file = "dashboard_sindico.csv"
        
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Data", "Pergunta", "Categoria"])
            writer.writerow([datetime.now().strftime("%d/%m/%Y %H:%M"), pergunta, categoria])
        
        print(f"📡 [LOG] Assunto identificado: {categoria}")
        return pergunta

    def criar_chain(self):
        """Pipeline de Processamento (RAG + Agentic Logging)."""
        template = """
        Você é o {nome_ia}, o assistente virtual do Condomínio {nome_condominio}.
        Sua missão é ajudar os moradores com base estrita no Regimento Interno fornecido.

        ### REGRAS DO REGIMENTO ENCONTRADAS:
        {context}

        ### PERGUNTA DO MORADOR:
        {question}

        ### ORIENTAÇÕES PARA SUA RESPOSTA:
        1. Seja cordial e cite o número da página ou artigo encontrado.
        2. Responda apenas o que estiver no texto. Se não souber, oriente a falar com a administração.
        3. Se o morador pedir "transcrição" ou "cópia", forneça o texto literal do regimento.

        RESPOSTA DO {nome_ia}:
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        return (
            {
                "context": RunnableLambda(self.recuperar_contexto),
                "question": RunnableLambda(self.registrar_log_sindico),
                "nome_ia": lambda x: self.config["nome_ia"],
                "nome_condominio": lambda x: self.config["nome_condominio"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

# --- EXECUÇÃO DO SISTEMA ---
if __name__ == "__main__":
    severino = SeverinoIA()
    chain = severino.criar_chain()
    
    print("\n" + "="*40)
    print(f"🏢 {severino.config['nome_ia']} ONLINE | CONDOMÍNIO {severino.config['nome_condominio']}")
    print("="*40)

    while True:
        entrada = input("\nMorador: ")
        if entrada.lower() in ["sair", "tchau", "encerrar"]:
            break
            
        try:
            inicio = time.time()
            resposta = chain.invoke(entrada)
            fim = time.time() - inicio
            
            print(f"\n{severino.config['nome_ia']}: {resposta}")
            print(f"\n[🕒 Tempo de processamento: {fim:.2f}s]")
        except Exception as e:
            print(f"⚠️ Erro ao processar solicitação: {e}")