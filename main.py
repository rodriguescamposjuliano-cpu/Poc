import os
import time
import csv
from datetime import datetime
from dotenv import load_dotenv

# LangChain e Integrações
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- IMPORT CORRIGIDO ---
from localclassifier import LocalClassifier 

# Carrega variáveis de ambiente (API Keys, etc)
load_dotenv()

class SeverinoIA:
    def __init__(self):
        # Configurações de Identidade
        self.config = {
            "nome_ia": "Severino",
            "nome_condominio": "Monte Fuji",
            "db_dir": "chroma_db",
            "model_llm": "llama-3.1-8b-instant"
        }
        
        # 1. Inicializa o Classificador Local (ML)
        self.classifier = LocalClassifier()
        
        # 2. Inicializa Embeddings (Mesmo modelo do indexador.py)
        print("🧠 Carregando inteligência vetorial...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 3. Conecta ao Banco de Dados criado pelo indexador.py
        if not os.path.exists(self.config["db_dir"]):
            print("❌ ERRO: Banco 'chroma_db' não encontrado. Rode o indexador.py primeiro!")
            exit()
            
        self.vectorstore = Chroma(
            persist_directory=self.config["db_dir"], 
            embedding_function=self.embeddings
        )
        
        # 4. Configura o LLM (Groq)
        self.llm = ChatGroq(
            model=self.config["model_llm"],
            temperature=0, # Respostas precisas, sem "criatividade" excessiva
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def recuperar_contexto(self, pergunta: str) -> str:
        """Busca no regimento os trechos mais relevantes."""
        # Aumentamos para K=10 para garantir cobertura de artigos longos
        docs = self.vectorstore.similarity_search(pergunta, k=10)
        contexto = "\n\n".join([
            f"[Página {d.metadata.get('page', 'S/N')}]: {d.page_content}" 
            for d in docs
        ])
        return contexto

    def registrar_log_sindico(self, pergunta: str):
        """Classifica a intenção e salva para o dashboard.py."""
        categoria = self.classifier.classificar(pergunta)
        log_file = "dashboard_sindico.csv"
        
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Data", "Pergunta", "Categoria"])
            writer.writerow([datetime.now().strftime("%d/%m/%Y %H:%M"), pergunta, categoria])
        
        print(f"📡 [LOG] Assunto: {categoria}")
        return pergunta

    def criar_chain(self):
        """Monta o pipeline de execução."""
        template = """
        Você é o {nome_ia}, o assistente virtual do Condomínio {nome_condominio}.
        Sua missão é ajudar os moradores com base estrita no Regimento Interno fornecido.

        ### REGRAS DO REGIMENTO ENCONTRADAS:
        {context}

        ### PERGUNTA DO MORADOR:
        {question}

        ### ORIENTAÇÕES PARA SUA RESPOSTA:
        1. Seja cordial e educado.
        2. Se a resposta estiver no contexto, cite o número da página ou artigo.
        3. Se não encontrar a informação, diga: "Sinto muito, mas não encontrei uma regra específica sobre isso no regimento atual. Recomendo consultar a administração."
        4. Se o morador pedir para "transcrever" ou "copiar", forneça o texto literal.

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

# --- LOOP PRINCIPAL ---
if __name__ == "__main__":
    severino = SeverinoIA()
    chain = severino.criar_chain()
    
    print("\n" + "="*30)
    print(f"🏢 {severino.config['nome_ia']} pronto para o {severino.config['nome_condominio']}!")
    print("="*30)

    while True:
        entrada = input("\nMorador: ")
        if entrada.lower() in ["sair", "tchau", "encerrar"]:
            break
            
        try:
            inicio = time.time()
            resposta = chain.invoke(entrada)
            tempo = time.time() - inicio
            
            print(f"\n{severino.config['nome_ia']}: {resposta}")
            print(f"\n[⏱️ {tempo:.2f}s]")
        except Exception as e:
            print(f"⚠️ Erro ao processar: {e}")