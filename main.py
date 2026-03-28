import os
import time
from datetime import datetime
from dotenv import load_dotenv
import json
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.panel import Panel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest

# Recursos NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

load_dotenv()
console = Console()
DEBUG = True
LOG_FILE = "severino_debug.log"

def limpar_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def log_debug(titulo, data):
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

class SeverinoIA:
    def __init__(self):
        console.print("[yellow]Iniciando motores do Severino...[/yellow]")
        # Mapeamento estático apenas para o menu, os nomes reais vêm do banco agora
        self.tenants_map = {
            "001": "WISH COIMBRA", "002": "REALITY BURITIS", 
            "003": "American Tower", "004": "Edifício Florença", "005": "Monte Fuji"
        }
        self.stemmer = SnowballStemmer("portuguese")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.vectorstore = Chroma(persist_directory="chroma_db", embedding_function=self.embeddings)
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    def _preprocess_local(self, text):
        words = word_tokenize(text.lower())
        return [self.stemmer.stem(w) for w in words if w.isalnum()]

    def buscar(self, query, tenant_id):
        results = self.vectorstore.get(where={"tenant_id": tenant_id})
        docs_base = [Document(page_content=c, metadata=m) for c, m in zip(results['documents'], results['metadatas'])]
        
        if not docs_base: 
            return []

        # 1. Busca Híbrida (BM25 + Vetorial)
        bm25 = BM25Retriever.from_documents(docs_base, preprocess_func=self._preprocess_local)
        bm25.k = 30
        docs_bm25 = bm25.invoke(query)

        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 30, "filter": {"tenant_id": tenant_id}})
        docs_vetoriais = vector_retriever.invoke(query)

        # 2. Deduplicação
        todos = docs_bm25 + docs_vetoriais
        docs_unicos = []
        vistos = set()
        for d in todos:
            if d.page_content not in vistos:
                docs_unicos.append(d)
                vistos.add(d.page_content)

        # 3. Reranking (Trilha B do edital)
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs_unicos)]
        ranked = self.reranker.rerank(RerankRequest(query=query, passages=passages))

        final = []
        for item in ranked[:5]: # Top 5 conforme boas práticas de RAG
            idx = item.get("id")
            doc = docs_unicos[idx]
            final.append(doc)
        
        return final

    def responder(self, pergunta, tenant_id):
        docs = self.buscar(pergunta, tenant_id)
        if not docs: 
            return "Informação não localizada no regimento enviado."
        
        # 1. Contexto formatado como "Base de Dados"
        contexto_lista = []
        for i, d in enumerate(docs):
            doc_id = d.metadata.get('doc_id', f'doc_{tenant_id}')
            ref_id = f"[{doc_id}#chunk_{i}]"
            contexto_lista.append(f"REGRA_TECNICA {ref_id}:\n{d.page_content}")
        
        contexto_string = "\n\n".join(contexto_lista)
        
        # 2. Prompt de "Engenharia de Precisão"
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Você é um extrator de dados jurídico chamado Severino. 
            Sua saída deve ser estritamente no formato: [Resposta Curta] [Código da Fonte].

            REGRAS DE FORMATAÇÃO:
            1. Proibido iniciar com "De acordo com" ou "O regimento diz".
            2. Cada afirmação deve ser seguida pela sua fonte no formato [chunk_X].
            3. Use APENAS a REGRA_TECNICA fornecida.
            
            EXEMPLO DE RESPOSTA:
            'Não é permitido barulho após as 22h [doc_001#chunk_4].'"""),
            ("human", """CONTEXTO:
            {context}

            PERGUNTA:
            {question}

            RESPOSTA (Obrigatório conter o código entre colchetes ao final):""")
        ])

        log_debug("DEBUG_PROMPT_ENVIADO", {"corpo": contexto_string})

        chain = prompt_template | self.llm | StrOutputParser()
        
        return chain.invoke({
            "context": contexto_string, 
            "question": pergunta, 
            "condominio": self.tenants_map.get(tenant_id, "Condomínio")
        })

if __name__ == "__main__":
    ia = SeverinoIA()
    
    while True: # Loop do Menu Principal
        limpar_console()
        console.print(Panel.fit("SEVERINO IA - SELECIONE UM CONDOMÍNIO", style="bold blue"))
        for id, nome in ia.tenants_map.items():
            console.print(f"[bold cyan]{id}[/bold cyan] - {nome}")
        console.print("\n[dim]Digite 'sair' para encerrar o programa.[/dim]")

        tenant = input("\nID do Condomínio: ").strip()
        
        if tenant.lower() == 'sair':
            break
        if tenant not in ia.tenants_map:
            console.print("[red]ID inválido![/red]")
            time.sleep(1)
            continue

        # Loop da Conversa
        limpar_console()
        nome_condo = ia.tenants_map[tenant]
        console.print(Panel(f"CONECTADO: {nome_condo}", style="bold green"))
        console.print("[dim]Comandos: 'voltar' para o menu ou 'sair' para encerrar.[/dim]\n")

        while True:
            pergunta = input(f"Morador ({nome_condo}): ").strip()
            
            if pergunta.lower() == 'voltar':
                break
            if pergunta.lower() == 'sair':
                exit()
            if not pergunta:
                continue

            with console.status("[bold yellow]Consultando..."):
                inicio = time.time()
                resposta = ia.responder(pergunta, tenant)
                tempo = time.time() - inicio

            console.print(f"\n[bold yellow]Severino:[/bold yellow]\n{resposta}")
            console.print(f"[dim]Tempo: {tempo:.2f}s | Log gerado em {LOG_FILE}[/dim]\n")