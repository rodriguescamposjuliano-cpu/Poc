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
from rich.text import Text
import sys

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

    import sys

def escrever_com_efeito(texto, delay=0.01):
    for char in texto:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

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
        # Pegamos os 3 melhores após o rerank
        for item in ranked[:3]:
            idx = item.get("id")
            doc = docs_unicos[idx]
            final.append(doc)
        
        return final

    def responder(self, pergunta, tenant_id):
        docs = self.buscar(pergunta, tenant_id)

        # 🔒 Regra de recusa (mais segura)
        if not docs or len(docs) == 0:
            return "Informação não localizada no regimento enviado."

        # Montagem do contexto com IDs REAIS
        contexto_lista = []
        for d in docs:
            doc_id = d.metadata.get("doc_id", f"doc_{tenant_id}")
            chunk_id = d.metadata.get("chunk_id", "unknown")

            ref_id = f"[{doc_id}#_{chunk_id}]"

            for i, d in enumerate(docs):
                doc_id = d.metadata.get("doc_id", f"doc_{tenant_id}")
                chunk_id = d.metadata.get("chunk_id", "unknown")

                ref_id = f"[{doc_id}#{chunk_id}]"

                texto = d.page_content

                # 👇 tenta puxar contexto anterior se parecer item de lista
                if texto.strip().startswith(("a)", "b)", "c)", "d)")) and i > 0:
                    texto = docs[i-1].page_content + "\n" + texto

                contexto_lista.append(f"Fonte {ref_id}:\n{texto}")

        contexto_string = "\n\n".join(contexto_lista)

        # Prompt forte (anti-alucinação + força citação)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
        Você é um assistente virtual especializado em regimento interno de condomínio.

        Responda APENAS com base no CONTEXTO.

        CONTEXTO:
        {context}

        PERGUNTA:
        {question}

        REGRAS:
        - Seja cordial e educado.
        - NÃO invente informações.
        - Use apenas o contexto fornecido.
        - Cite fontes obrigatoriamente.
        - Responda de forma lógica e coerente.

        REGRAS ESPECIAIS:
        - Se a pergunta for "Posso...", "É permitido...", etc:
        → Responda começando com "Sim," ou "Não," corretamente
        → Nunca contradiga a resposta

        - SEMPRE inclua um TRECHO LITERAL do regimento:
        → Copie exatamente como está no CONTEXTO
        → Preserve quebras de linha
        → NÃO reescreva
        → NÃO resuma o trecho
        → Inclua o artigo completo (ex: "Art. 5º") quando existir
        → SEMPRE inclua o parágrafo único ou parágrafos numerados (§) se eles existirem logo após o artigo citado.
        → NÃO traga apenas itens isolados como "a)" ou "b)"
        → Se houver lista, inclua o cabeçalho junto
       

        FORMATO OBRIGATÓRIO (JSON):
        {{
        "resposta": "explicação clara e direta",
        "trecho": "trecho literal copiado exatamente do contexto com quebras de linha",
        "citacoes": ["doc_id#chunk_id"]
        }}

        Se não encontrar resposta:
        {{
        "resposta": "Informação não localizada no regimento enviado.",
        "trecho": "",
        "citacoes": []
        }}
        """)
        ])
                
        # Log para auditoria (ótimo pro professor)
        log_debug("DEBUG_PROMPT_ENVIADO", {
            "pergunta": pergunta,
            "contexto": contexto_string
        })

        chain = prompt_template | self.llm | StrOutputParser()

        resposta = chain.invoke({
            "context": contexto_string,
            "question": pergunta
        }).strip()

        try:
            data = json.loads(resposta)

            resposta = data.get("resposta", "")
            trecho = data.get("trecho", "")
            citacoes = data.get("citacoes", [])

            citacoes_formatadas = " ".join([f"[{c}]" for c in citacoes])

            resposta_final = f"""{resposta}
                📄 Trecho do regimento:
                {trecho}

                📎 Fontes:
                {citacoes_formatadas}
                """.strip()

            return resposta_final

        except:
            return "Informação não localizada no regimento enviado."

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
                resposta_final = ia.responder(pergunta, tenant)
                tempo = time.time() - inicio

            # --- APRESENTAÇÃO ORGANIZADA ---
            console.print("\n" + "━" * 50, style="yellow")
            console.print(f"[bold yellow]Severino[/bold yellow] [dim](em {tempo:.2f}s)[/dim]")
            
            # Separação das partes para estilização individual
            partes = resposta_final.split("📄 Trecho do regimento:")
            resposta_principal = partes[0].strip()

            # 1. Resposta da IA (Texto principal)
            escrever_com_efeito(f"\n{resposta_principal}", 0.01)

            if len(partes) > 1:
                trecho_e_fontes = partes[1].split("📎 Fontes:")
                
                # 2. Caixa para o Trecho do Regimento (Grounding)
                trecho_texto = trecho_e_fontes[0].strip()
                console.print(
                    Panel(
                        trecho_texto, 
                        title="📄 Trecho do Regimento", 
                        border_style="green", 
                        padding=(1, 2)
                    )
                )

                # 3. Exibição das Fontes (Requisito doc_XXX#chunk_YYY)
                if len(trecho_e_fontes) > 1:
                    fontes_raw = trecho_e_fontes[1].strip()
                    
                    # Criamos um objeto de texto para evitar conflito de markup
                    linha_fontes = Text()
                    linha_fontes.append("\n📎 Fontes: ", style="bold magenta")
                    
                    # Adicionamos a fonte literal. O style "italic" será aplicado 
                    # sem tentar processar os colchetes dentro de 'fontes_raw'
                    linha_fontes.append(fontes_raw, style="italic cyan") 
                    
                    console.print(linha_fontes)

