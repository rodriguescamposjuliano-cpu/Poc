import os
import sys
import re
import json
from nltk.stem import SnowballStemmer
from langchain_huggingface import HuggingFaceEmbeddings
from flashrank import Ranker, RerankRequest
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from nltk.tokenize import word_tokenize
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from serverino_logging import ServerinoLogging
import prompts

class SeverinoIA:
    def __init__(self):
        # Mapeamento estático apenas para o menu
        self.tenants_map = {
            "001": "WISH COIMBRA", 
            "002": "REALITY BURITIS", 
            "003": "American Tower", 
            "004": "Edifício Florença", 
            "005": "Monte Fuji"
        }
        
        self.stemmer = SnowballStemmer("portuguese")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.vectorstore = Chroma(persist_directory="chroma_db", embedding_function=self.embeddings)
        self.reranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir="./models_flashrank"
        )
        
        # Configuração do Gemini 
        chave_api = os.environ.get("GEMINI_API_KEY")
        if not chave_api:
            sys.exit(1)
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0, 
            api_key=chave_api
        )

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
        bm25.k = 5
        docs_bm25 = bm25.invoke(query)

        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5, "filter": {"tenant_id": tenant_id}})
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
        # Pegamos os 5 melhores após o rerank
        for item in ranked[:5]:
            idx = item.get("id")
            doc = docs_unicos[idx]
            final.append(doc)
        
        return final

    def buscar_avaliar_modos(self, query, tenant_id, k_max=10):
        """
        Método exclusivo para Benchmark.
        Retorna top K para os modos Sparse, Dense e Híbrido.
        """
        results = self.vectorstore.get(where={"tenant_id": tenant_id})
        docs_base = [Document(page_content=c, metadata=m) for c, m in zip(results['documents'], results['metadatas'])]
        
        if not docs_base: 
            return {"sparse": [], "dense": [], "hybrid": []}

        # --- MODO 1: SPARSE (BM25) ---
        bm25 = BM25Retriever.from_documents(docs_base, preprocess_func=self._preprocess_local)
        bm25.k = k_max
        docs_bm25 = bm25.invoke(query)

        # --- MODO 2: DENSE (VETORIAL) ---
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k_max, "filter": {"tenant_id": tenant_id}})
        docs_vetoriais = vector_retriever.invoke(query)

        # --- MODO 3: HYBRID (Fusão com Deduplicação + Rerank) ---
        # Buscamos um número maior (pool) para a fusão e deduplicação ser eficaz antes do rerank
        pool_size = max(15, k_max * 2)
        bm25_hyb = BM25Retriever.from_documents(docs_base, preprocess_func=self._preprocess_local)
        bm25_hyb.k = pool_size
        docs_bm25_hyb = bm25_hyb.invoke(query)
        
        vector_retriever_hyb = self.vectorstore.as_retriever(search_kwargs={"k": pool_size, "filter": {"tenant_id": tenant_id}})
        docs_vetoriais_hyb = vector_retriever_hyb.invoke(query)

        # Fusão e Deduplicação rigorosa baseada no conteúdo
        todos = docs_bm25_hyb + docs_vetoriais_hyb
        docs_unicos = []
        vistos = set()
        for d in todos:
            if d.page_content not in vistos:
                docs_unicos.append(d)
                vistos.add(d.page_content)

        # Reranking
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs_unicos)]
        ranked = self.reranker.rerank(RerankRequest(query=query, passages=passages))

        docs_hybrid = []
        for item in ranked[:k_max]:
            idx = item.get("id")
            docs_hybrid.append(docs_unicos[idx])

        return {
            "sparse": docs_bm25,
            "dense": docs_vetoriais,
            "hybrid": docs_hybrid
        }

    def responder(self, pergunta, tenant_id):
        docs = self.buscar(pergunta, tenant_id)

        if not docs or len(docs) == 0:
            return "Informação não localizada no regimento enviado."

        contexto_lista = []
        for i, d in enumerate(docs):
            doc_id = d.metadata.get("doc_id", f"doc_{tenant_id}")
            chunk_id = d.metadata.get("chunk_id", "unknown")

            ref_id = f"[{doc_id}#{chunk_id}]"
            texto = d.page_content

            if texto.strip().startswith(("a)", "b)", "c)", "d)")) and i > 0:
                texto = docs[i-1].page_content + "\n" + texto

            contexto_lista.append(f"Fonte {ref_id}:\n{texto}")

        contexto_string = "\n\n".join(contexto_lista)

        prompt_template = prompts.obtenha_prompt_severino()
                
        ServerinoLogging().log_debug("DEBUG_PROMPT_ENVIADO", {"pergunta": pergunta, "contexto": contexto_string})

        chain = prompt_template | self.llm | StrOutputParser()
        resposta = chain.invoke({"context": contexto_string, "question": pergunta}).strip()

        match = re.search(r'\{.*\}', resposta, re.DOTALL)
        if match:
            resposta_limpa = match.group(0)
        else:
            resposta_limpa = resposta.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(resposta_limpa)

            resposta_texto = data.get("resposta", "")
            trecho = data.get("trecho", "")
            citacoes = data.get("citacoes", [])

            citacoes_formatadas = " ".join([f"[{c}]" for c in citacoes])

            resposta_final = f"""{resposta_texto} 📄 Trecho do regimento: {trecho} 📎 Fontes: {citacoes_formatadas} """.strip()

            return resposta_final

        except Exception as e:
            ServerinoLogging().log_debug("ERRO_PARSE_JSON", {"erro": str(e), "resposta_bruta": resposta, "resposta_limpa": resposta_limpa})
            return "Informação não localizada no regimento enviado."