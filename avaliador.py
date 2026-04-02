import json
import os
import re
import math
import traceback
import pandas as pd
from datasets import Dataset

# =====================================================================
# SILENCIADOR DE LOGS E AVISOS (WARNINGS)
# =====================================================================
import warnings
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("ragas").setLevel(logging.WARNING)

# =====================================================================
# IMPORTS DO RAGAS E LANGCHAIN
# =====================================================================
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness
from langchain_core.output_parsers import StrOutputParser

try:
    from ragas.metrics import ResponseRelevancy
    RelevancyMetric = ResponseRelevancy
except ImportError:
    from ragas.metrics import AnswerRelevancy
    RelevancyMetric = AnswerRelevancy

class Avaliador:
    def __init__(self, ia, prompt_juiz):
        """Inicializa o avaliador recebendo a instância da IA e o prompt do juiz."""
        self.ia = ia
        self.prompt_juiz = prompt_juiz
        
        # Configuração Ragas
        self.ragas_llm = LangchainLLMWrapper(ia.llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(ia.embeddings)
        self.metric_faithfulness = Faithfulness(llm=self.ragas_llm)
        self.metric_relevancy = RelevancyMetric(llm=self.ragas_llm, embeddings=self.ragas_embeddings)

    def avaliar_chunks(self, chunks_esperados, chunks_gerados):
        set_esperados = set(chunks_esperados)
        set_gerados = set(chunks_gerados)
        return "SIM" if set_esperados == set_gerados else "NAO"

    def extrair_nota(self, valor):
        if isinstance(valor, list) or hasattr(valor, 'tolist'):
            try: valor = list(valor)[0]
            except IndexError: valor = 0.0
        try:
            nota = float(valor)
            return 0.0 if math.isnan(nota) else nota
        except (ValueError, TypeError):
            return 0.0

    def avaliar_resposta_customizada(self, pergunta, resposta_esperada, resposta_obtida):
        chain = self.prompt_juiz | self.ia.llm | StrOutputParser()
        try:
            resultado_str = chain.invoke({
                "pergunta": pergunta,
                "esperada": resposta_esperada,
                "obtida": resposta_obtida
            }).strip()
            
            resultado_limpo = resultado_str.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', resultado_limpo, re.DOTALL)
            if match:
                resultado_limpo = match.group(0)
                
            dados = json.loads(resultado_limpo)
            return float(dados.get("nota", 0.0)), dados.get("justificativa", "Sem justificativa válida.")
        except Exception as e:
            return 0.0, f"Erro no LLM Juiz: {str(e)}"

    def formatar_id_chunk_documento(self, doc, tenant_id):
        doc_id = doc.metadata.get("doc_id", f"doc_{tenant_id}")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        return f"{doc_id}#chunk_{str(chunk_id).replace('chunk_', '')}"

    def calcular_recall_at_k(self, docs_retornados, chunks_esperados, k, tenant_id):
        if not chunks_esperados:
            return None
            
        docs_k = docs_retornados[:k]
        ids_retornados = set([self.formatar_id_chunk_documento(d, tenant_id) for d in docs_k])
        ids_esperados = set(chunks_esperados)
        
        intersecao = ids_retornados.intersection(ids_esperados)
        return len(intersecao) / len(ids_esperados)

    def rodar_avaliacao(self, caminho_json):
        """Lê o JSON, processa todos os itens e exibe os resultados formatados no console."""
        try:
            with open(caminho_json, 'r', encoding='utf-8') as f:
                golden_set = json.load(f)
        except FileNotFoundError:
            print(f"Erro: Arquivo '{caminho_json}' não encontrado.")
            return

        FRASE_RECUSA = "Informação não localizada no regimento enviado."
        
        soma_ragas = 0.0
        soma_fidelidade = 0.0
        soma_relevancia = 0.0
        soma_juiz = 0.0
        total_perguntas_processadas = 0
        
        soma_recalls = {"sparse": {3: 0, 5: 0, 10: 0}, "dense": {3: 0, 5: 0, 10: 0}, "hybrid": {3: 0, 5: 0, 10: 0}}
        total_perguntas_com_chunks = 0

        print("\n" + "="*60)
        print("INICIANDO MOTOR DE AVALIAÇÃO (MODO APRESENTAÇÃO)")
        print("="*60)

        for item in golden_set:
            id_pergunta = item.get('id')
            pergunta = item.get('pergunta')
            resposta_esperada = item.get('resposta_referencia', '')
            chunks_esperados = item.get('chunks_esperados', [])
            
            print(f"\n>>> PROCESSANDO ID {id_pergunta} ...")
            
            arquivo_origem = item.get('arquivo_origem', '')
            tenant_id = arquivo_origem.split('_')[0] if arquivo_origem else "001"
            
            # 1. Executa a Resposta do RAG
            resposta_bruta = self.ia.responder(pergunta, tenant_id)
            
            # 2. Benchmark de Recalls Adicionais
            recalls = {"sparse": {}, "dense": {}, "hybrid": {}}
            if chunks_esperados:
                total_perguntas_com_chunks += 1
                modos_docs = self.ia.buscar_avaliar_modos(pergunta, tenant_id, k_max=10)
                
                for modo, docs in modos_docs.items():
                    for k in [3, 5, 10]:
                        recall_k = self.calcular_recall_at_k(docs, chunks_esperados, k, tenant_id)
                        recalls[modo][k] = recall_k
                        soma_recalls[modo][k] += recall_k
            
            # 3. Extrai Respostas e Contextos
            if "📄 Trecho do regimento:" in resposta_bruta:
                partes = resposta_bruta.split("📄 Trecho do regimento:")
                resposta_gerada = partes[0].strip()
                trecho_e_fontes = partes[1].split("📎 Fontes:")
                trecho_texto = trecho_e_fontes[0].strip()
            else:
                resposta_gerada = resposta_bruta.strip()
                trecho_texto = ""

            raw_chunks = re.findall(r'doc_\d+#(?:chunk_)?\d+', resposta_bruta)
            chunks_informados = list(set([f"{c.split('#')[0]}#chunk_{c.split('#')[1].replace('chunk_', '')}" for c in raw_chunks]))
            
            # 4. Avaliação
            if resposta_gerada == FRASE_RECUSA and resposta_esperada == FRASE_RECUSA:
                f_score, r_score, nota_ragas_10 = 1.0, 1.0, 10.0
                nota_custom, just_custom = 10.0, "Recusa correta e exata. Bypass aplicado."
            else:
                data_samples = {
                    'question': [pergunta],
                    'answer': [resposta_gerada],
                    'contexts': [[trecho_texto]] if trecho_texto else [[""]],
                    'ground_truth': [resposta_esperada]
                }
                dataset = Dataset.from_dict(data_samples)
                
                resultado_ragas = evaluate(
                    dataset, 
                    metrics=[self.metric_faithfulness, self.metric_relevancy],
                    show_progress=False
                )
                
                try: f_score = self.extrair_nota(resultado_ragas["faithfulness"])
                except: f_score = 0.0
                    
                try: r_score = self.extrair_nota(resultado_ragas["answer_relevancy"])
                except:
                    try: r_score = self.extrair_nota(resultado_ragas["response_relevancy"])
                    except: r_score = 0.0

                nota_ragas_10 = ((f_score + r_score) / 2) * 10

                nota_custom, just_custom = self.avaliar_resposta_customizada(
                    pergunta, resposta_esperada, resposta_gerada
                )
            
            # Atualiza Acumuladores
            total_perguntas_processadas += 1
            soma_ragas += nota_ragas_10
            soma_fidelidade += f_score
            soma_relevancia += r_score
            soma_juiz += nota_custom
            
            # --- EXIBIÇÃO NO CONSOLE PER-QUESTION (Formato Exato Solicitado) ---
            print(f"-> Pergunta: {pergunta}")
            print(f"-> Esperada: {resposta_esperada}")
            print(f"-> Gerada:   {resposta_gerada}")
            print(f"-> Chunks Esperados: {chunks_esperados} | Chunks Gerados: {chunks_informados}")
            print(f"-> Avaliação RAGAS:  [{nota_ragas_10:.1f}/10] (Fidelidade: {f_score:.2f} | Relevância: {r_score:.2f})")
            print(f"-> Nota LLM-Judge:   [{nota_custom:.1f}/10]")
            print(f"-> Justificativa:    {just_custom}")
            
            if chunks_esperados:
                print(f"-> Recalls: ESPARSO: Recall@3: {recalls['sparse'].get(3, 0.0):.2f} | Recall@5: {recalls['sparse'].get(5, 0.0):.2f} | Recall@10: {recalls['sparse'].get(10, 0.0):.2f}")
                print(f"           DENSO:   Recall@3: {recalls['dense'].get(3, 0.0):.2f} | Recall@5: {recalls['dense'].get(5, 0.0):.2f} | Recall@10: {recalls['dense'].get(10, 0.0):.2f}")
                print(f"           HÍBRIDO: Recall@3: {recalls['hybrid'].get(3, 0.0):.2f} | Recall@5: {recalls['hybrid'].get(5, 0.0):.2f} | Recall@10: {recalls['hybrid'].get(10, 0.0):.2f}")
            else:
                print("-> Recalls: N/A (Pergunta Fora do Corpus)")

        # =====================================================================
        # EXIBIÇÃO FINAL DAS MÉDIAS
        # =====================================================================
        if total_perguntas_processadas > 0:
            media_ragas = soma_ragas / total_perguntas_processadas
            media_fidelidade = soma_fidelidade / total_perguntas_processadas
            media_relevancia = soma_relevancia / total_perguntas_processadas
            media_juiz = soma_juiz / total_perguntas_processadas

            medias_recall = {"sparse": {}, "dense": {}, "hybrid": {}}
            if total_perguntas_com_chunks > 0:
                for modo in soma_recalls:
                    for k in [3, 5, 10]:
                        medias_recall[modo][k] = soma_recalls[modo][k] / total_perguntas_com_chunks

            print("\n" + "="*80)
            print("🏆 RESUMO FINAL: MÉDIAS DE TODAS AS PERGUNTAS PROCESSADAS")
            print("="*80)
            
            print(f"\n📊 1) AVALIAÇÃO RAGAS MÉDIA (Global):")
            print(f"   Nota Final (0-10): {media_ragas:.2f}")
            print(f"   Fidelidade (0-1):  {media_fidelidade:.2f}")
            print(f"   Relevância (0-1):  {media_relevancia:.2f}")

            print(f"\n⚖️  2) NOTA LLM-AS-JUDGE MÉDIA (Global):")
            print(f"   Nota Juiz (0-10):  {media_juiz:.2f}")

            print("\n🔍 3) AVALIAÇÃO DE RECALL MÉDIO (Somente perguntas do Corpus):")
            for modo, mr in medias_recall.items():
                if mr:
                    print(f"   [{modo.upper():<6}] Recall@3: {mr[3]:.2%} | Recall@5: {mr[5]:.2%} | Recall@10: {mr[10]:.2%}")
            print("="*80 + "\n")