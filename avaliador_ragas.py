import json
import time
import os
import re
import math
import traceback
import pandas as pd
from dotenv import load_dotenv
from prompts import obtenha_prompt_severino_juiz

# =====================================================================
# SILENCIADOR DE LOGS E AVISOS (WARNINGS)
# =====================================================================
import warnings
import logging

# Ignora os "DeprecationWarning" do Ragas
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Esconde os logs de INFO de requisições HTTP e do Google GenAI
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("ragas").setLevel(logging.WARNING)

# =====================================================================
# IMPORTS DO RAGAS E LANGCHAIN
# =====================================================================
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from ragas.metrics import Faithfulness

# Imports corrigidos do Langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Tratamento para compatibilidade de versão do Ragas
try:
    from ragas.metrics import ResponseRelevancy
    RelevancyMetric = ResponseRelevancy
except ImportError:
    from ragas.metrics import AnswerRelevancy
    RelevancyMetric = AnswerRelevancy

# =====================================================================
# IMPORT DA IA PRINCIPAL (CORRIGIDO PARA LER DO SERVERINO_IA)
# =====================================================================
try:
    from serverino_ia import SeverinoIA
except ImportError:
    print("Aviso: Módulo 'serverino_ia.SeverinoIA' não encontrado. Verifique se o arquivo serverino_ia.py está na pasta.")
    exit(1)

load_dotenv()

def avaliar_chunks(chunks_esperados, chunks_gerados):
    """Compara se os chunks recuperados são iguais aos esperados."""
    set_esperados = set(chunks_esperados)
    set_gerados = set(chunks_gerados)
    sao_iguais = "SIM" if set_esperados == set_gerados else "NAO"
    return sao_iguais, list(set_gerados)

def extrair_nota(valor):
    """Função auxiliar para garantir extração de floats nas notas do Ragas"""
    if isinstance(valor, list) or hasattr(valor, 'tolist'):
        try:
            valor = list(valor)[0]
        except IndexError:
            valor = 0.0
    try:
        nota = float(valor)
        return 0.0 if math.isnan(nota) else nota
    except (ValueError, TypeError):
        return 0.0

def avaliar_resposta_customizada(llm, pergunta, resposta_esperada, resposta_obtida):
    """
    Usa o modelo Langchain instanciado no SeverinoIA como juiz para 
    avaliar a resposta com base na 'Regra do 7' e justificar.
    """
    prompt_juiz = obtenha_prompt_severino_juiz()
    chain = prompt_juiz | llm | StrOutputParser()
    
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

# =====================================================================
# FUNÇÕES DE RECALL E MÉDIAS FINAIS
# =====================================================================
def formatar_id_chunk_documento(doc, tenant_id):
    """Extrai os metadados do Langchain Document e transforma no padrão do golden_set."""
    doc_id = doc.metadata.get("doc_id", f"doc_{tenant_id}")
    chunk_id = doc.metadata.get("chunk_id", "unknown")
    return f"{doc_id}#chunk_{str(chunk_id).replace('chunk_', '')}"

def calcular_recall_at_k(docs_retornados, chunks_esperados, k, tenant_id):
    """Calcula a métrica de Recall@K cortando a lista dinamicamente"""
    if not chunks_esperados:
        return None # Ignorar queries fora do corpus
        
    docs_k = docs_retornados[:k]
    ids_retornados = set([formatar_id_chunk_documento(d, tenant_id) for d in docs_k])
    ids_esperados = set(chunks_esperados)
    
    intersecao = ids_retornados.intersection(ids_esperados)
    return len(intersecao) / len(ids_esperados)

def exibir_resumo_final(medias_recall, media_ragas, media_juiz, media_fidelidade, media_relevancia):
    """Exibe o resumo de médias de todo o processamento e a análise de trade-offs."""
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
    for modo, recalls in medias_recall.items():
        print(f"   [{modo.upper():<6}] Recall@3: {recalls[3]:.2%} | Recall@5: {recalls[5]:.2%} | Recall@10: {recalls[10]:.2%}")
    print("="*80 + "\n")


def main():
    print("="*60)
    print("INICIANDO MOTOR DE AVALIAÇÃO E BENCHMARK DE RECALL")
    print("="*60)
    
    try:
        ia = SeverinoIA()
    except Exception as e:
        print(f"Erro ao inicializar SeverinoIA: {e}")
        return

    # Configuração Ragas
    print("\n[+] Configurando Wrappers do Ragas...")
    ragas_llm = LangchainLLMWrapper(ia.llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(ia.embeddings)
    metric_faithfulness = Faithfulness(llm=ragas_llm)
    metric_relevancy = RelevancyMetric(llm=ragas_llm, embeddings=ragas_embeddings)

    try:
        with open('golden_set.json', 'r', encoding='utf-8') as f:
            golden_set = json.load(f)
    except FileNotFoundError:
        print("Erro: Arquivo 'golden_set.json' não encontrado.")
        return

    resultados_finais = []
    arquivo_saida = 'resultado_avaliacao_completa.xlsx'
    FRASE_RECUSA = "Informação não localizada no regimento enviado."
    
    # Acumuladores Globais para o Final
    soma_ragas = 0.0
    soma_fidelidade = 0.0
    soma_relevancia = 0.0
    soma_juiz = 0.0
    total_perguntas_processadas = 0
    
    # Acumuladores de Recall
    soma_recalls = {"sparse": {3: 0, 5: 0, 10: 0}, "dense": {3: 0, 5: 0, 10: 0}, "hybrid": {3: 0, 5: 0, 10: 0}}
    total_perguntas_com_chunks = 0
    
    for item in golden_set:
        id_pergunta = item.get('id')
        pergunta = item.get('pergunta')
        resposta_esperada = item.get('resposta_referencia', '')
        chunks_esperados = item.get('chunks_esperados', [])

        print(f"\n>>> PROCESSANDO ID {id_pergunta} ...")
        
        try:
            arquivo_origem = item.get('arquivo_origem', '')
            tenant_id = arquivo_origem.split('_')[0] if arquivo_origem else "001"
            
            # 1. Executa a Resposta do RAG
            resposta_bruta = ia.responder(pergunta, tenant_id)
            
            # 2. Avaliação / Benchmark de Recalls Adicionais
            recalls = {"sparse": {}, "dense": {}, "hybrid": {}}
            if chunks_esperados:
                total_perguntas_com_chunks += 1
                modos_docs = ia.buscar_avaliar_modos(pergunta, tenant_id, k_max=10)
                
                for modo, docs in modos_docs.items():
                    for k in [3, 5, 10]:
                        recall_k = calcular_recall_at_k(docs, chunks_esperados, k, tenant_id)
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
            
            # 4. Avaliação de Recusa Exata (Bypass) ou RAGAS
            if resposta_gerada == FRASE_RECUSA and resposta_esperada == FRASE_RECUSA:
                f_score, r_score, nota_ragas_10 = 1.0, 1.0, 10.0
                nota_custom, just_custom = 10.0, "Recusa correta e exata. Bypass aplicado."
            else:
                # --- AVALIAÇÃO RAGAS ---
                data_samples = {
                    'question': [pergunta],
                    'answer': [resposta_gerada],
                    'contexts': [[trecho_texto]] if trecho_texto else [[""]],
                    'ground_truth': [resposta_esperada]
                }
                dataset = Dataset.from_dict(data_samples)
                
                resultado_ragas = evaluate(
                    dataset, 
                    metrics=[metric_faithfulness, metric_relevancy],
                    show_progress=False
                )
                
                try: f_score = extrair_nota(resultado_ragas["faithfulness"])
                except: f_score = 0.0
                    
                try: r_score = extrair_nota(resultado_ragas["answer_relevancy"])
                except:
                    try: r_score = extrair_nota(resultado_ragas["response_relevancy"])
                    except: r_score = 0.0

                nota_ragas_10 = ((f_score + r_score) / 2) * 10

                # --- AVALIAÇÃO LLM JUIZ CUSTOMIZADO ---
                nota_custom, just_custom = avaliar_resposta_customizada(
                    ia.llm, pergunta, resposta_esperada, resposta_gerada
                )
            
            # Atualiza Acumuladores Globais
            total_perguntas_processadas += 1
            soma_ragas += nota_ragas_10
            soma_fidelidade += f_score
            soma_relevancia += r_score
            soma_juiz += nota_custom

            # 5. Avaliação dos Chunks
            correspondencia, _ = avaliar_chunks(chunks_esperados, chunks_informados)
            
            # --- EXIBIÇÃO NO CONSOLE PER-QUESTION (Conforme Especificação) ---
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
                print("9) Recalls: N/A (Pergunta Fora do Corpus)")
            print("-" * 80)

            # --- ARMAZENAMENTO EXCEL ---
            linha_excel = {
                'ID': id_pergunta,
                'Categoria': item.get('categoria', 'N/A'),
                'Pergunta Feita': pergunta,
                'Resposta Esperada': resposta_esperada,
                'Resposta Gerada': resposta_gerada,
                'Chunks Esperados': ", ".join(chunks_esperados),
                'Chunks Gerados': ", ".join(chunks_informados),
                'Avaliação RAGAS (Nota 0-10)': nota_ragas_10,
                'RAGAS - Fidelidade (0-1)': f_score,
                'RAGAS - Relevância (0-1)': r_score,
                'Nota LLM-Judge (0-10)': nota_custom,
                'Justificativa LLM-Judge': just_custom
            }
            
            # Adiciona colunas de Recall na planilha
            for k in [3, 5, 10]:
                linha_excel[f'Recall@{k} Híbrido'] = recalls["hybrid"].get(k, None)
                linha_excel[f'Recall@{k} Sparse'] = recalls["sparse"].get(k, None)
                linha_excel[f'Recall@{k} Dense'] = recalls["dense"].get(k, None)
                
            resultados_finais.append(linha_excel)
            
            time.sleep(2)

        except Exception as e:
            print(f"[!] Erro Crítico no ID {id_pergunta}: {type(e).__name__} - {e}")
            traceback.print_exc()

    # =====================================================================
    # EXIBIÇÃO FINAL E SALVAMENTO DE ARQUIVO
    # =====================================================================
    if resultados_finais:
        # Calcular médias RAGAS e JUIZ
        media_ragas = soma_ragas / total_perguntas_processadas if total_perguntas_processadas > 0 else 0
        media_fidelidade = soma_fidelidade / total_perguntas_processadas if total_perguntas_processadas > 0 else 0
        media_relevancia = soma_relevancia / total_perguntas_processadas if total_perguntas_processadas > 0 else 0
        media_juiz = soma_juiz / total_perguntas_processadas if total_perguntas_processadas > 0 else 0

        # Calcular médias dos Recalls
        medias_recall = {"sparse": {}, "dense": {}, "hybrid": {}}
        if total_perguntas_com_chunks > 0:
            for modo in soma_recalls:
                for k in [3, 5, 10]:
                    medias_recall[modo][k] = soma_recalls[modo][k] / total_perguntas_com_chunks
        
        # Exibir o Resumo Final Solicitado
        exibir_resumo_final(medias_recall, media_ragas, media_juiz, media_fidelidade, media_relevancia)

        # Salvar o Excel
        df = pd.DataFrame(resultados_finais)
        df.to_excel(arquivo_saida, index=False)
        print(f"✅ AVALIAÇÃO FINALIZADA COM SUCESSO!")
        print(f"📊 Planilha gerada contendo todos os dados: {arquivo_saida}")
    else:
        print("\n⚠️ Nenhum dado foi processado para salvar.")

if __name__ == "__main__":
    main()