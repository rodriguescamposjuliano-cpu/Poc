import json
import time
import os
import re
import math
import traceback
import pandas as pd
from dotenv import load_dotenv

# =====================================================================
# IMPORTS DO RAGAS E LANGCHAIN
# =====================================================================
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from ragas.metrics import Faithfulness

# Tratamento para compatibilidade de versão do Ragas
try:
    from ragas.metrics import ResponseRelevancy
    RelevancyMetric = ResponseRelevancy
except ImportError:
    # Em versões mais recentes do Ragas, a métrica foi renomeada
    from ragas.metrics import AnswerRelevancy
    RelevancyMetric = AnswerRelevancy

# =====================================================================
# IMPORT DA IA PRINCIPAL
# =====================================================================
try:
    from main import SeverinoIA
except ImportError:
    print("Aviso: Módulo 'main.SeverinoIA' não encontrado. Verifique se o arquivo main.py está na pasta.")
    exit(1)

load_dotenv()

def avaliar_chunks(chunks_esperados, chunks_gerados):
    """Compara se os chunks recuperados são iguais aos esperados."""
    set_esperados = set(chunks_esperados)
    set_gerados = set(chunks_gerados)
    sao_iguais = "SIM" if set_esperados == set_gerados else "NAO"
    return sao_iguais, list(set_gerados)

def main():
    print("="*50)
    print("INICIANDO MOTOR DE AVALIAÇÃO SEVERINO IA (COM RAGAS)")
    print("="*50)
    
    try:
        ia = SeverinoIA()
    except Exception as e:
        print(f"Erro ao inicializar SeverinoIA: {e}")
        return

    # 1. Configurar o Juiz (LLM) e o Embedder para o Ragas
    print("\n[+] Configurando os Wrappers do Ragas com Langchain...")
    ragas_llm = LangchainLLMWrapper(ia.llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(ia.embeddings)

    # 2. Instanciar e configurar as métricas
    metric_faithfulness = Faithfulness(llm=ragas_llm)
    metric_relevancy = RelevancyMetric(llm=ragas_llm, embeddings=ragas_embeddings)

    try:
        with open('golden_set.json', 'r', encoding='utf-8') as f:
            golden_set = json.load(f)
    except FileNotFoundError:
        print("Erro: Arquivo 'golden_set.json' não encontrado.")
        return

    resultados_finais = []
    arquivo_saida = 'resultado_avaliacao_ragas.xlsx'
    
    for item in golden_set:
        id_pergunta = item.get('id')
        pergunta = item.get('pergunta')
        resposta_esperada = item.get('resposta_referencia', '')
        chunks_esperados = item.get('chunks_esperados', [])
        
        print(f"\n>>> [ID {id_pergunta}] PROCESSANDO PERGUNTA...")
        
        try:
            arquivo_origem = item.get('arquivo_origem', '')
            tenant_id = arquivo_origem.split('_')[0] if arquivo_origem else "001"
            
            # 1. Executa o sistema RAG
            resposta_bruta = ia.responder(pergunta, tenant_id)
            
            # 2. Extrai a resposta e o contexto gerado para enviar ao Ragas
            if "📄 Trecho do regimento:" in resposta_bruta:
                partes = resposta_bruta.split("📄 Trecho do regimento:")
                resposta_gerada = partes[0].strip()
                
                # Extraindo o trecho (contexts)
                trecho_e_fontes = partes[1].split("📎 Fontes:")
                trecho_texto = trecho_e_fontes[0].strip()
            else:
                resposta_gerada = resposta_bruta.strip()
                trecho_texto = ""

            # 3. Extrai e normaliza os chunks informados
            raw_chunks = re.findall(r'doc_\d+#(?:chunk_)?\d+', resposta_bruta)
            chunks_informados = list(set([f"{c.split('#')[0]}#chunk_{c.split('#')[1].replace('chunk_', '')}" for c in raw_chunks]))
            
            # 4. Construção do Dataset no formato exigido pelo Ragas
            data_samples = {
                'question': [pergunta],
                'answer': [resposta_gerada],
                'contexts': [[trecho_texto]] if trecho_texto else [[""]],
                'ground_truth': [resposta_esperada]
            }
            dataset = Dataset.from_dict(data_samples)
            
            # 5. Avaliação do Conteúdo (Ragas Evaluate)
            # show_progress=False evita que barras de carregamento quebrem o design do seu console
            resultado_ragas = evaluate(
                dataset, 
                metrics=[metric_faithfulness, metric_relevancy],
                show_progress=False
            )
            
            # Função auxiliar rápida para garantir que sempre teremos um float válido
            def extrair_nota(valor):
                # Se for uma lista (ex: [0.85]), pegamos o primeiro item
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

            # Extração das notas acessando o objeto diretamente
            try:
                f_score = extrair_nota(resultado_ragas["faithfulness"])
            except (KeyError, TypeError):
                f_score = 0.0
                
            try:
                r_score = extrair_nota(resultado_ragas["answer_relevancy"])
            except (KeyError, TypeError):
                try:
                    r_score = extrair_nota(resultado_ragas["response_relevancy"])
                except (KeyError, TypeError):
                    r_score = 0.0

            # Transformando notas de 0-1 em escala de 0-10 e formatando a justificativa
            nota_final_10 = ((f_score + r_score) / 2) * 10
            justificativa = f"Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f}"
            
            # 6. Avaliação dos Chunks
            correspondencia, lista_chunks_inf = avaliar_chunks(chunks_esperados, chunks_informados)
            
            # --- EXIBIÇÃO NO CONSOLE (DETALHADA) ---
            print(f"❓ Pergunta: {pergunta}")
            print(f"🎯 Resposta Esperada: {resposta_esperada}")
            print(f"🤖 Resposta Obtida: {resposta_gerada}")
            print(f"📂 Chunks Esperados: {chunks_esperados}")
            print(f"📂 Chunks Informados: {chunks_informados}")
            print(f"⚖️  Nota: [{nota_final_10:.1f}/10]")
            print(f"💬 Justificativa: {justificativa}")
            print(f"🔗 Correspondência Chunks: {correspondencia}")
            print("-" * 60)

            # --- ARMAZENAMENTO PARA O EXCEL ---
            resultados_finais.append({
                'ID': id_pergunta,
                'Pergunta': pergunta,
                'Categoria': item.get('categoria', 'N/A'),
                'Resposta Esperada': resposta_esperada,
                'Resposta Obtida': resposta_gerada,
                'Nota Final (0-10)': nota_final_10,
                'Fidelidade (Faithfulness)': f_score,
                'Relevância (Relevancy)': r_score,
                'Justificativa': justificativa,
                'Chunks Esperados': ", ".join(chunks_esperados),
                'Chunks Informados': ", ".join(chunks_informados),
                'Chunks Iguais': correspondencia
            })
            
            # Pequena pausa para evitar sobrecarga de requisições no Rate Limit do Gemini
            time.sleep(2)

        except Exception as e:
            print(f"[!] Erro Crítico no ID {id_pergunta}: {type(e).__name__} - {e}")
            print("--- INÍCIO DO TRACEBACK ---")
            traceback.print_exc()
            print("--- FIM DO TRACEBACK ---")

    # Salva o arquivo final em Excel
    if resultados_finais:
        df = pd.DataFrame(resultados_finais)
        df.to_excel(arquivo_saida, index=False)
        print(f"\n✅ AVALIAÇÃO FINALIZADA COM SUCESSO!")
        print(f"📊 Planilha gerada: {arquivo_saida}")
    else:
        print("\n⚠️ Nenhum dado foi processado para salvar.")

if __name__ == "__main__":
    main()