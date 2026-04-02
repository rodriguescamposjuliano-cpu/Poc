import json
import time
import os
import re
import math
import traceback
import pandas as pd
from dotenv import load_dotenv

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
    prompt_juiz = ChatPromptTemplate.from_messages([
        ("system", """Você é um auditor rigoroso de sistemas RAG. 
        Sua tarefa é comparar a RESPOSTA OBTIDA (gerada pela IA) com a RESPOSTA ESPERADA (Gabarito oficial).

        REGRAS DE PONTUAÇÃO (0 a 10):
        1. Avalie APENAS se a RESPOSTA OBTIDA contém as informações factuais da RESPOSTA ESPERADA. Ignore diferenças de sinônimos ou estilo de escrita.
        2. REGRA DE OURO: Se a RESPOSTA OBTIDA contiver TODAS as informações cruciais da RESPOSTA ESPERADA, a nota DEVE ser no mínimo 7 (podendo chegar a 10 se for perfeita e direta).
        3. Se a RESPOSTA OBTIDA trouxer informações extras, não a penalize, a menos que a informação extra contradiga o gabarito.
        4. Se faltarem informações importantes do gabarito, a nota deve ser menor que 7, proporcional ao que faltou.
        5. Se o gabarito diz que a informação "não foi localizada", e a resposta obtida disser a mesma coisa, a nota é 10.

        FORMATO OBRIGATÓRIO DE SAÍDA (JSON ESTRITO):
        {{
            "nota": <número inteiro ou decimal de 0 a 10>,
            "justificativa": "<Explicação concisa de até 2 linhas do porquê dessa nota, focando no que faltou ou no que estava correto>"
        }}"""),
        ("human", """
            PERGUNTA FEITA PELO USUÁRIO: {pergunta}
            RESPOSTA ESPERADA (GABARITO): {esperada}
            RESPOSTA OBTIDA (SISTEMA): {obtida}
            """)
        ])
    
    chain = prompt_juiz | llm | StrOutputParser()
    
    try:
        resultado_str = chain.invoke({
            "pergunta": pergunta,
            "esperada": resposta_esperada,
            "obtida": resposta_obtida
        }).strip()
        
        # Limpeza robusta do JSON usando regex e replace
        resultado_limpo = resultado_str.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', resultado_limpo, re.DOTALL)
        if match:
            resultado_limpo = match.group(0)
            
        dados = json.loads(resultado_limpo)
        
        return float(dados.get("nota", 0.0)), dados.get("justificativa", "Sem justificativa válida.")
        
    except Exception as e:
        return 0.0, f"Erro no LLM Juiz: {str(e)}"

def main():
    print("="*60)
    print("INICIANDO MOTOR DE AVALIAÇÃO HÍBRIDO (RAGAS + LLM JUIZ)")
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
    
    for item in golden_set:
        id_pergunta = item.get('id')
        pergunta = item.get('pergunta')
        resposta_esperada = item.get('resposta_referencia', '')
        chunks_esperados = item.get('chunks_esperados', [])

        print(f"\n>>> [ID {id_pergunta}] PROCESSANDO PERGUNTA...")
        
        try:
            arquivo_origem = item.get('arquivo_origem', '')
            tenant_id = arquivo_origem.split('_')[0] if arquivo_origem else "001"
            
            # 1. Executa o RAG
            resposta_bruta = ia.responder(pergunta, tenant_id)
            
            # 2. Extrai respostas e contextos
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
            
            # 3. Avaliação de Recusa Exata (Bypass Rápido)
            if resposta_gerada == FRASE_RECUSA and resposta_esperada == FRASE_RECUSA:
                f_score, r_score, nota_ragas_10 = 1.0, 1.0, 10.0
                nota_custom, just_custom = 10.0, "Recusa correta e exata. Bypass aplicado."
                
            else:
                # --- AVALIAÇÃO 1: RAGAS (Matemática e Contexto) ---
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
                    
                try:
                    r_score = extrair_nota(resultado_ragas["answer_relevancy"])
                except:
                    try: r_score = extrair_nota(resultado_ragas["response_relevancy"])
                    except: r_score = 0.0

                nota_ragas_10 = ((f_score + r_score) / 2) * 10

                # --- AVALIAÇÃO 2: LLM JUIZ CUSTOMIZADO (Regra de Negócio) ---
                nota_custom, just_custom = avaliar_resposta_customizada(
                    ia.llm, pergunta, resposta_esperada, resposta_gerada
                )
            
            # 4. Avaliação dos Chunks
            correspondencia, _ = avaliar_chunks(chunks_esperados, chunks_informados)
            esperados_contidos = "SIM" if set(chunks_esperados).issubset(set(chunks_informados)) else "NAO"
            
            # --- EXIBIÇÃO NO CONSOLE ---
            print(f"❓ Pergunta: {pergunta}")
            print(f"🎯 Esperada: {resposta_esperada}")
            print(f"🤖 Obtida: {resposta_gerada}")
            print(f"📂 Chunks Esperados: {chunks_esperados} | Informados: {chunks_informados}")
            print(f"🔗 Iguais: {correspondencia}")
            print(f"🔗 Esperados Contidos nos Obtidos: {esperados_contidos}")
            print(f"📊 NOTA RAGAS: [{nota_ragas_10:.1f}/10] (Fidelidade: {f_score:.2f} | Relevância: {r_score:.2f})")
            print(f"⚖️  NOTA JUIZ: [{nota_custom:.1f}/10]")
            print(f"💬 Justificativa: {just_custom}")
            print("-" * 60)

            # --- ARMAZENAMENTO EXCEL ---
            resultados_finais.append({
                'ID': id_pergunta,
                'Pergunta': pergunta,
                'Categoria': item.get('categoria', 'N/A'),
                'Resposta Esperada': resposta_esperada,
                'Resposta Obtida': resposta_gerada,
                'Nota Juiz (0-10)': nota_custom,
                'Justificativa Juiz': just_custom,
                'Nota Ragas (0-10)': nota_ragas_10,
                'Fidelidade Ragas (0-1)': f_score,
                'Relevância Ragas (0-1)': r_score,
                'Chunks Esperados': ", ".join(chunks_esperados),
                'Chunks Informados': ", ".join(chunks_informados),
                'Chunks Iguais': correspondencia,
                'Esperados Contidos nos Obtidos': esperados_contidos
            })
            
            time.sleep(2)

        except Exception as e:
            print(f"[!] Erro Crítico no ID {id_pergunta}: {type(e).__name__} - {e}")
            traceback.print_exc()

    # Salva o arquivo final
    if resultados_finais:
        df = pd.DataFrame(resultados_finais)
        df.to_excel(arquivo_saida, index=False)
        print(f"\n✅ AVALIAÇÃO FINALIZADA COM SUCESSO!")
        print(f"📊 Planilha gerada: {arquivo_saida}")
    else:
        print("\n⚠️ Nenhum dado foi processado para salvar.")

if __name__ == "__main__":
    main()