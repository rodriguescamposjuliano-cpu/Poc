import os
import sys
import time
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
from langchain_core.documents import Document
from rich.text import Text
import logging
import warnings

from serverino_ia import SeverinoIA
from avaliador import Avaliador
from prompts import obtenha_prompt_severino_juiz

# Evita outro warning comum
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Ignora avisos de versão
warnings.filterwarnings("ignore", category=UserWarning) 

# Ignora os "DeprecationWarning" do Ragas
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Esconde os logs de INFO de requisições HTTP e do Google GenAI
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("ragas").setLevel(logging.WARNING)

# Recursos NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

load_dotenv()
console = Console()

def limpar_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def escrever_com_efeito(texto, delay=0.01):
    for char in texto:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def exiba_resposta(resposta_final):
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

if __name__ == "__main__":
    ia = SeverinoIA()
    
    while True:
        limpar_console()
        console.print(Panel.fit("SEVERINO IA", style="bold blue"))

        console.print("[1] Conversar com IA")
        console.print("[2] Rodar avaliação (RAGAS + Juiz + Recalls)")
        console.print("[sair] Encerrar\n", markup=False)

        opcao = input("Escolha: ").strip()

        if opcao == "sair":
            break

        if opcao == "2":
            with console.status("[bold yellow]Realizando avaliação e benchmarks...[/bold yellow]\n"):
                avaliador = Avaliador(ia, obtenha_prompt_severino_juiz())
                avaliador.rodar_avaliacao("golden_set_apresentacao.json")

            input("\nPressione ENTER para voltar ao menu principal...")
            continue

        if opcao != "1":
            continue

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
                sys.exit()

            if not pergunta:
                continue

            with console.status("[bold yellow]Consultando..."):
                inicio = time.time()
                resposta_final = ia.responder(pergunta, tenant)
                tempo = time.time() - inicio

            # --- APRESENTAÇÃO ORGANIZADA ---
            console.print("\n" + "━" * 50, style="yellow")
            console.print(f"[bold yellow]Severino[/bold yellow] [dim](em {tempo:.2f}s)[/dim]")
            exiba_resposta(resposta_final)