import pandas as pd
import matplotlib.pyplot as plt
import os

def gerar_graficos_sindico():
    file_path = "dashboard_sindico.csv"
    
    if not os.path.exists(file_path):
        print("❌ O arquivo de logs ainda não existe. Converse com o Severino primeiro!")
        return

    # Lendo os dados
    df = pd.read_csv(file_path)
    
    if df.empty:
        print("❌ O arquivo de logs está vazio.")
        return

    # 1. Gráfico de Pizza: Distribuição de Assuntos
    plt.figure(figsize=(10, 6))
    df['Categoria'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Distribuição de Conflitos e Dúvidas - Gestão Proativa')
    plt.ylabel('') # Remove o label lateral
    
    # Salva o gráfico para o TCC
    plt.savefig('grafico_conflitos.png')
    print("✅ Gráfico 'grafico_conflitos.png' gerado com sucesso!")
    
    # 2. Resumo para o relatório do TCC
    print("\n--- RESUMO PARA O SÍNDICO ---")
    print(df['Categoria'].value_counts())
    
    plt.show()

if __name__ == "__main__":
    gerar_graficos_sindico()