from bn_as_fitness_v2 import misturar_e_transformar_com_tnormal, repository, functions

import random       
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time



# Expert Data
expert_data = [
    {"AT": "VL", "AC": "VH", "AE_expert": [0.274, 0.323, 0.274, 0.081, 0.048]},
    {"AT": "VH", "AC": "VL", "AE_expert": [0.172, 0.259, 0.345, 0.172, 0.052]},
    {"AT": "VL",  "AC": "VL",  "AE_expert": [0.333, 0.333, 0.283, 0.050, 0.0]},
    {"AT": "VH",  "AC": "VH", "AE_expert": [0.0, 0.055, 0.273, 0.309, 0.364]},
    {"AT": "VL",  "AC": "M", "AE_expert": [0.2, 0.3, 0.34, 0.1, 0.06]},
    {"AT": "M",   "AC": "VL", "AE_expert": [0.357, 0.357, 0.179, 0.107, 0.0]},
]

def gerar_pesos(n_pais):
    return np.random.randint(1, 6, size=n_pais)  # Valores entre 1 e 5 


variancias = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
# Representação do Indivíduo
class Individuo:
    def __init__(self, n_pais):
        self.funcao = random.choice(list(functions.keys()))
        self.pesos = gerar_pesos(n_pais)
        self.variancia = random.choice(variancias)
        self.fitness = None

    def avaliar(self, repository):
        erros = []
        #print(f"\n[Indivíduo] Função: {self.funcao} | Pesos: {self.pesos} | Variância: {self.variancia}")
   
        for c in expert_data:
            estados_pais = [c["AT"], c["AC"]]
            probs_model = misturar_e_transformar_com_tnormal(
                estados_pais, self.pesos, repository,
                variance=self.variancia,
                func_comb=functions[self.funcao]
            )
            #print(f"  AT={estados_pais[0]}, AC={estados_pais[1]} → Probs: {probs_model}")
            erro = mean_squared_error(c["AE_expert"], probs_model)
            erros.append(erro)
        self.fitness = np.mean(erros)
        return self.fitness

# Inicialização da População
def inicializar_populacao(tam_pop, n_pais):
    return [Individuo(n_pais) for _ in range(tam_pop)]

# Seleção por Torneio
def selecao_torneio(populacao, k=3):
    return min(random.sample(populacao, k), key=lambda ind: ind.fitness)

# Crossover de 1 Ponto
def crossover(pai1, pai2):
    filho = Individuo(len(pai1.pesos))
    filho.funcao = pai1.funcao if random.random() < 0.5 else pai2.funcao
    ponto = random.randint(1, len(pai1.pesos) - 1)
    filho.pesos = np.concatenate((pai1.pesos[:ponto], pai2.pesos[ponto:]))
    #filho.pesos = np.round(filho.pesos / np.sum(filho.pesos), 2) #REMOVIDA A NORMALIZACÃO
    filho.variancia = random.choice([pai1.variancia, pai2.variancia])
    return filho

# Mutação
def mutacao(ind, taxa_mutacao):
    if random.random() < taxa_mutacao:
        ind.funcao = random.choice(list(functions.keys()))
    if random.random() < taxa_mutacao:
        ind.pesos = gerar_pesos(len(ind.pesos))
    if random.random() < taxa_mutacao:
        ind.variancia = random.choice(variancias)
    return ind

# Algoritmo Genético Principal
def algoritmo_genetico(tam_pop, n_pais, max_gen, taxa_mutacao, repository, functions):
    populacao = inicializar_populacao(tam_pop, n_pais)
    for ind in populacao:
        ind.avaliar(repository)

    for geracao in range(max_gen):
        nova_populacao = []
        elite = min(populacao, key=lambda ind: ind.fitness)
        nova_populacao.append(elite)  # Elitismo

        while len(nova_populacao) < tam_pop:
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            # ✅ Controle da Taxa de Cruzamento (80%)
            if random.random() < 0.8:
                filho = crossover(pai1, pai2)
            else:
                filho = selecao_torneio(populacao)  # Replicação direta de um pai
            
            filho = mutacao(filho, taxa_mutacao)
            filho.avaliar(repository)
            nova_populacao.append(filho)

        populacao = nova_populacao
        
        # Logging
        melhores = sorted(populacao, key=lambda ind: ind.fitness)
        #print(f"[GEN {geracao}] Melhor Brier: {melhores[0].fitness:.5f} | Função: {melhores[0].funcao} | Pesos: {melhores[0].pesos} | Variância: {melhores[0].variancia}")

    melhor = min(populacao, key=lambda ind: ind.fitness)
    print("\nMelhor configuração encontrada:")
    print(f"Função: {melhor.funcao}, Pesos: {melhor.pesos}, Variância: {melhor.variancia}, Brier Score: {melhor.fitness}")
    return melhor

# Exemplo de Execução
inicio = time.time()
melhor_ind = algoritmo_genetico(tam_pop=50, n_pais=2, max_gen=10, taxa_mutacao=0.1, repository=repository, functions=functions)
print(melhor_ind)
fim = time.time()


print(f"Tempo de execução AG:  {fim - inicio:.4f} segundos")

minutos = int((fim - inicio) // 60)
segundos = (fim - inicio) % 60
print(f"Tempo de execução AG: {minutos} minutos e {segundos:.2f} segundos")
