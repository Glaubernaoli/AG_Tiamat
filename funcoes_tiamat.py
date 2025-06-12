import random 
import pickle
import numpy as np


###############################################################################
#                             Cria população                                  #
###############################################################################

def gene_feat(valores_possiveis):
    """Sorteia um valor para a feature dentro do valores_posiveis permitido com valor mínimo e máximo"""
    valores_possiveis = range(int(valores_possiveis[0]), int(valores_possiveis[1]))
    gene = random.choice(valores_possiveis)
    return gene

def cria_candidato_feature(valores_possiveis):
    """Cria uma lista com n valores entre o intervalo.

    Args:
      n: inteiro que representa o número de caixas.
      intervalo: lista com o valor mínimo e máximo das features
    """

    candidato = []
    for valor in valores_possiveis:
        gene = gene_feat(valor)
        candidato.append(gene)
    return candidato

def cria_populacao_feature(tamanho, valores_possiveis):

    """Cria uma população para encontrar o melhor candidato com maior dureza.

    Args:
      tamanho: tamanho da população
      n: inteiro que representa o número de caixas de cada indivíduo.

    """
    populacao = []
    for _ in range(tamanho):
        populacao.append(cria_candidato_feature(valores_possiveis))
    return populacao

###############################################################################
#                             Função Objetivo                                 #
###############################################################################

with open("knn.pkl", "rb") as modelo:
    modelo = pickle.load(modelo)


def funcao_objetivo_feature(candidato):
    """Computa a função objetivo no problema de otimização da dureza

    Args:
      candidato: uma lista contendo os candidatos para a resolução do problema

    """
    previsao = modelo.predict(np.array(candidato).reshape(1, -1))

    return previsao



def funcao_objetivo_pop_feature(populacao):
    """Computa a função objetivo para uma população no problema de otimização da dureza

    Args:
      populacao: lista contendo os individuos do problema

    """
    fitness = []
    for individuo in populacao:
        fitness.append(funcao_objetivo_feature(individuo))
    return fitness


###############################################################################
#                               Função Seleção                                #
###############################################################################
def selecao_elitismo_torneio(populacao, fitness, tamanho_torneio, n_elitismo=5):
    selecionados = []
 
    ordem = sorted(zip(fitness, populacao), reverse=True)
 
    elitistas = [ind for _, ind in ordem[:n_elitismo]]
    selecionados.extend(elitistas)
 
    populacao_restante = [ind for ind in populacao if ind not in elitistas]
    fitness_restante = [fitness[populacao.index(ind)] for ind in populacao_restante]
 
    n_restante = len(populacao) - n_elitismo
 
    for _ in range(n_restante):
        indices_sorteados = random.sample(range(len(populacao_restante)), tamanho_torneio)
        torneio = [populacao_restante[i] for i in indices_sorteados]
        fitness_torneio = [fitness_restante[i] for i in indices_sorteados]
 
        # Seleciona o melhor do torneio
        melhor_indice = fitness_torneio.index(max(fitness_torneio))
        selecionado = torneio[melhor_indice]
        selecionados.append(selecionado)
 
    return selecionados


# def selecao_elitismo(populacao, fitness):     
#     fitness_sorteado = sorted(zip(populacao, fitness), key=lambda x: x[1], reverse=True)     
#     t1 = int(len(fitness_sorteado) * 0.002)     
#     indiv_elite = [i for i, (indice, score) in enumerate(fitness_sorteado) if score >= fitness_sorteado[t1][1]]
    
#     return indiv_elite

# def selecao_torneio_max(populacao, fitness, tamanho_torneio):
#     """Faz a seleção de uma população usando torneio.

#     Nota: da forma que está implementada, só funciona em problemas de
#     maximização.

#     Args:
#       populacao: lista contendo os individuos do problema
#       fitness: lista contendo os valores computados da funcao objetivo
#       tamanho_torneio: quantidade de invíduos que batalham entre si

#     """
#     selecionados = []

#     for _ in range(len(populacao)):
#         sorteados = random.sample(populacao, tamanho_torneio)

#         fitness_sorteados = []
#         for individuo in sorteados:
#             indice_individuo = populacao.index(individuo)
#             fitness_sorteados.append(fitness[indice_individuo])

#         max_fitness = max(fitness_sorteados)
#         indice_max_fitness = fitness_sorteados.index(max_fitness)
#         individuo_selecionado = sorteados[indice_max_fitness]

#         selecionados.append(individuo_selecionado)

#     return selecionados

###############################################################################
#                              Função Mutação                                 #
###############################################################################


def mutacao_simples(populacao, chance_de_mutacao, valores_possiveis):
    """Realiza mutação simples

    Args:
      populacao: lista contendo os indivíduos do problema
      chance_de_mutacao: float entre 0 e 1 representando a chance de mutação
      valores_possiveis: lista com todos os valores possíveis dos genes

    """
    for individuo in populacao:
        if random.random() < chance_de_mutacao:
            gene = random.randint(0, len(individuo) - 1)
            valor_gene = individuo[gene]
            intervalo_feature = valores_possiveis[gene]
            
            #valores_sorteio = set(valores_possiveis) - set([valor_gene])
            
            valor_substituicao= random.uniform(intervalo_feature[0], intervalo_feature[1])
            
            if valor_substituicao == valor_gene:
                valor_substituicao = random.uniform(intervalo_feature[0], intervalo_feature[1])
            
            individuo[gene] = valor_substituicao

###############################################################################
#                              Função Cruzamento                              #
###############################################################################

def cruzamento_ponto_simples(pai, mae, chance_de_cruzamento):
    """Realiza cruzamento de ponto simples

    Args:
      pai: lista representando um individuo
      mae: lista representando um individuo
      chance_de_cruzamento: float entre 0 e 1 representando a chance de cruzamento

    """
    if random.random() < chance_de_cruzamento:
        corte = random.randint(1, len(mae) - 1)
        filho1 = pai[:corte] + mae[corte:]
        filho2 = mae[:corte] + pai[corte:]
        return filho1, filho2
    else:
        return pai, mae