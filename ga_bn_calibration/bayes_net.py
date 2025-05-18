import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import truncnorm
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde
import json
import ast
import time

from sklearn.metrics import mean_squared_error
import itertools

import pickle

# ================================
# Class Bayesian Network
# ================================
class BNetwork:
    def __init__(self):
        self.model = BayesianNetwork()
        self.nodes = {}
        self.evidence = {}

    def createNode(self, node_id, name, outcomes):
        self.nodes[node_id] = {"name": name, "outcomes": outcomes}
        self.model.add_node(node_id)

    def addEdge(self, parent_id, child_id):
        self.model.add_edge(parent_id, child_id)

    def setNodeCPD(self, node_id, cpt_values):
        parent_ids = list(self.model.get_parents(node_id))
        parent_cards = [len(self.nodes[p]["outcomes"]) for p in parent_ids]
        var_card = len(self.nodes[node_id]["outcomes"])
        cpt_values = np.array(cpt_values)
        if cpt_values.shape != (var_card, np.prod(parent_cards)):
            raise ValueError(f"Format error for {node_id}. Espected {(var_card, np.prod(parent_cards))}, but received {cpt_values.shape}")
        cpd = TabularCPD(
            variable=node_id,
            variable_card=var_card,
            values=cpt_values.tolist(),
            evidence=parent_ids if parent_ids else None,
            evidence_card=parent_cards if parent_cards else None,
            state_names={node_id: self.nodes[node_id]["outcomes"], **{pid: self.nodes[pid]["outcomes"] for pid in parent_ids}}
        )
        self.model.add_cpds(cpd)

    def updateBeliefs(self):
        self.model.check_model()
        infer = VariableElimination(self.model)
        evidence_dict = self.evidence if self.evidence else {}
        return {
            node_id: infer.query([node_id], evidence=evidence_dict).values.tolist()
            for node_id in self.nodes if node_id not in evidence_dict
        }

    def setEvidence(self, node_id, state_name):
        self.evidence[node_id] = state_name

    def calculateTPN(self, node_id):
        infer = VariableElimination(self.model)
        return infer.query([node_id], evidence=self.evidence).values.tolist()



# ================================
# Begin of the Bayesian Network
# ================================
bn = BNetwork()

states = {
    'VL': {'lower': 0.0, 'upper': 0.2},
    'L':  {'lower': 0.2, 'upper': 0.4},
    'M':  {'lower': 0.4, 'upper': 0.6},
    'H':  {'lower': 0.6, 'upper': 0.8},
    'VH': {'lower': 0.8, 'upper': 1.0}
}

bn.createNode("AT", "Aptidão Técnica", list(states.keys()))
bn.createNode("AC", "Aptidão Colaborativa", list(states.keys()))
bn.createNode("AE", "Aptidão Equipe", list(states.keys()))

bn.addEdge("AT", "AE")
bn.addEdge("AC", "AE")

cpd_at = [[0.2], [0.2], [0.2], [0.2], [0.2]]
cpd_ac = [[0.2], [0.2], [0.2], [0.2], [0.2]]

def wmean(*args):
    if len(args) % 2 != 0:
        raise ValueError("An even number of arguments (weight-value pairs) is required.")

    partial_sum_value = 0.0
    partial_sum_weight = 0.0

    for i in range(0, len(args), 2):
        w = args[i]
        x = args[i + 1]
        partial_sum_value += w * x
        partial_sum_weight += w

    if partial_sum_weight == 0:
        return None

    return partial_sum_value / partial_sum_weight



#nova:
def wmin(*args):
    if len(args) % 2 != 0:
        raise ValueError("An even number of arguments (weight-value pairs) is required.")

    n = len(args) // 2
    if n < 2:
        return None

    weights = [args[2 * i] for i in range(n)]
    values = [args[2 * i + 1] for i in range(n)]

    if sum(weights) == 0:
        return None

    S = sum(values)
    current_min = float('inf')
    for i in range(n):
        w_i = weights[i]
        denom = w_i + (n - 1)
        numerator = w_i * values[i] + (S - values[i])
        e_i = numerator / denom
        current_min = np.minimum(current_min, e_i)  # Comparação elemento a elemento (alteracao feita no codigo)

    return current_min


   
    
    
    

    




#nova:
def wmax(*args):
    if len(args) % 2 != 0:
        return None

    n = len(args) // 2
    if n < 2:
        return None

    weights = []
    values = []
    for i in range(n):
        w_i = args[2 * i]
        x_i = args[2 * i + 1]
        if w_i < 0 or not np.all((0 <= x_i) & (x_i <= 1)): #adaptando essa linha para trabalhar com vetores de amostras, e não com valores escalares.
            return None
        weights.append(w_i)
        values.append(x_i)

    all_zero_denominators = True
    for i in range(n):
        denom = weights[i] + (n - 1)
        if denom != 0:
            all_zero_denominators = False
            break
    if all_zero_denominators:
        return None

    max_e = None
    sum_all = sum(values)
    for i in range(n):
        w_i = weights[i]
        x_i = values[i]
        denom = w_i + (n - 1)
        numerator = w_i * x_i + (sum_all - x_i)
        e_i = numerator / denom
        if max_e is None: #adaptando aqui para comparar arrays posição a posição.
            max_e = e_i
        else:
            max_e = np.maximum(max_e, e_i)


    return max_e



#nova:
def mixminmax(*args):
    """
    Calculates:
      (wmin * min(values) + wmax * max(values)) / (wmin + wmax)
    Subject to:
      - All weights must be non-negative
      - Return None if sum of weights is zero
      - Raise ValueError if invalid arguments are passed
    Expected Input:
      mixminmax(w1, x1, w2, x2, ..., wn, xn)
    """
    if len(args) % 2 != 0:
        raise ValueError("An even number of arguments (weight-value pairs) is required.")

    n = len(args) // 2
    weights = [args[2 * i] for i in range(n)]
    values = [args[2 * i + 1] for i in range(n)]

    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative.")
    if sum(weights) == 0:
        return None

    values_array = np.array(values)  # matriz N x amostras
    mins = np.min(values_array, axis=0)
    maxs = np.max(values_array, axis=0)

    return (weights[0] * mins + weights[1] * maxs) / sum(weights)



# ============================================


# ================================
# Etapa 2 – Mistura com a funcao escolhida e conversão com TNormal
# ================================
import numpy as np
from scipy.stats import truncnorm

def misturar_e_transformar_com_tnormal(estados_pais, pesos, repositorio, variance, func_comb):
    #Validações
    if not estados_pais:
        raise KeyError("estados_pais is empty")

    pesos = np.array(pesos, dtype=float)
    if len(pesos) != len(estados_pais):
        raise ValueError("length mismatch between pesos and estados_pais")
    if np.any(pesos < 0):
        raise ValueError("invalid weights (negative values)")
    if np.sum(pesos) <= 0:
        raise ValueError("invalid weights (sum ≤ 0)")

    amostras_por_pai = []
    for estado in estados_pais:
        if estado not in repositorio:
            raise KeyError(f"Estado {estado} not in repositorio")
        samples = repositorio[estado]['amostras']
        if len(samples) < 10000:
            raise ValueError("less than 10 000 samples")
        #amostras_por_pai.append(np.random.choice(samples, size=10000, replace=False))
        amostras_por_pai.append(np.array(samples[:10000]))

    intercalado = [item for pair in zip(pesos, amostras_por_pai) for item in pair] #adaptado para empacotar pesos e amostras
    valores_continuos = func_comb(*intercalado)

    
    if (not isinstance(valores_continuos, np.ndarray) or
        valores_continuos.ndim != 1 or
        len(valores_continuos) != 10000 or
        np.any(valores_continuos < 0) or
        np.any(valores_continuos > 1)):
        raise ValueError("incompatible shape or values returned by func_comb")

    mean = np.mean(valores_continuos)
    if variance <= 0:
        variance = 0.0001 #ATRIBUINDO VALOR MINIMO PRA EVITAR ERERRO COM VARIANCIA NULA OU NEGATICA
    max_variance = mean * (1 - mean)
    if variance > max_variance:
        variance = max(max_variance, 0.0001)  # garante limite mínimo pra variancia
        #raise ValueError("variance exceeds μ(1-μ)")
    std = np.sqrt(variance)
    
    try:
        dist = truncnorm((0 - mean) / std, (1 - mean) / std, loc=mean, scale=std)
    except (FloatingPointError, ZeroDivisionError):
        raise ValueError("invalid distribution parameters")

    bins = np.linspace(0, 1, 6)
    probs = np.array([dist.cdf(bins[i+1]) - dist.cdf(bins[i]) for i in range(5)])
    probs = np.round(probs, 3)

    if np.abs(np.sum(probs) - 1) > 1e-6:
        probs /= np.sum(probs)  # Renormaliza

    return np.round(probs, 3)



# ================================
# Etapa 3 – Construção da tabela TPN para AE (25 combinações)
# ================================
'''
    
ae_cpt = []

for at in states:
    for ac in states:
        #Melhor configuração encontrada:
        #Função: WMIN, Pesos: [0.56 0.44], Variância: 0.1, Brier Score: 0.0026242
        probs = misturar_e_transformar_com_tnormal(
            estados_pais=[at, ac],
            pesos=[0.56, 0.44],
            repositorio=repositorio,
            variance=0.1,#min 0.0005 e max 1 escrever justificativa lógica
            func_comb=wmin
        )
        ae_cpt.append(probs)
        #print(f"(AT={at}, AC={ac}) → {probs}")

ae_cpt = np.array(ae_cpt).T  # Transpor para [5 x 25]

# ================================
# Etapa 4 – Atribuir CPDs à rede
# ================================
bn.setNodeCPD("AT", cpd_at)
bn.setNodeCPD("AC", cpd_ac)
bn.setNodeCPD("AE", ae_cpt.tolist())

expert_data = [
    {"AT": "VL", "AC": "VH", "AE_expert": [0.274, 0.323, 0.274, 0.081, 0.048]},
    {"AT": "VH", "AC": "VL", "AE_expert": [0.172, 0.259, 0.345, 0.172, 0.052]},
    {"AT": "VL",  "AC": "VL",  "AE_expert": [0.333, 0.333, 0.283, 0.050, 0.0]},
    {"AT": "VH",  "AC": "VH", "AE_expert": [0.0, 0.055, 0.273, 0.309, 0.364]},
    {"AT": "VL",  "AC": "M", "AE_expert": [0.2, 0.3, 0.34, 0.1, 0.06]},
    {"AT": "M",   "AC": "VL", "AE_expert": [0.357, 0.357, 0.179, 0.107, 0.0]},
]






# ================================
# Geração da CPT já calibrada com a melhor configuração de função e variância
# ================================

# A CPT será construída usando os melhores parâmetros encontrados
ae_cpt_calibrada = []

for at in states:
    for ac in states:
        probs = misturar_e_transformar_com_tnormal(
            at_estado=at,
            ac_estado=ac,
            repositorio=repositorio,
            peso_at=melhor_config["peso_at"],
            peso_ac=melhor_config["peso_ac"],
            variance=melhor_config["variance"],  
            func_comb=funcoes[melhor_config["funcao"]]
        )
        ae_cpt_calibrada.append(probs)
        print(f"(AT={at}, AC={ac}) → {probs}")

# Transpor a matriz para o formato [5 linhas × 25 colunas]
ae_cpt_calibrada = np.array(ae_cpt_calibrada).T

# Atribui a nova CPT ao nó AE
bn.setNodeCPD("AE", ae_cpt_calibrada.tolist())
print("\n✅ CPT calibrada atribuída ao nó AE com sucesso.")

'''


funcoes = {
    "WMEAN": wmean,
    "WMIN": wmin,
    "WMAX": wmax,
    "MIXMINMAX": mixminmax
}
with open('repositorio.pkl', 'rb') as f:
        repositorio = pickle.load(f)
# ======== Executar e salvar os 48 cenários ========
if __name__ == "__main__":
    with open('repositorio.pkl', 'rb') as f:
        repositorio = pickle.load(f)
    
    
    '''
    # lê os cenários no csv
    df = pd.read_csv("cenarios_estratificados_2_3_pais.csv")

    # Mapeamento das funções
    func_map = {
        'wmean': wmean,
        'wmin': wmin,
        'wmax': wmax,
        'mixminmax': mixminmax
    }

    #repositorio = gerar_amostras_base_por_estado()
    def carregar_amostras_json(caminho_arquivo='repositorio.json'):
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            # converte as listas de volta para arrays NumPy
            for estado in dados:
                dados[estado]['amostras'] = np.array(dados[estado]['amostras'])
            return dados

    repositorio = carregar_amostras_json()
    
    linhas_csv = []

    for _, c in df.iterrows():
        estados_pais = [c["at"], c["ac"]]
        
        # Verifica dinamicamente se há um terceiro pai (hn)
        if "hn" in c and pd.notna(c["hn"]):
            estados_pais.append(c["hn"])
            
        # Converte a string de pesos e variância para lista e float
        pesos = [float(p) for p in ast.literal_eval(c["pesos"])]
        variance_raw = ast.literal_eval(str(c["variance"]))
        if isinstance(variance_raw, list):
            variance = float(variance_raw[0])
        else:
            variance = float(variance_raw)
        func = func_map[c["funcao"].lower()]

        probs = misturar_e_transformar_com_tnormal(estados_pais, pesos, repositorio, variance, func)

        linha = {
            "id": c["id"],
            "funcao": c["funcao"],
            "estados": "-".join(estados_pais),
            "pesos": "-".join(map(str, pesos)),
            "variance": variance,
            "prob_VL": probs[0],
            "prob_L": probs[1],
            "prob_M": probs[2],
            "prob_H": probs[3],
            "prob_VH": probs[4],
        }

        linhas_csv.append(linha)

    # Salvar no CSV
    df_resultado = pd.DataFrame(linhas_csv)
    df_resultado.to_csv("resultados_modelo_60cenarios.csv", index=False, encoding="utf-8")
    print("Resultados salvos em resultados_modelo_60cenarios.csv")
    '''
