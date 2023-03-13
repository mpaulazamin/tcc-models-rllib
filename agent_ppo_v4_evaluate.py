import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.algorithms.algorithm import Algorithm

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from controle_temperatura_saida import simulacao_malha_temperatura
from controle_temperatura_saida import modelagem_sistema
from controle_temperatura_saida import modelo_valvula_saida
from controle_temperatura_saida import calculo_iqb
from controle_temperatura_saida import custo_eletrico_banho
from controle_temperatura_saida import custo_gas_banho
from controle_temperatura_saida import custo_agua

seed = 33
random.seed(seed)
np.random.seed(seed)


class ShowerEnv(gym.Env):
    """Ambiente para simulação do modelo de chuveiro."""

    def __init__(self, config=None):

        # Tempo de simulação:
        self.dt = 0.01

        # Tempo de cada iteracao:
        self.tempo_iteracao = 2

        # Distúrbios e temperatura ambiente - Fd, Td, Tf, Tinf:
        self.Fd = 0
        self.Td = 25
        self.Tf = 25
        self.Tinf = 25

        # Não utiliza split-range:
        self.split_range = 0

        # Potência da resistência elétrica em kW:
        self.potencia_eletrica = 5.5

        # Potência do aquecedor boiler em kcal/h:
        self.potencia_aquecedor = 29000

        # Custo da energia elétrica em kWh, do kg do gás, e do m3 da água:
        self.custo_eletrico_kwh = 2
        self.custo_gas_kg = 1
        self.custo_agua_m3 = 3

        # Ações - xq, SPTq, xs, Sr:
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(low=0.01, high=0.99, shape=(1,), dtype=np.float32),
                gym.spaces.Box(low=40, high=55, shape=(1,), dtype=np.float32),
                gym.spaces.Box(low=0.3, high=0.7, shape=(1,), dtype=np.float32),
                gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            ),
        )

        # Estados - Ts, Tq, Tt, h, Fs, xf, iqb:
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([100, 100, 100, 10000, 100, 1, 1]),
            dtype=np.float32, 
        )

    def reset(self):

        # Random seed:
        super().reset(seed=seed)

        # Tempo inicial:
        self.tempo_inicial = 0

        # Nível do tanque de aquecimento e setpoint:
        self.h = 80
        self.SPh = 80

        # Temperatura de saída:
        self.Ts = self.Tinf

        # Temperatura do boiler:
        self.Tq = 55

        # Temperatura do tanque:
        self.Tt = self.Tinf

        # Vazão de saída:
        self.Fs = 0

        # Abertura da válvula quente:
        self.xq = 0

        # Abertura da válvula fria:
        self.xf = 0

        # Índice de qualidade do banho:
        self.iqb = 0

        # Custo elétrico do banho:
        self.custo_eletrico = 0

        # Custo do gás do banho:
        self.custo_gas = 0

        # Custo da água do banho:
        self.custo_agua = 0

        # Condições iniciais - Tq, h, Tt, Ts:
        self.Y0 = np.array([self.Tq, self.h] + 50 * [self.Tinf])

        # Define o buffer para os ganhos integral e derivativo das malhas de controle:
        # 0 - malha boiler, 1 - malha nível, 2 - malha tanque, 3 - malha saída
        id = [0, 1, 2, -1]
        self.Kp = np.array([1, 0.3, 2.0, 0.51])
        self.b = np.array([1, 1, 1, 0.8])
        self.I_buffer = self.Kp * self.Y0[id] * (1 - self.b)
        self.D_buffer = np.array([0, 0, 0, 0])  

        # Estados - Ts, Tq, Tt, h, Fs, xf, iqb:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.iqb],
                             dtype=np.float32)
        
        return self.obs

    def step(self, action):

        # Tempo de cada iteração:
        self.tempo_final = self.tempo_inicial + self.tempo_iteracao

        # Abertura da válvula quente:
        self.xq = round(action[0][0], 2)

        # Fração de aquecimento do boiler:
        self.SPTq = round(action[1][0], 1)

        # Abertura da válvula de saída:
        self.xs = round(action[2][0], 2)

        # Fração da resistência elétrica
        self.Sr = round(action[3][0], 2)

        # Variáveis para simulação - tempo, SPTq, SPh, xq, xs,Tf, Td, Tinf, Fd, Sr:
        self.UT = np.array(
            [   
                [self.tempo_inicial, self.SPTq, self.SPh, self.xq, self.xs, self.Tf, self.Td, self.Tinf, self.Fd, self.Sr],
                [self.tempo_final, self.SPTq, self.SPh, self.xq, self.xs, self.Tf, self.Td, self.Tinf, self.Fd, self.Sr]
            ]
        )

        # Solução do sistema:
        self.TT, self.YY, self.UU, self.Y0, self.I_buffer, self.D_buffer = simulacao_malha_temperatura(
            modelagem_sistema, 
            self.Y0, 
            self.UT, 
            self.dt, 
            self.I_buffer,
            self.D_buffer,
            self.Tinf,
            self.split_range
        )

        # Valor final da temperatura do boiler:
        self.Tq = self.YY[:,0][-1]

        # Valor final do nível do tanque:
        self.h = self.YY[:,1][-1]

        # Valor final da temperatura do tanque:
        self.Tt = self.YY[:,2][-1]

        # Valor final da temperatura de saída:
        self.Ts = self.YY[:,3][-1]

        # Fração do aquecedor do boiler utilizada durante a iteração:
        self.Sa_total =  self.UU[:,0]

        # Fração da resistência elétrica utilizada durante a iteração:
        self.Sr = self.UU[:,8][-1]

        # Valor final da abertura de corrente fria:
        self.xf = self.UU[:,1][-1]

        # Valor final da abertura de corrente quente:
        self.xq = self.UU[:,2][-1]

        # Valor final da abertura da válvula de saída:
        self.xs = self.UU[:,3][-1]

        # Valor final da vazão de saída:
        self.Fs = modelo_valvula_saida(self.xs)

        # Cálculo do índice de qualidade do banho:
        self.iqb = calculo_iqb(self.Ts, self.Fs)

        # Cálculo do custo elétrico do banho:
        self.custo_eletrico = custo_eletrico_banho(self.Sr, self.potencia_eletrica, self.custo_eletrico_kwh, self.tempo_iteracao)

        # Cálculo do custo de gás do banho:
        self.custo_gas = custo_gas_banho(self.Sa_total, self.potencia_aquecedor, self.custo_gas_kg, self.dt)

        # Cálculo do custo da água:
        self.custo_agua = custo_agua(self.Fs, self.custo_agua_m3, self.tempo_iteracao)

        # Estados - Ts, Tq, Tt, h, Fs, xf, iqb, custo_eletrico, custo_gas, custo_agua:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.iqb],
                             dtype=np.float32)

        # Define a recompensa:
        custos_total = self.custo_eletrico + self.custo_gas + self.custo_agua
        reward = self.iqb / (self.iqb + custos_total)

        # Incrementa tempo inicial:
        self.tempo_inicial = self.tempo_inicial + self.tempo_iteracao

        # Termina o episódio se o tempo for maior que 14 ou se o nível do tanque ultrapassar 100:
        done = False
        if self.tempo_final == 14 or self.h > 100: 
            done = True

        # Para visualização:
        self.SPTq_total = np.repeat(self.SPTq, 201)
        self.Tq_total = self.YY[:,0]
        self.SPh_total = np.repeat(self.SPh, 201)
        self.h_total = self.YY[:,1]
        self.Tt_total = self.YY[:,2]
        self.Ts_total = self.YY[:,3]
        self.xq_total = self.UU[:,2]
        self.xf_total = self.UU[:,1]
        self.Sr_total = np.repeat(self.Sr, 201)
        self.xs_total = np.repeat(self.xs, 201)
        self.Fs_total = np.repeat(self.Fs, 201)    
        self.Fd_total = np.repeat(self.Fd, 201) 
        self.Td_total = np.repeat(self.Td, 201) 
        self.Tf_total = np.repeat(self.Tf, 201) 
        self.Tinf_total = np.repeat(self.Tinf, 201) 

        info = {"SPTq": self.SPTq_total,
                "Tq": self.Tq_total,
                "SPh": self.SPh_total,
                "h": self.h_total,
                "Tt": self.Tt_total,
                "Ts": self.Ts_total,
                "Sr": self.Sr_total,
                "Sa": self.Sa_total,
                "xq": self.xq_total,
                "xf": self.xf_total,
                "xs": self.xs_total,
                "Fs": self.Fs_total,
                "iqb": self.iqb,
                "recompensa": self.reward,
                "custo_eletrico": self.custo_eletrico,
                "custo_gas": self.custo_gas,
                "custo_agua": self.custo_agua,
                "Fd": self.Fd_total,
                "Td": self.Td_total,
                "Tf": self.Tf_total,
                "Tinf": self.Tinf_total}

        return self.obs, reward, done, info
    
    def render(self):
        pass


# Inicia o Ray:
ray.shutdown()
info = ray.init(ignore_reinit_error=True)

# Carrega o agente salvo:
config = ppo.PPOConfig()
config.environment(env=ShowerEnv)
agent = config.build()
checkpoint_root = "C:\\Users\\maria\\ray_ppo_checkpoints\\agent_ppo_v4\\checkpoint_000050"
agent.restore(checkpoint_root)

# Constrói o ambiente:
env = ShowerEnv()
done = False
obs = env.reset()

# Para visualização
SPTq_list = []
Tq_list = []
SPh_list = []
h_list = []
Tt_list = []
SPTs_list = []
Ts_list = []
Sr_list = []
Sa_list = []
xq_list = []
xf_list = []
xs_list = []
Fs_list = []
iqb_list = []
recompensa_list = []
custo_eletrico_list = []
custo_gas_list = []
custo_agua_list = []
Fd_list = []
Td_list = []
Tf_list = []
Tinf_list = []
time_total = np.arange(start=0, stop=14 + 0.07, step=0.01, dtype="float")
time_actions = np.arange(start=1, stop=8, step=1, dtype="int")

# Roda episódios com ações sugeridas pelo agente treinado:
for i in range(0, 1):

    episode_reward = 0
    print(f"Episódio {i}.")

    for i in range(0, 7):

        # Seleciona ações:
        action = agent.compute_single_action(obs)
        print(f"Iteração: {i}")
        print(f"Ações: {action}")

        # Retorna os estados e a recompensa:
        obs, reward, done, info = env.step(action)
        print(f"Estados: {obs}")

        # Recompensa total:
        episode_reward += reward
        print(f"Recompensa: {reward}.")

        # Para visualização:
        SPTq_list.append(info.get("SPTq"))
        Tq_list.append(info.get("Tq"))
        SPh_list.append(info.get("SPh"))
        h_list.append(info.get("h"))
        Tt_list.append(info.get("Tt"))
        Ts_list.append(info.get("Ts"))
        Sr_list.append(info.get("Sr"))
        Sa_list.append(info.get("Sa"))
        xq_list.append(info.get("xq"))
        xf_list.append(info.get("xf"))
        xs_list.append(info.get("xs"))
        Fs_list.append(info.get("Fs"))
        iqb_list.append(info.get("iqb"))
        recompensa_list.append(info.get("recompensa"))
        custo_eletrico_list.append(info.get("custo_eletrico"))
        custo_gas_list.append(info.get("custo_gas"))
        custo_agua_list.append(info.get("custo_agua"))
        Fd_list.append(info.get("Fd"))
        Td_list.append(info.get("Td"))
        Tf_list.append(info.get("Tf"))
        Tinf_list.append(info.get("Tinf"))

    print(f"Recompensa total: {episode_reward}")
    print("")

# Para visualização:
SPTq = np.concatenate(SPTq_list, axis=0)
Tq = np.concatenate(Tq_list, axis=0)
SPh = np.concatenate(SPh_list, axis=0)
h = np.concatenate(h_list, axis=0)
Tt = np.concatenate(Tt_list, axis=0)
Ts = np.concatenate(Ts_list, axis=0)
Sr = np.concatenate(Sr_list, axis=0)
Sa = np.concatenate(Sa_list, axis=0)
xq = np.concatenate(xq_list, axis=0)
xf = np.concatenate(xf_list, axis=0)
xs = np.concatenate(xs_list, axis=0)
Fs = np.concatenate(Fs_list, axis=0)
Fd = np.concatenate(Fd_list, axis=0)
Td = np.concatenate(Td_list, axis=0)
Tf = np.concatenate(Tf_list, axis=0)
Tinf = np.concatenate(Tinf_list, axis=0)

# Gráficos:
sns.set_style("darkgrid")
fig, ax = plt.subplots(3, 3, figsize=(20, 17))

ax[0, 0].plot(time_total, Ts, label="Ts", color="royalblue", linestyle="solid")
ax[0, 0].plot(time_total, Tt, label="Tt", color="deepskyblue", linestyle="solid")
# ax[0, 0].set_title("Temperaturas de saída (Ts) e do tanque (Tt)")
ax[0, 0].set_xlabel("Tempo")
ax[0, 0].set_ylabel("Temperatura")
ax[0, 0].legend()

ax[0, 1].plot(time_total, SPTq, label="SPTq", color="purple", linestyle="dashed")
ax[0, 1].plot(time_total, Tq, label="Tq", color="mediumorchid", linestyle="solid")
# ax[0, 1].set_title("Temperatura do boiler (Tq)")
ax[0, 1].set_xlabel("Tempo")
ax[0, 1].set_ylabel("Temperatura")
ax[0, 1].legend()

ax[0, 2].plot(time_actions, iqb_list, label="IQB", color="black", linestyle="solid")
ax[0, 2].plot(time_actions, recompensa_list, label="Recompensa", color="black", linestyle="solid")
# ax[0, 2].set_title("Qualidade do banho (IQB)")
ax[0, 2].set_xlabel("Ações")
ax[0, 2].set_ylabel("Índice final")
ax[0, 2].legend()

ax[1, 0].plot(time_total, SPh, label="SPh", color="darkslategray", linestyle="dashed")
ax[1, 0].plot(time_total, h, label="h", color="teal", linestyle="solid")
# ax[1, 0].set_title("Nível do tanque (h)")
ax[1, 0].set_xlabel("Tempo")
ax[1, 0].set_ylabel("Nível")
ax[1, 0].legend()

ax[1, 1].plot(time_total, xq, label="xq", color="darkmagenta", linestyle="solid")
ax[1, 1].plot(time_total, xf, label="xf", color="deeppink", linestyle="solid")
ax[1, 1].plot(time_total, xs, label="xs", color="palevioletred", linestyle="solid")
# ax[1, 1].set_title("Aberturas das válvulas quente (xq), fria (xf) e de saída (xs)")
ax[1, 1].set_xlabel("Tempo")
ax[1, 1].set_ylabel("Abertura")
ax[1, 1].legend()

ax[1, 2].plot(time_actions, custo_eletrico_list, label="Custo elétrico", color="black", linestyle="solid")
ax[1, 2].plot(time_actions, custo_gas_list, label="Custo do gás", color="gray", linestyle="solid")
ax[1, 2].plot(time_actions, custo_agua_list, label="Custo da água", color="dodgerblue", linestyle="solid")
# ax[1, 2].set_title("Custos do banho")
ax[1, 2].set_xlabel("Ações")
ax[1, 2].set_ylabel("Custos")
ax[1, 2].legend()

ax[2, 0].plot(time_total, Sa, label="Sa", color="skyblue", linestyle="solid")
ax[2, 0].plot(time_total, Sr, label="Sr", color="darkcyan", linestyle="solid")
# ax[2, 0].set_title("Frações da resistência elétrica (Sr) e do aquecimento do boiler (Sa)")
ax[2, 0].set_xlabel("Tempo")
ax[2, 0].set_ylabel("Fração")
ax[2, 0].legend()

ax[2, 1].plot(time_total, Fs, label="Fs", color="slateblue", linestyle="solid")
ax[2, 1].plot(time_total, Fd, label="Fd", color="darkorchid", linestyle="solid")
# ax[2, 1].set_title("Vazões de saída (Fs) e de distúrbio (Fd)")
ax[2, 1].set_xlabel("Tempo")
ax[2, 1].set_ylabel("Vazão")
ax[2, 1].legend()

ax[2, 2].plot(time_total, Tinf, label="Tinf", color="steelblue", linestyle="solid")
ax[2, 2].plot(time_total, Tf, label="Tf", color="slategray", linestyle="solid")
ax[2, 2].plot(time_total, Td, label="Td", color="black", linestyle="solid")
# ax[2, 2].set_title("Temperaturas ambiente (Tinf), da corrente fria (Tf) e de distúrbio (Td)")
ax[2, 2].set_xlabel("Tempo")
ax[2, 2].set_ylabel("Temperatura")
ax[2, 2].legend()
plt.show()

# Reseta o Ray:
ray.shutdown()