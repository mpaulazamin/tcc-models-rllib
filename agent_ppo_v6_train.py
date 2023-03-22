import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.algorithms.algorithm import Algorithm

import gymnasium as gym
import numpy as np
import random
import pandas as pd
import json
import os
import shutil
import sys
import matplotlib.pyplot as plt
import pprint

from controle_temperatura_saida import simulacao_malha_temperatura
from controle_temperatura_saida import modelagem_sistema
from controle_temperatura_saida import modelo_valvula_saida
from controle_temperatura_saida import calculo_iqb
from controle_temperatura_saida import custo_eletrico_banho
from controle_temperatura_saida import custo_gas_banho
from controle_temperatura_saida import custo_agua_banho

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
        self.custo_agua_m3 = 4

        # Ações - xq, SPTq, xs, Sr:
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(low=0.01, high=0.99, shape=(1,), dtype=np.float32),
                gym.spaces.Box(low=40, high=70, shape=(1,), dtype=np.float32),
                gym.spaces.Box(low=0.3, high=0.7, shape=(1,), dtype=np.float32),
                gym.spaces.Discrete(11, start=0),
            ),
        )

        # Estados - Ts, Tq, Tt, h, Fs, xf, iqb, custo_eletrico:
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([100, 100, 100, 10000, 100, 1, 1, 1]),
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
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.iqb, self.custo_eletrico],
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
        self.Sr = action[3] / 10

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
        self.custo_agua = custo_agua_banho(self.Fs, self.custo_agua_m3, self.tempo_iteracao)

        # Estados - Ts, Tq, Tt, h, Fs, xf, iqb, custo_eletrico, custo_gas, custo_agua:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xf, self.iqb, self.custo_eletrico],
                             dtype=np.float32)

        # Define a recompensa:
        reward = 4 * self.iqb + 2 * (1 - self.Sr)

        # Incrementa tempo inicial:
        self.tempo_inicial = self.tempo_inicial + self.tempo_iteracao

        # Termina o episódio se o tempo for maior que 14 ou se o nível do tanque ultrapassar 100:
        done = False
        if self.tempo_final == 14 or self.h > 100: 
            done = True

        info = {}

        return self.obs, reward, done, info
    
    def render(self):
        pass


# Folder para checkpoints:
checkpoint_root = "C:\\Users\\maria\\ray_ppo_checkpoints\\agent_ppo_v6"
shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)

# Folder para os resultados:
ray_results = f"C:\\Users\zamin\\ray_results"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# Inicializa o Ray:
ray.shutdown()
info = ray.init(ignore_reinit_error=True)

# Define as configurações para o algoritmo PPO:
config = ppo.PPOConfig()
config.environment(env=ShowerEnv)

# Constrói o agente:
agent = config.build()

# Armazena resultados:
results = []
episode_data = []

# Realiza o treinamento:
n_iter = 76
for n in range(1, n_iter):

    # Treina o agente:
    result = agent.train()
    results.append(result)
    
    # Armazena dados do episódio:
    episode = {
        "n": n,
        "episode_reward_min": result["episode_reward_min"],
        "episode_reward_mean": result["episode_reward_mean"], 
        "episode_reward_max": result["episode_reward_max"],  
        "episode_len_mean": result["episode_len_mean"],
    }

    episode_data.append(episode)

    # Salva checkpoint a cada 5 iterações:
    if n % 5 == 0:
        file_name = agent.save(checkpoint_root)
        print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}. Checkpoint saved to {file_name}.')
    else:
        print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}.')


# Salva resultados e plota dados do episódio:
print(results)
df = pd.DataFrame(data=episode_data)
df.to_csv("episode_data_agent_ppo_v6.csv")

policy = agent.get_policy()
model = policy.model
pprint.pprint(model.variables())
pprint.pprint(model.value_function())
print(model.base_model.summary())

# Reseta o Ray:
ray.shutdown()