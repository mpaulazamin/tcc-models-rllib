import numpy as np
import random
import matplotlib.pyplot as plt

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


class ShowerEnv():
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

        # Potência da resistência elétrica em kW:
        self.potencia_eletrica = 5.5

        # Potência do aquecedor boiler em kcal/h:
        self.potencia_aquecedor = 29000

        # Custo da energia elétrica em kWh, do kg do gás, e do m3 da água:
        self.custo_eletrico_kwh = 2
        self.custo_gas_kg = 1
        self.custo_agua_m3 = 3

        # Fração da resistência elétrica:
        self.Sr = 0

    def reset(self):

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

        # Estados - Ts, Tq, Tt, h, Fs, xq, xf, iqb:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xq, self.xf, self.iqb],
                             dtype=np.float32)
        
        return self.obs

    def step(self, action):

        # Tempo de cada iteração:
        self.tempo_final = self.tempo_inicial + self.tempo_iteracao

        # Setpoint da temperatura de saída:
        self.SPTs = action[0][0]

        # Setpoint da temperatura do boiler:
        self.SPTq = action[1][0]

        # Abertura da válvula de saída:
        self.xs = action[2][0]

        # Split-range:
        self.split_range = action[3]

        # Variáveis para simulação - tempo, SPTq, SPh, xq, xs,Tf, Td, Tinf, Fd, Sr:
        self.UT = np.array(
            [   
                [self.tempo_inicial, self.SPTq, self.SPh, self.SPTs, self.xs, self.Tf, self.Td, self.Tinf, self.Fd, self.Sr],
                [self.tempo_final, self.SPTq, self.SPh, self.SPTs, self.xs, self.Tf, self.Td, self.Tinf, self.Fd, self.Sr]
            ]
        )
        print(self.UT)

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
        self.Tq_total = self.YY[:,0]

        # Valor final do nível do tanque:
        self.h = self.YY[:,1][-1]
        self.h_total = self.YY[:,1]

        # Valor final da temperatura do tanque:
        self.Tt = self.YY[:,2][-1]
        self.Tt_total = self.YY[:,2]

        # Valor final da temperatura de saída:
        self.Ts = self.YY[:,3][-1]
        self.Ts_total = self.YY[:,3]

        # Fração do aquecedor do boiler utilizada durante a iteração:
        self.Sa_total =  self.UU[:,0]

        # Fração da resistência elétrica utilizada durante a iteração:
        self.Sr_total = self.UU[:,8]

        # Valor final da abertura de corrente fria:
        self.xf = self.UU[:,1][-1]
        self.xf_total = self.UU[:,1]

        # Valor final da abertura de corrente quente:
        self.xq = self.UU[:,2][-1]
        self.xq_total = self.UU[:,2]

        # Valor final da abertura da válvula de saída:
        self.xs = self.UU[:,3][-1]

        # Valor final da vazão de saída:
        self.Fs = modelo_valvula_saida(self.xs)

        # Cálculo do índice de qualidade do banho:
        self.iqb = calculo_iqb(self.Ts, self.Fs)

        # Cálculo do custo elétrico do banho:
        self.custo_eletrico = custo_eletrico_banho(self.Sr_total, self.potencia_eletrica, self.custo_eletrico_kwh, self.dt)

        # Cálculo do custo de gás do banho:
        self.custo_gas = custo_gas_banho(self.Sa_total, self.potencia_aquecedor, self.custo_gas_kg, self.dt)

        # Cálculo do custo da água:
        self.custo_agua = custo_agua(self.Fs, self.custo_agua_m3, self.tempo_iteracao)

        # Estados - Ts, Tq, Tt, h, Fs, xq, xf, iqb, custo_eletrico, custo_gas, custo_agua:
        self.obs = np.array([self.Ts, self.Tq, self.Tt, self.h, self.Fs, self.xq, self.xf, self.iqb],
                             dtype=np.float32)

        # Define a recompensa:
        reward = self.iqb

        # Incrementa tempo inicial:
        self.tempo_inicial = self.tempo_inicial + self.tempo_iteracao

        # Termina o episódio se o tempo for maior que 14 ou se o nível do tanque ultrapassar 100:
        done = False
        if self.tempo_final == 14 or self.h > 100: 
            done = True

        info = {"Ts": self.Ts_total,
                "h": self.h_total,
                "Sa": self.Sa_total,
                "Sr": self.Sr_total,
                "xq": self.xq_total,
                "xf": self.xf_total}

        return self.obs, reward, done, info
    
    def render(self):
        pass

env = ShowerEnv()

# SPTs, SPTq, xs, split_range
actions = [(np.array([25]), np.array([55]), np.array([0.5]), 1),
           (np.array([30]), np.array([55]), np.array([0.5]), 1),
           (np.array([35]), np.array([55]), np.array([0.5]), 1),
           (np.array([35]), np.array([60]), np.array([0.7]), 1),
           (np.array([35]), np.array([55]), np.array([0.7]), 1),
           (np.array([31]), np.array([55]), np.array([0.8]), 1),
           (np.array([31]), np.array([55]), np.array([0.8]), 1),
           (np.array([31]), np.array([55]), np.array([0.8]), 1),]

Ts_list = []
h_list = []
Sa_list = []
Sr_list = []
xq_list = []
xf_list = []
time = np.arange(start=0, stop=16 + 0.09, step=0.01, dtype="float")

obs = env.reset()
for action in actions:
    print(action)
    obs, reward, done, info = env.step(action)
    Ts_list.append(info.get("Ts"))
    h_list.append(info.get("h"))
    Sa_list.append(info.get("Sa"))
    Sr_list.append(info.get("Sr"))
    xq_list.append(info.get("xq"))
    xf_list.append(info.get("xf"))
    print(obs)
    print("")

Ts = np.concatenate(Ts_list, axis=0)
h = np.concatenate(h_list, axis=0)
Sa = np.concatenate(Sa_list, axis=0)
Sr = np.concatenate(Sr_list, axis=0)
xq = np.concatenate(xq_list, axis=0)
xf = np.concatenate(xf_list, axis=0)

plt.figure(figsize=(15, 10))
plt.subplot(2,2,1)
plt.plot(time, Ts, label="Ts")
plt.legend()
plt.subplot(2,2,2)
plt.plot(time, h, label="h")
plt.legend()
plt.subplot(2,2,3)
plt.plot(time, Sa, label="Sa")
plt.plot(time, Sr, label="Sr")
plt.legend()
plt.subplot(2,2,4)
plt.plot(time, xq, label="xq")
plt.plot(time, xf, label="xf")
plt.legend()
plt.show()
