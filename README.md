## Agent PPO - V8

Modelo com malha de inventário para o nível do tanque e com controle liga-desliga do boiler. Sem malha cascata, sem split-range.

![chuveiro](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v8/imagens/chuveiro_controle_t4a_sem_split.jpg)

### Espaço de ações

- SPTs: 30 a 40 - contínuo
- SPTq: 30 a 70 - contínuo
- xs: 0.01 a 0.99 - contínuo
- Sr: 0 a 10 - discreto (depois divide-se cada valor por 10)

### Espaço de estados

- Ts: 0 a 100
- Tq: 0 a 100
- Tt: 0 a 100
- h: 0 a 10000
- Fs: 0 a 100
- xq: 0 a 1
- xf: 0 a 1
- iqb: 0 a 1
- custo_eletrico: 0 a 1

### Variáveis fixas

- Fd: 0
- Td: 25
- Tf: 25
- Tinf: 25
- custo_eletrico_kwh: 2

### Episódios

- Tempo de cada iteração: 2 minutos
- Tempo total de cada episódio: 14 minutos
- 7 ações em cada episódio
- 50 steps no PPO, totalizando 200000 episódios

### Parâmetros do PPO

- Parâmetros default

### Recompensa

Definida como:

```bash
if Sr == 0:
    reward = 3 * iqb + 1
else:
    reward = 3 * iqb + 0.01 * (1 / (custo_eletrico / custo_eletrico_max))
```

### Resultados

O sistema consegue chegar a IQBs acima de 0.8 sem utilizar a energia elétrica. 

A figura abaixo apresenta o sistema após o agente ser treinado com 50 steps:

![image](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v8/imagens/avalia%C3%A7%C3%A3o_agent_ppo_v8.png)

### Próximos passos

Incluir os custos do gás e da água na recompensa. Depois, treinar com temperaturas ambiente diferentes (dia frio, ameno e quente).
