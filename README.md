## Agent PPO - V10

Modelo com malha de inventário para o nível do tanque, com controle liga-desliga do boiler e com malha cascata. Sem split-range.

![chuveiro](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v9/imagens/chuveiro_controle_t4a_sem_split.jpg)

### Espaço de ações

- SPTs: 30 a 40 - contínuo
- SPTq: 30 a 70 - contínuo
- xs: 0.01 a 0.99 - contínuo
- Sr: 0 a 1 - contínuo

### Espaço de estados

- Ts: 0 a 100
- Tq: 0 a 100
- Tt: 0 a 100
- h: 0 a 10000
- Fs: 0 a 100
- xq: 0 a 1
- xf: 0 a 1
- iqb: 0 a 1

### Variáveis fixas

- Fd: 0
- Td: 15, 20 ou 25
- Tf: 15, 20 ou 25
- Tinf: 15, 20 ou 25
- custo_eletrico_kwh: 2
- custo_gas_kg: 3
- custo_agua_m3 = 4

### Episódios

- Tempo de cada iteração: 2 minutos
- Tempo total de cada episódio: 14 minutos
- 7 ações em cada episódio
- 100 steps no PPO, totalizando 400000 episódios
- 3 modelos, um para cada temperatura ambiente

### Parâmetros do PPO

- Parâmetros default

### Recompensa

Definida como:

```bash
reward = iqb
```

Isso foi feito para comparar com os resultados do agent_ppo_v9, onde os custos foram incluídos na recompensa.

### Resultados

A figuras abaixos mostram o sistema com o agente treinado com 50 steps para diferentes temperaturas ambientes.

![image](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v10/imagens/avalia%C3%A7%C3%A3o_agent_ppo_v10_Tinf25.png)

![image](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v10/imagens/avalia%C3%A7%C3%A3o_agent_ppo_v10_Tinf20.png)

![image](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v10/imagens/avalia%C3%A7%C3%A3o_agent_ppo_v10_Tinf15.png)

### Próximos passos

- Treinar os agentes v9 e v10 com a energia elétrica mais barata.
