## Agent PPO - V9

Modelo com malha de inventário para o nível do tanque, com controle liga-desliga do boiler e com malha cascata. Sem split-range.

![chuveiro](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v9/imagens/chuveiro_controle_t4a_sem_split.jpg)

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
- custo_gas: 0 a 1
- custo_agua: 0 a 1

### Variáveis fixas

- Fd: 0
- Td: 25
- Tf: 25
- Tinf: 25
- custo_eletrico_kwh: 2
- custo_gas_kg: 3

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
if custo_eletrico == 0 and custo_gas != 0:
    reward = 3 * iqb + 4 + 0.01 * (1 / (custo_gas / custo_gas_max)) + 0.01 * (1 / (custo_agua / custo_agua_max))
    
if custo_eletrico != 0 and custo_gas == 0:
    reward = 3 * iqb + 0.05 * (1 / (custo_eletrico / custo_eletrico_max)) + 0.01 * (1 / (custo_agua / custo_agua_max))
    
if custo_eletrico == 0 and custo_gas == 0:
    reward = 3 * iqb + 0.01 * (1 / (custo_agua / custo_agua_max))
    
if custo_eletrico != 0 and custo_gas != 0:
    reward = 3 * iqb + 0.05 * (1 / (custo_eletrico / custo_eletrico_max)) + 0.01 * (1 / (custo_gas / custo_gas_max)) + 0.01 * (1 / (custo_agua / custo_agua_max))
```

### Resultados

TBD

### Próximos passos

TBD
