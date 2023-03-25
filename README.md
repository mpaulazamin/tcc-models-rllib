## Agent PPO - V9

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
- custo_agua_m3 = 4

### Episódios

- Tempo de cada iteração: 2 minutos
- Tempo total de cada episódio: 14 minutos
- 7 ações em cada episódio
- 50 steps no PPO, totalizando 200000 episódios

### Parâmetros do PPO

- Parâmetros default

### Recompensa

Em um primeiro momento, ela foi definida como:

```bash
if custo_eletrico == 0 and custo_gas != 0:
    reward = 3 * iqb + 4 + 0.01 * (1 / (custo_gas / custo_gas_max))
    
if custo_eletrico != 0 and custo_gas == 0:
    reward = 3 * iqb + 0.05 * (1 / (custo_eletrico / custo_eletrico_max))
    
if custo_eletrico == 0 and custo_gas == 0:
    reward = 3 * iqb
    
if custo_eletrico != 0 and custo_gas != 0:
    reward = 3 * iqb + 0.05 * (1 / (custo_eletrico / custo_eletrico_max)) + 0.01 * (1 / (custo_gas / custo_gas_max))
```

Porém, isso não funciona, pois o custo do gás varia muito a escala, e é muito difícil de controlar isso. Por exemplo, se Sa for utilizado somente 1 vez durante os 2 minutos inteiros, o valor `1 / (custo_gas / custo_gas_max)` será alto, na casa dos 30. Isso aumenta a recompensa, independendemente se o IQB é bom ou ruim. 

Isso pode ser observado [nesse notebook](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v9/reward.ipynb).

Então, decidi utilizar:

```bash
reward = 5 * iqb - 2 * custo_eletrico - custo_gas - custo_agua
```

### Resultados

TBD

### Próximos passos

TBD
