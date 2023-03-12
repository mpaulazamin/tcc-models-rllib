## Agent PPO - V0

Modelo com malha cascata para Ts com split-range, malha de inventário para o nível do tanque, e controle liga-desliga no boiler.

![chuveiro](https://github.com/mpaulazamin/tcc-rllib/blob/agent_ppo_v1/imagens/chuveiro_controle_t4a.jpg)

### Espaço de ações

- SPTs: 30 a 45 - contínuo
- SPTq: 30 a 70 - contínuo
- xs: 0.01 a 0.99 - contínuo
- split_range: 0 (desligado) e 1 (ligado)

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
- Td: 25
- Tf: 25
- Tinf: 25

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
recompensa = iqb
```

### Resultados

TBD

### Próximos passos

TBD
