## Agent PPO - V3

Modelo com malha de inventário para o nível do tanque e com controle liga-desliga do boiler. Sem malha cascata, sem split-range.

![chuveiro](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v1/imagens/chuveiro_controle_h.jpg)

### Espaço de ações

- xq: 0.01 a 0.99 - contínuo
- SPTq: 30 a 70 - contínuo
- xs: 0.3 a 0.7 - contínuo
- Sr: 0 a 1 - contínuo

### Espaço de estados

- Ts: 0 a 100
- Tq: 0 a 100
- Tt: 0 a 100
- h: 0 a 10000
- Fs: 0 a 100
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
- 40 steps no PPO, totalizando 160000 episódios

### Parâmetros do PPO

- Parâmetros default

### Recompensa

Definida como:

```bash
recompensa = iqb / (iqb + custos)
```

onde _custos_ é a soma de todos os custos (elétrico, gás, água).

### Resultados

O sistema chega a recompensas próximas de 1, entretanto o IQB fica em torno de 5 ou 6. Isso porque os custos em temperaturas menores são menores, e a razão da recompensa quase se iguala ou fica maior do que a recompensa para temperaturas altas. Pode-se concluir que essa não é uma boa função de recompensa. Por isso, parei o treinamento antes do tempo.

A figura abaixo ilustra um exemplo de 1 episódio completo com o último checkpoint do agente (step 40):

![image](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v3/imagens/avalia%C3%A7%C3%A3o_agent_ppo_v3.jpg)

### Próximos passos

Testar as seguintes recompensas:

```bash
recompensa = iqb, se iqb < 0.6
recompensa = iqb - (custos / 10), se iqb >= 0.6
```

ou

```bash
recompensa = iqb - custos
se recompensa < 0: recompensa = 0
```