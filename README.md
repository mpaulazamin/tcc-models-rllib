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

O agente consegue atingir IQBs próximos a 1 (acima de 0.8) a partir da ação 2. Na ação 1, o IQB é baixo, mas isso acontece devido a característica do sistema, com um tempo morto de
aproximadamente 0.8 segundos. O agente prefere selecionar valores altos de SPTs, acima de 42, porém Ts não chega nessas temperaturas. A temperatura que o sistema chega
geralmente é entre 37 a 38 graus, que é a faixa de temperatura que possibilita um IQB alto. Outro ponto interessante é que mesmo quando o agente escolhe como ação o split-range 1 (ligado), o sistema acaba não utilizando o split-range devido ao fato de xq não ultrapassar o valor de 1. Talvez isso aconteça porque para que Ts atinja valores acima de 40 graus, seria necessário refazer o tuning do PID ou aumentar o tempo de cada iteração. Desse modo, xq nunca ultrapassa 1 para que possa ser utilizado a estratégia de split-range. Porém, as temperaturas acima de 40 graus não são do nosso interesse, logo não queremos que o sistema chega a um valor de Ts muito alto devido ao IQB.

### Próximos passos

Rodar o sistema novamente, mas diminuir o valor de SPTs, deixando as ações entre 30 a 40 graus. Seria interessante que o sistema conseguisse chegar ao valor de SPTs selecionado,
porque ao tomar banho, seleciona-se o valor de SPTs para que Ts tenha esse mesmo valor.
