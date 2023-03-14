## Agent PPO - V3

Modelo com malha cascata para Ts com split-range, malha de inventário para o nível do tanque, e controle liga-desliga no boiler.

![chuveiro](https://github.com/mpaulazamin/tcc-rllib/blob/agent_ppo_v1/imagens/chuveiro_controle_t4a.jpg)

### Espaço de ações

- SPTs: 30 a 40 - contínuo
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

O agente continua conseguindo atingir IQBs próximos a 1 depois dos primeiros 2 minutos (primeira ação). Esse atraso ainda ocorre por causa do tempo morto. Outro ponto para melhorar é que talvez Tq deva iniciar em 70, pois o agente geralmente seleciona temperaturas altas para o primeiro SPTq, mas como a válvula de corrente quente fica totalmente aberta, o sistema não consegue alcançar Tq acima de 55 graus. 

Além disso, às vezes, o agente ainda seleciona o valor 1 para o split-range (ligado), mas xq não ultrapassa o valor 1, então não é utilizado o split-range. Isso parece não atrapalhar muito o agente, mas talvez seja outro ponto para melhorar.

Se começarmos Tq em 70 graus, talvez nem seja necessário otimizar os custos nesse caso, porque o agente usa o split-range majoritariamente no começo do episódio. 

A figura abaixo ilustra um exemplo de 1 episódio completo com o último checkpoint do agente (step 50):

![image](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v3/imagens/avalia%C3%A7%C3%A3o_agente_ppo_v3.jpg)

### Próximos passos

Tentar incluir os custos no sistema, mas não acho que fará muita diferença.
