## Agent PPO - V1

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
- custo_eletrico: 2 em ambos os treinos
- custo_gas: 1 (steps=50) e 3 com steps=100
- custo_agua: 3 (steps=50) e 4 (steps=100)

### Episódios

- Tempo de cada iteração: 2 minutos
- Tempo total de cada episódio: 14 minutos
- 7 ações em cada episódio
- 50 steps no PPO, totalizando 200000 episódios
- 100 steps no PPO, totalizando 400000 episódios

### Parâmetros do PPO

- Parâmetros default

### Recompensa

Definida como:

```bash
recompensa = iqb
```

### Resultados para steps=50

O sistema consegue atingir IQBs altos (acima de 0.8) a partir de ação 1. Na primeira ação o IQB ainda é baixo por causa do tempo morto do sistema, mas depois ele consegue controlar bem as variáveis sem precisar da malha cascata ou do split-range. 

Na minha opinião, essa é a configuração que mais tem espaço para otimização de custos, porque dessa vez, o agente prefiriu utilizar mais o Sr do que o Sa, e escolheu temperaturas menores para SPTq. Isso acontece pois Sr esquente mais água (já que no tanque há a mistura de corrente quente e fria), então o aquecimento é mais eficiente do que o gás. Entretanto, o objetivo é utilizar mais gás do que energia elétrica, então é necessário incluir os custos na recompensa.

Finalmente, talvez seria interessante rodar o sistema com 100 steps (400000 episódios), pois pelas curvas de loss e entropy no tensorboard, parece que o agente ainda poderia aprender mais. Porém, esse resultado já mostra o potencial do agente.

A figura abaixo ilustra um exemplo de 1 episódio completo com o último checkpoint do agente (step 50):

![image](https://github.com/mpaulazamin/tcc-models-rllib/blob/agent_ppo_v1/imagens/avalia%C3%A7%C3%A3o_agent_ppo_v1.jpg)

### Próximos passos

Tentar incluir os custos na recompensa. Uma forma seria:

```bash
recompensa = iqb / (iqb + custos)
```

onde _custos_ é a soma de todos os custos (elétrico, gás, água). 

Recompensas negativas, como _iqb - custos_, podem fazer com que o agente tente atingir o estado terminal logo (quando o nível do tanque é maior que 100).
Além disso, vi que técnicas de shaping (recompensas entre 0 e 1) geralmente fazem com que o agente aprenda mais rápido.