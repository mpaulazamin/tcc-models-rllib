## Instalação

Siga estas instruções: https://github.com/ray-project/ray/tree/master/rllib#installation-and-setup.

Depois, instale o pacote `tensorflow_probability`: https://github.com/tensorflow/probability/releases.

```bash
pip install tensorflow-probability==0.19.0
```

## Integração com Tensorboard

Siga estas instruções: https://stackoverflow.com/questions/45095820/tensorboard-command-not-found.

Execute o seguinte comando:

```bash
pip show tensorflow
```

Entre no local onde o `tensorflow` está instalado:

```bash
cd C:\users\maria\appdata\roaming\python\python38\site-packages
```

Entre no folder do `tensorboard`:

```bash
cd tensorboard
```

Execute o seguinte comando:

```bash
python main.py --logdir "C:\users\maria\ray_results\folder_experiment"
```

## Sanity check

Treinando o sistema com o ambiente customizado [neste script](https://github.com/mpaulazamin/tcc-models-rllib/blob/main/sanity_check.py), obtém-se o resultado abaixo, 
que reproduz o resultado encontrado no [notebook](https://github.com/mpaulazamin/tcc-models-rllib/blob/main/chuveiro_turbinado.ipynb) do professor.

![check](https://github.com/mpaulazamin/tcc-models-rllib/blob/main/imagens/custom_env.jpg)

## TO-DO

- Separar gráficos em ações e estados
- Treinar com diferentes temperaturas ambientes

## Observações

- Entender qual abordagem controla melhor o sistema: o agente altera diretamente os setpoints, ou ele otimiza os setpoins. Para isso, treinar as duas abordagens somente com o IQB como recompensa. De acordo com os modelos treinados, é possível perceber que o agente controla melhor o sistema no papel de otimizador de setpoints;
- O sistema com o agente otimizando o split-range já realiza uma otimização de custos;
- Algumas opções de experimentos considerando os seguintes sistemas: 1) sistema com controle de nível de tanque e controle do boiler; 2) sistema com controle de nível de tanque, controle do boiler e malha cascata; 3) sistema com controle de nível de tanque, controle do boiler, malha cascata e split-range:
  - Fixar os custos e treinar os sistemas 1 e 2 com temperaturas ambiente diferentes (dia frio, ameno e quente). Entender qual terá um melhor controle do sistema.
  - Fixar os custos e treinar os sistemas 2 e 3 com temperaturas ambiente diferentes (dia frio, ameno e quente). Entender se o sistema 2 consegue chegar em custos similares da energia elétrica que o sistema 3. Incluir custo do gás e água na recompensa?
  - Fixar os custos e treinar os sistemas 1, 2 e 3 com temperaturas ambiente diferentes (dia frio, ameno e quente).
  - Fixar os custos e treinar somente o sistema 2 com temperaturas ambiente diferentes (dia frio, ameno e quente). Comparar os resultados com o sistema 2 apenas com o IQB na recompensa, mas também treinado com diferentes temperaturas ambiente.
- Sistema multiagente com custos variando poderia estar nos próximos passos da conclusão.

Atualização: 02/04/2023

- Entender qual é a melhor abordagem para que o agente controle o sistema: otimizando os setpoints ou operando diretamente as válvulas. Utilizar somente IQB como recompensa. Rodar 1 agente de cada com Tinf=25 é o suficiente para tirar essa conclusão, ou preciso rodar com outras temperaturas ambientes?
- A partir da melhor abordagem, entender se o agente consegue otimizar os custos: treinar o sistema com três temperaturas ambientes diferentes (15, 20 e 25), e incluir os custos na recompensa. 
