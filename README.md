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
