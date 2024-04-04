# Audiovisual MNIST

## Dataset

The visual modality is MNIST.

The audio modality is the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).

Each MNIST written digit image is paired with a spoken digit audio of the same label to make up an audiovisual dataset sample.

## Dependency

We need a utility for processing the Free Spoken Digit Dataset: [torch-fsdd](https://github.com/eonu/torch-fsdd/)

```bash
git clone https://github.com/eonu/torch-fsdd.git
cd torch-fsdd; python setup.py install
```

Compulsory packages: pytorch, torchvision, torchaudio, numpy

Optional packages: pandas, matplotlib (not needed if you're not plotting and saving)

## Run

```bash
python avmnist.py
```

If you need to adjust hyperparameters, run `python avmnist.py -h` to examine the settings.