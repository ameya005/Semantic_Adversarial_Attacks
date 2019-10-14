# Semantic Adversarial Attacks
Code for Semantic Adversarial Attacks as described in the [paper](https://arxiv.org/abs/1904.08489). We use a attribute controlled generative model to enforce a semantic constraint on adversarial examples.



## Setup

Use 

```
git clone --recursive <remote-path> 
```

to pull the repo.

You can install the requirements using 

```
pip install -r ./requirements.txt
```

Path for models: [Link](https://drive.google.com/drive/folders/1-UO_S5WwoPILNhi3f8X04AiJwMcISjOL?usp=sharing)

## Usage

The code has options to reproduce four different attack scenarios:
1. Single Attribute attack with Fader Networks ( Refer [attack_single_attribute.py](attack_single_attribute.py) 
2. Multi Attribute attack with Fader Networks ( Refer [attack_fadernets.py](attack_fadernets.py) 
3. Sequential multi-attribute attack with Fader Networks ( Refer [attack_fadernets_seq.py](attack_fadernets_seq.py) 
4. Multi attribute attack with AttGAN ( Refer [attack_attgan.py](attack_attgan.py) 

These have been implemented in separate files for ease of use.

## Attacking with AttGAN.

1. Download the AttGAN model from the link above.
2. Train a target model using [simple_classifier.py](simple_classifier.py) or [resnet.py](resnet.py).
3. Run [attack_attgan.py](attack_attgan.py).

Usage:

```
python attack_attgan.py -m <path to target model> -f <path to attgan model> -o <path to save images (adversarial of otherwise)> -d <data directory containing images> -a <Path to attribute file (only required for CelebA)> -t <Attack type: optimize over attgan (att) or random sampling of attributes (rn)> -dt <Dataset used: celeba or bdd> -ct <Classifier type: simple or resnet> --proj_flag <Flag to enforce infinity ball projection on attributes> --eps <Radius of infinity ball. Only used in case of proj_flag> --nclasses <no. of classes, default=2> --attk_attributes <Choice of attack attributes. Refer below>
```

Choice of attack attributes for CelebA:

```
[ 
        "Bald",
        "Bangs",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Eyeglasses",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "No_Beard",
        "Pale_Skin",
        "Young"
]
```



