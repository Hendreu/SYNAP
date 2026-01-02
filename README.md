# SYNAP

**SYNAP** (SYstem for Neural Array Processing) é uma rede neural implementada **do zero**, utilizando apenas Python e bibliotecas matemáticas básicas, sem qualquer framework de machine learning.

O projeto tem como foco **entendimento profundo** de como redes neurais funcionam internamente — da matemática ao fluxo de execução.

---

## 📌 Visão geral

O modelo foi desenvolvido para classificar **dígitos manuscritos** do dataset MNIST.

Cada amostra é uma imagem **28×28 pixels**, onde:
- cada pixel é convertido em uma entrada numérica
- a rede aprende a mapear esses valores para um dígito de 0 a 9

A arquitetura utilizada é um **MLP (Multi-Layer Perceptron)** totalmente implementado manualmente.

---

## 🧠 O que foi implementado

- Forward propagation
- Backpropagation
- Gradient Descent
- ReLU
- Softmax
- One-hot encoding
- Cálculo explícito de erro e derivadas
- Inferência e visualização de previsões

Tudo isso escrito **sem TensorFlow, PyTorch ou qualquer framework de ML**.

---

## 🧮 Tecnologias utilizadas

- Python
- NumPy
- Pandas
- Matplotlib

---

## ☁️ Ambiente de execução

O projeto foi inicialmente desenvolvido e treinado no **Kaggle**, utilizando o ambiente em nuvem como forma de praticar conceitos de **cloud computing**, execução remota e gerenciamento de recursos.

Posteriormente, o código foi **refatorado para rodar localmente**, com organização de arquitetura, paths e fluxo de execução, garantindo consistência entre ambientes cloud e local.

📦 Dataset

Este projeto utiliza o MNIST dataset em formato CSV.

O arquivo original de treino ultrapassa o limite de 104MB imposto pelo GitHub, portanto o dataset está disponibilizado compactado em um arquivo .rar.

