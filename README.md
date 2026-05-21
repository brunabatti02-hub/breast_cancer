# Breast Cancer Experiments

Repositório de experimentos para classificação de imagens médicas relacionadas ao câncer de mama, com foco em:

- classificação histopatológica no dataset `BreaKHis`;
- classificação mamográfica no dataset `INBreast`;
- transferência de aprendizado entre domínios de histologia e mamografia;
- comparação entre diferentes arquiteturas profundas, incluindo `SEConformer`, `HFTNet` e `HistoDX`.

## Objetivo

Este projeto organiza experimentos de aprendizado profundo voltados à análise de câncer de mama em dois domínios distintos:

- `Histologia`: imagens microscópicas de tecido tumoral mamário;
- `Mamografia`: imagens de mamografia digital com rótulos clínicos baseados em `BI-RADS`.

A proposta central é investigar o desempenho de modelos no domínio de origem e avaliar, em especial, a viabilidade de reutilização de representações aprendidas em cenários de transferência de aprendizado entre domínios médicos visualmente distintos.

## Datasets Utilizados

### 1. BreaKHis

O `BreaKHis` é um dataset de imagens histopatológicas de câncer de mama amplamente utilizado em tarefas de classificação.

Principais características:

- `7.909` imagens microscópicas;
- `82` pacientes;
- `2.480` imagens benignas;
- `5.429` imagens malignas;
- fatores de ampliação: `40X`, `100X`, `200X` e `400X`;
- resolução original de `700 x 460` pixels;
- imagens `RGB` com `3 canais`;
- profundidade de `8 bits` por canal;
- formato `PNG`.

Distribuição por ampliação:

- `40X`: `1.995` imagens;
- `100X`: `2.081` imagens;
- `200X`: `2.013` imagens;
- `400X`: `1.820` imagens.

Estrutura local no projeto:

- `BrestCancer Datasets/BreaKHis/`

Link de download:

- `https://www.kaggle.com/datasets/ambarish/breakhis`

Referência acadêmica:

- Spanhol, F. A., Oliveira, L. S., Petitjean, C., Heutte, L. "A Dataset for Breast Cancer Histopathological Image Classification."

### 2. INBreast

O `INBreast` é um conjunto de mamografias digitais de campo total, utilizado neste projeto para classificação no domínio mamográfico.

Principais características:

- `115` casos;
- `410` imagens;
- `90` casos com `4 imagens por exame`;
- `25` casos de pacientes submetidas à mastectomia, com `2 imagens por exame`;
- presença de anotações especializadas em `XML`;
- inclusão de diferentes achados, como:
  - massas;
  - calcificações;
  - assimetrias;
  - distorções.

No projeto, o dataset é utilizado em formulações associadas à escala `BI-RADS`, com foco em classificação multiclasse.

Estrutura local relacionada:

- `INbreast_Folds.csv`
- rotinas de leitura e preparação dentro dos módulos dos modelos

Link de download:

- `https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset`

Referência acadêmica:

- Moreira, I. C., Amaral, I., Domingues, I., Cardoso, A., Cardoso, M. J., Cardoso, J. S. "INbreast: Toward a Full-field Digital Mammographic Database."

## Modelos e Experimentos

Os principais experimentos do repositório estão organizados em torno dos seguintes modelos:

- `SEConformer`: arquitetura híbrida baseada em convolução, recalibração de canais e atenção;
- `SEConformer_TL`: configuração experimental voltada à transferência de aprendizado para o domínio mamográfico;
- `HFTNET`: arquitetura de fusão heterogênea de múltiplos extratores profundos;
- `HISTODX`: baseline baseada em backbone moderno para classificação histopatológica;
- `HISTODX_TORCH`: variante experimental em PyTorch;
- `SqueezeNet_SVM`: experimento adicional com abordagem híbrida.

## Organização do Repositório

### Diretórios principais

- `SEConformer/`: implementação modular do modelo SEConformer;
- `SEConformer_TL/`: rotinas de transferência de aprendizado com pesos previamente treinados;
- `HFTNET/`: implementação do HFTNet;
- `HISTODX/`: implementação da baseline HistoDX;
- `HISTODX_TORCH/`: variante da baseline em PyTorch;
- `SqueezeNet_SVM/`: experimentos complementares;
- `results/`: métricas, históricos, modelos salvos e gráficos gerados;
- `BrestCancer Datasets/`: armazenamento local dos datasets;
- `.ipynb_checkpoints/`: checkpoints automáticos de notebooks.

### Arquivos relevantes na raiz

- `SEConformer.ipynb`: experimento do SEConformer;
- `SEConformer_INBreast_DICOM.ipynb`: experimento no domínio mamográfico;
- `SEConformer_TL_INBreast_DICOM.ipynb`: transferência de aprendizado para INBreast;
- `HFT-Net completo.ipynb`: notebook do HFTNet;
- `HistoDX.ipynb`: notebook do HistoDX;
- `Folds.csv`, `Folds_HFT.csv`, `INbreast_Folds.csv`: divisões e particionamentos experimentais;
- `seconformer_comparacao.tex`: comparação textual dos resultados do SEConformer;
- `secao_transferencia_fapesp.tex`: seção em LaTeX sobre transferência entre domínios;
- `secao_baselines_fapesp.tex`: seção em LaTeX com tabela resumida de resultados.

## Estrutura dos Módulos

Nas implementações modulares, a organização tende a seguir o padrão:

- `config.py`: hiperparâmetros e configurações gerais;
- `data.py`: leitura, parsing, transformações e particionamento dos dados;
- `model.py`: definição das arquiteturas;
- `train.py`: treinamento e, em alguns casos, rotinas de transferência;
- `eval.py`: avaliação e métricas;
- `io_utils.py`: persistência de resultados e organização de diretórios de execução.

## Resultados Gerados

As execuções costumam salvar artefatos em `results/`, incluindo:

- `final_metrics.json`;
- `history.csv`;
- pesos dos modelos treinados;
- curvas de treinamento;
- matrizes de confusão;
- curvas ROC e gráficos auxiliares.

Exemplos já presentes no repositório:

- `results/SEConformer/...`
- `results/SEConformer_TL/...`

## Fluxo Geral de Uso

### 1. Baixar os datasets

Baixe manualmente os datasets a partir dos links:

- `BreaKHis`: `https://www.kaggle.com/datasets/ambarish/breakhis`
- `INBreast`: `https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset`

### 2. Organizar os arquivos localmente

Mantenha os dados em caminhos compatíveis com o projeto, especialmente dentro de:

- `BrestCancer Datasets/`

Se necessário, ajuste os caminhos diretamente em:

- notebooks;
- `config.py`;
- scripts de treinamento.

### 3. Executar os experimentos

O repositório contém tanto notebooks quanto implementações modulares. Em geral, os experimentos podem ser conduzidos de duas formas:

- execução pelos notebooks da raiz;
- importação das rotinas presentes nas pastas dos modelos.

Exemplos de pontos de entrada:

- `SEConformer.ipynb`
- `SEConformer_INBreast_DICOM.ipynb`
- `SEConformer_TL_INBreast_DICOM.ipynb`
- `HFT-Net completo.ipynb`
- `HistoDX.ipynb`

## Observações Importantes

- O nome da pasta `BrestCancer Datasets` foi mantido conforme está no workspace atual.
- Os resultados de transferência entre domínios devem ser interpretados com cautela, pois histologia e mamografia apresentam forte diferença de escala, textura, semântica visual e estrutura de rótulos.
- Parte dos relatórios em LaTeX do projeto já foi organizada para uso em documentação acadêmica e relatórios institucionais.

## Referências de Dados

- BreaKHis: `https://www.kaggle.com/datasets/ambarish/breakhis`
- INBreast: `https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset`

## Status do Projeto

O repositório reúne código experimental, notebooks, resultados intermediários e textos de apoio em LaTeX. Ele funciona como base de pesquisa para comparação entre arquiteturas e análise de transferência de aprendizado aplicada ao diagnóstico auxiliado por imagem em câncer de mama.
