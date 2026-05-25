# Breast Cancer Experiments

Repositório de experimentos em classificação de imagens médicas relacionadas ao câncer de mama, com foco em três arquiteturas:

- `SEConformer`
- `HFTNet`
- `HistoDX`

O projeto reúne baselines em histopatologia e mamografia, além de um cenário de transferência de aprendizado entre `BreaKHis` e `INBreast`.

## Objetivo

Este repositório foi organizado para comparar diferentes estratégias de aprendizado profundo em cenários de:

- classificação histopatológica no `BreaKHis`;
- classificação histopatológica nos datasets `BACH` e `BRACS`;
- classificação mamográfica no `INBreast`;
- transferência de aprendizado de histologia para mamografia.

Mais do que reproduzir um único paper, a proposta aqui é manter uma base experimental comparável, com código modular, notebooks e resultados salvos em disco.

## Estrutura Atual do Repositório

```text
breast_cancer/
|- HFTNET/
|- HISTODX/
|- SEConformer/
|- results/
`- README.md
```

### Pastas principais

- `SEConformer/`: implementação modular do SEConformer, notebooks e arquivos auxiliares.
- `HFTNET/`: implementação do HFTNet, notebooks e utilitários de treino.
- `HISTODX/`: baseline HistoDX em PyTorch.
- `results/`: métricas finais e artefatos de execução.

### Arquivos de resultado já presentes

- `results/summary_metrics.csv`: tabela consolidada de métricas.
- `results/*/final_metrics.json`: métricas finais por experimento.

### Artigos incluídos no repositório

- `SEConformer/Advancing_Breast_Cancer_Detection_SE-Conformer_Framework_for_Malignancy_Detection_in_Histopathology_Images.pdf`
- `HFTNET/HFT-Net_Hybrid_Fusion_Transformer_Network_for_Mult.pdf`
- `HISTODX/HistoDX_Revolutionizing_Breast_Cancer_Diagnosis_Through_Advanced_Imaging_Techniques.pdf`

## Datasets Considerados

Os dados locais usados nos experimentos deste repositório estão organizados em:

```text
C:\Users\franc\OneDrive\Desktop\Breast Cancer\BrestCancer Datasets
```

Observação importante:

- alguns arquivos do projeto usam caminhos absolutos apontando para essa pasta;
- em especial, `HFTNET/config.py`, `SEConformer/breakhis_binary_folds.csv` e `SEConformer/inbreast_birads_folds.csv` dependem dessa organização;
- se os datasets forem movidos para outro local, esses caminhos precisam ser ajustados.

### BreaKHis

Dataset histopatológico amplamente usado em classificação de câncer de mama.

- `7.909` imagens
- `82` pacientes
- `2.480` imagens benignas e `5.429` malignas
- ampliações de `40X`, `100X`, `200X` e `400X`
- arquivo auxiliar `Folds.csv` com colunas `fold`, `mag`, `grp` e `filename`

No código deste repositório ele aparece em dois formatos:

- binário (`benign` vs `malignant`)
- multiclasse por subtipo histológico, dependendo do modelo

Estrutura local observada:

```text
BreaKHis/
|- BreaKHis_v1/
|  `- BreaKHis_v1/
|     `- histology_slides/
|        `- breast/
|           |- benign/
|           `- malignant/
`- Folds.csv
```

### INBreast

Dataset de mamografias digitais com anotações clínicas e rótulos derivados de `BI-RADS`.

- `410` imagens
- `115` casos
- dados originalmente em `DICOM`
- `410` linhas em `INbreast.csv`
- metadados auxiliares em `INbreast.xls`, `README.txt` e `inbreast.pdf`

Nos experimentos daqui, as imagens são convertidas para 3 canais e redimensionadas para entrada de rede.

Estrutura local observada:

```text
INBreast/
`- INbreast/
   |- AllDICOMs/
   |- INbreast.csv
   |- INbreast.xls
   |- README.txt
   `- inbreast.pdf
```

Observação prática:

- o arquivo `SEConformer/inbreast_birads_folds.csv` já referencia diretamente os arquivos `.dcm` dentro de `AllDICOMs`.

### BACH

Baseline adicional de histopatologia para medir generalização entre arquiteturas em imagens microscópicas de tecido mamário.

- `400` imagens de treino `.tif`
- `100` imagens por classe em `Normal`, `Benign`, `InSitu` e `Invasive`
- `100` imagens no conjunto de teste em `ICIAR2018_BACH_Challenge_TestDataset/Photos`
- arquivo `microscopy_ground_truth.csv` disponível no pacote de treino

Estrutura local observada:

```text
BACH/
|- ICIAR2018_BACH_Challenge/
|  `- ICIAR2018_BACH_Challenge/
|     `- Photos/
|        |- Benign/
|        |- InSitu/
|        |- Invasive/
|        |- Normal/
|        `- microscopy_ground_truth.csv
`- ICIAR2018_BACH_Challenge_TestDataset/
   `- ICIAR2018_BACH_Challenge_TestDataset/
      |- Photos/
      `- WSI/
```

### BRACS

Baseline adicional em histopatologia com divisão pronta em `train`, `val` e `test`.

- `7` classes: `0_N`, `1_PB`, `2_UDH`, `3_FEA`, `4_ADH`, `5_DCIS` e `6_IC`
- `3.655` imagens em `train`
- `311` imagens em `val`
- `570` imagens em `test`

Estrutura local observada:

```text
BRACS/
|- train/
|- val/
`- test/
```

Observação prática:

- os scripts de `BACH` e `BRACS` importam o módulo auxiliar `histology_datasets`, que não está versionado no repositório.

### Links úteis dos dados

- `BreaKHis`: <https://www.kaggle.com/datasets/ambarish/breakhis>
- `INBreast`: <https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset>

## Modelos Implementados

### SEConformer

Implementação própria com blocos convolucionais residuais com `Squeeze-and-Excitation`, seguidos de um bloco de atenção do tipo conformer simplificado.

### HFTNet

Implementação híbrida baseada em fusão de múltiplos backbones via `timm`, incluindo:

- `DenseNet201`
- `Xception`
- `ViT`
- `DeiT`
- `Swin Transformer`

### HistoDX

Baseline baseada em `EfficientNetV2-S` via `torchvision`, usada como referência forte para tarefas binárias e multiclasses.

## Dependências Principais

As implementações usam principalmente:

- `torch`
- `torchvision`
- `timm`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `Pillow`
- `pydicom` para `INBreast`

Observação importante:

- alguns scripts de treino para `BACH` e `BRACS` importam um módulo auxiliar chamado `histology_datasets`, que não está versionado neste repositório. Os notebooks e os resultados já gerados continuam úteis, mas a reprodução exata desses cenários pode exigir esse módulo externo.

## Notebooks

Os notebooks principais estão organizados dentro de cada pasta de modelo, por exemplo:

- `SEConformer/notebooks/`
- `HFTNET/notebooks/`
- `HISTODX/notebooks/`

Eles funcionam como ponto de entrada mais prático para reproduzir os experimentos já executados no projeto.

## Resultados Consolidados

Os resultados abaixo foram extraídos de `results/summary_metrics.csv`.

Observação:

- em cenários multiclasses, `precision`, `recall` e `F1` podem corresponder a médias macro, dependendo da implementação de cada modelo;
- `specificity` aparece apenas quando foi calculada e salva no experimento correspondente.

| Modelo | Cenário | Accuracy | Precision | Recall | F1 | AUC | Specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HFTNet | BACH | 83.53 | 88.74 | 82.16 | 85.32 | 91.27 | - |
| HFTNet | BRACS | 87.23 | 91.86 | 86.14 | 88.91 | 93.98 | - |
| HFTNet | BreaKHis | 91.62 | 96.35 | 90.85 | 93.52 | 96.48 | - |
| HistoDX | BreaKHis | 93.94 | 97.28 | 92.95 | 95.07 | 97.12 | - |
| SEConformer | BACH | 91.12 | 95.68 | 89.61 | 92.55 | 96.54 | 95.12 |
| SEConformer | BRACS | 86.21 | 90.84 | 84.73 | 87.68 | 93.15 | 91.86 |
| SEConformer | BreaKHis | 92.79 | 98.57 | 90.58 | 94.41 | 98.69 | 97.30 |
| SEConformer | Transferência BreaKHis -> INBreast | 62.32 | 61.80 | 60.45 | 61.12 | 74.83 | - |

## Discussão dos Resultados

Os resultados mostram três comportamentos bem claros:

1. No `BreaKHis`, todos os modelos tiveram desempenho forte, com destaque para `HistoDX` em `accuracy` (`93.94%`) e para `SEConformer` em `AUC` (`98.69%`) e `precision` (`98.57%`).
2. Em `BACH` e `BRACS`, o desempenho continua bom, mas já aparece maior sensibilidade à arquitetura e ao protocolo de treino. O `SEConformer` foi melhor em `BACH`, enquanto o `HFTNet` ficou ligeiramente à frente em `BRACS`.
3. A transferência de `BreaKHis` para `INBreast` foi, de longe, o cenário mais difícil. A `accuracy` caiu para `62.32%` e a `AUC` para `74.83%`, o que indica que as representações aprendidas em histologia não se adaptam automaticamente ao domínio mamográfico.

Em termos práticos, o repositório sugere que:

- baselines treinadas e avaliadas no mesmo domínio visual conseguem resultados competitivos;
- a troca de dataset já altera bastante a hierarquia entre modelos;
- transferência entre domínios médicos muito distintos exige cuidado extra e não deve ser tratada como adaptação simples.

## Por Que Algumas Métricas Ficam Abaixo dos Papers?

Essa diferença é esperada e, neste projeto, há várias razões técnicas plausíveis para isso.

### 1. As implementações daqui são reimplementações experimentais

Os nomes dos modelos seguem os artigos, mas o código do repositório não é necessariamente uma reprodução exata da arquitetura, do pipeline de pré-processamento ou da rotina de otimização descrita nos papers.

Exemplos:

- o `SEConformer` daqui é uma implementação enxuta e própria;
- o `HFTNet` foi montado com backbones de `timm` e uma etapa de fusão customizada;
- o `HistoDX` aqui funciona como baseline baseada em `EfficientNetV2-S`.

Isso já basta para deslocar as métricas para cima ou para baixo em relação aos artigos originais.

### 2. O protocolo experimental não é idêntico ao dos artigos

Neste repositório coexistem:

- `holdout` aleatório;
- `StratifiedKFold`;
- separações `train/val/test` específicas para alguns datasets.

Papers frequentemente reportam:

- média de várias rodadas;
- validação cruzada com protocolo fixo;
- divisão por paciente;
- seleção criteriosa de folds;
- ensemble ou pós-processamento.

Quando o protocolo muda, a comparação direta deixa de ser justa.

### 3. Os datasets usados aqui são pequenos e sensíveis à partição

Isso é especialmente importante em:

- `INBreast`, que é pequeno para treino profundo;
- `BACH` e `BRACS`, onde pequenas diferenças de split afetam bastante as métricas;
- `BreaKHis`, que pode variar de dificuldade conforme o recorte usado.

Em datasets pequenos, alguns pontos percentuais de diferença podem vir apenas da amostragem.

### 4. O pré-processamento reduz informação importante

No caso do `INBreast`, as imagens:

- são originalmente `DICOM`;
- passam por normalização;
- são convertidas para 3 canais;
- são redimensionadas para `224 x 224`.

Esse passo simplifica o pipeline, mas pode remover detalhes finos de lesões, bordas e textura mamográfica que artigos mais fortes preservam com imagens em resolução maior, recortes por ROI ou pré-processamento clínico mais cuidadoso.

### 5. Transferência entre histologia e mamografia tem grande gap de domínio

`BreaKHis` e `INBreast` diferem em quase tudo:

- escala espacial;
- textura;
- contraste;
- semântica visual;
- distribuição de classes;
- tipo de rótulo.

Além disso, no caso de transferência do `SEConformer`, parte do backbone pode ser congelada, o que reduz a capacidade de adaptação ao novo domínio. Então é natural que esse experimento fique bem abaixo dos números vistos em cenários intra-domínio.

### 6. Treino, seleção de checkpoint e tuning são mais simples aqui

Nos scripts atuais, a seleção do melhor modelo e os hiperparâmetros são razoavelmente diretos. Em muitos papers, os autores usam:

- tuning mais agressivo;
- políticas de learning rate mais refinadas;
- augmentations específicas do domínio;
- regularização calibrada por dataset;
- treino repetido com escolha da melhor execução.

Sem esse nível de ajuste, a tendência é ficar alguns pontos abaixo dos resultados publicados.

## Leitura Correta dos Resultados

Os números deste repositório devem ser interpretados como:

- uma comparação interna entre implementações sob um pipeline consistente;
- uma base experimental útil para análise acadêmica;
- um ponto de partida para melhorias de protocolo e reprodução mais fiel dos artigos.

Eles não devem ser apresentados como reprodução oficial ou fiel do estado da arte dos papers anexados sem deixar essa ressalva explícita.

## Próximos Passos Recomendados

Se o objetivo for aproximar mais os resultados dos artigos, o passo mais importante é reproduzir exatamente o protocolo de split de cada paper.

## Resumo

O repositório está consistente como base de experimentação comparativa. Hoje, os melhores resultados aparecem nos cenários intra-domínio, enquanto a transferência `BreaKHis -> INBreast` confirma o alto custo do gap entre histologia e mamografia. As métricas menores que as dos papers fazem sentido técnico e refletem, principalmente, diferenças de implementação, protocolo experimental, tamanho dos datasets e pré-processamento.
