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

## Diferenças Detalhadas Entre Implementações e Papers

Leitura correta:

- os modelos do repositório são inspirados pelos artigos, mas não devem ser tratados como reproduções fiéis linha a linha;
- em vários pontos o código simplifica a arquitetura, muda o dataset-alvo ou amplia o escopo do paper original;
- por isso, diferenças de métrica em relação aos artigos são esperadas e tecnicamente justificáveis.

### SEConformer vs. paper

O que o paper propõe em alto nível:

- um modelo para histopatologia com blocos residuais `Squeeze-and-Excitation`;
- uma etapa `Conformer` para modelar dependências mais globais;
- avaliação centrada em histopatologia, com menções a `BreaKHis`, `BACH`, resultados por magnificação e estudos de ablação.

O que o código deste repositório implementa:

- três blocos convolucionais residuais com `SE` em `SEConformer/model.py`;
- canais fixos `32 -> 64 -> 128`;
- `AdaptiveAvgPool2d((8, 8))`, flatten completo e projeção linear para `256`;
- um único bloco chamado `ConformerBlock` com `MultiheadAttention` de `4` cabeças e MLP simples;
- classificador final com apenas uma camada linear.

Diferenças arquiteturais importantes:

1. O paper posiciona o `SE-Conformer` como arquitetura própria de detecção/classificação em histopatologia, enquanto o código atual é uma versão bem mais enxuta e experimental.
2. No código só existem `3` blocos convolucionais principais antes da atenção. Isso é muito menor do que o tipo de pipeline profundo normalmente descrito em papers dessa família.
3. A parte `Conformer` do código não recebe uma sequência rica de patches ou tokens espaciais. Depois do flatten, o vetor é convertido em sequência com `x.unsqueeze(1)`, ou seja, sequência de comprimento `1`.
4. Com comprimento de sequência `1`, o `MultiheadAttention` não modela relações entre múltiplos tokens espaciais; ele funciona mais como transformação residual de um único embedding global.
5. O classificador do código é compacto quando comparado ao que se espera de uma arquitetura publicada como contribuição central de paper. Com a ideia de ser mais rápido para treino e teste, sem perder a essência do paper e mantendo um resultado condizente. 

Diferenças de pipeline e protocolo:

1. O paper está claramente focado em histopatologia, enquanto o repositório estende o modelo para `INBreast` e até para transferência `BreaKHis -> INBreast`.
2. Essa transferência entre histologia e mamografia é uma extensão do repositório, não uma reprodução direta do escopo principal do artigo.
3. O código mistura cenários com `holdout`, `StratifiedKFold`, treino em `BACH`, `BRACS`, `BreaKHis` e `INBreast`, enquanto o paper destaca especialmente resultados em histopatologia e estudos de ablação.
4. No repositório, `BreaKHis` pode ser binário ou multiclasse dependendo do CSV e do experimento; isso também altera a comparabilidade com o artigo.
5. O pré-processamento do `INBreast` converte `DICOM` para RGB de 3 canais e redimensiona para `224 x 224`, o que é um pipeline adicional que não faz parte do problema original de histopatologia do paper.
6. O treino do repositório usa `Adam`, `WeightedRandomSampler`, augmentations relativamente simples e seleção de melhor época por `accuracy` ou `f1_macro`.
7. O paper cita ablação e comparação metodológica mais controlada; o código aqui prioriza reuso experimental e consistência entre diferentes datasets.

Conclusão prática sobre o `SEConformer`:

- o modelo do repositório é melhor entendido como uma implementação inspirada no paper;
- ele preserva a ideia de combinar `SE` com uma etapa de atenção;
- mas a etapa conformer foi simplificada e o escopo experimental foi ampliado além do artigo.

### HFTNet vs. paper

O que o paper propõe em alto nível:

- uma arquitetura híbrida de fusão entre CNNs e transformers;
- exploração explícita de características locais e globais;
- uso de múltiplos datasets de câncer de mama, incluindo `BreaKHis`, `BACH` e `BRACS`;
- uma etapa de fusão descrita no paper como núcleo da contribuição metodológica.

O que o código deste repositório implementa:

- cinco backbones do `timm`: `DenseNet201`, `Xception`, `ViT`, `DeiT` e `Swin Tiny`;
- extração independente de features globais de cada backbone;
- projeção linear de cada saída para dimensão `768`;
- concatenação das `5` features, compressão por `Linear(5 * 768 -> 768)`;
- um bloco de `MultiheadAttention` com `16` cabeças;
- um `FFN` simples e um classificador `768 -> 512 -> num_classes`.

Diferenças arquiteturais importantes:

1. O paper apresenta o `HFT-Net` como uma arquitetura de fusão transformer própria, enquanto o repositório monta essa ideia com backbones prontos do `timm`.
2. Em vez de uma implementação dedicada do bloco de fusão do artigo, o código usa uma estratégia direta de `extrai -> projeta -> concatena -> comprime`.
3. Isso quer dizer que o `MultiheadAttention` atual não está fazendo atenção entre múltiplas fontes como tokens independentes;
4. O paper enfatiza a complementaridade entre informações locais e globais; o código preserva essa ideia apenas de forma indireta, porque cada backbone já entrega uma feature global pooled.

Diferenças de pipeline e protocolo:

1. O código usa `ImageNet pretraining` via `timm` em boa parte dos cenários, mas desliga `pretrained` em alguns experimentos como `BACH` e `BRACS`.
2. O treino usa `AdamW` com `StepLR`, `WeightedRandomSampler` e augmentations genéricas, o que pode divergir bastante da configuração fina do artigo.
3. O critério de melhor modelo no código é `val_acc`, mesmo em cenários multiclasses; um paper pode priorizar outra métrica.

Conclusão prática sobre o `HFTNet`:

- esta implementação mantém a intuição do paper de combinar backbones CNN e transformer;
- porém a fusão foi reduzida a uma composição mais simples com módulos prontos;
- o resultado é uma baseline híbrida forte, mas não uma reconstrução fiel do bloco de fusão transformer descrito no artigo.

### HistoDX vs. paper


O que o paper propõe em alto nível:

- um sistema `HistoDX` focado em diagnóstico por imagem histopatológica;
- foco explícito em `invasive ductal carcinoma (IDC)`;
- uso principal de `EfficientNetV2-B3` segundo os metadados e trechos extraídos do PDF;
- validação complementar em `BreaKHis` e `BACH` para discutir generalização multi-dataset.

O que o código deste repositório implementa:

- uma baseline baseada em `EfficientNetV2-S` do `torchvision`;
- substituição apenas da última camada classificadora;
- um pipeline único de treino para histopatologia e mamografia;
- cenários adicionais em `BRACS`, `INBreast` e transferência entre domínios.

Diferenças arquiteturais importantes:

1. O paper menciona `EfficientNetV2-B3`, enquanto o código usa `EfficientNetV2-S`.
2. Essa troca já altera capacidade, número de parâmetros, custo computacional e comportamento de generalização.
3. Isso faz desta pasta muito mais uma baseline inspirada no paper do que uma cópia estrita da arquitetura publicada.
4. A escolha por `EfficientNetV2-S` parece motivada por disponibilidade estável no `torchvision`, não por fidelidade absoluta ao artigo.

Diferenças de treino e avaliação:

1. O repositório usa splits `train/val/test` e `holdout` definidos no próprio código, enquanto o artigo está ancorado em uma validação principal sobre `IDC` com validação complementar multi-dataset.
2. O código usa `Adam` simples, sem scheduler explícito.
3. A seleção do melhor modelo não é feita por um pipeline complexo de early stopping ou tuning avançado.
4. O tratamento de desequilíbrio usa `WeightedRandomSampler` e pesos de classe derivados do `train_df`.
5. Caso o paper use procedimentos adicionais de interpretação, explicabilidade ou análise clínica, eles são implementados.

Conclusão prática sobre o `HistoDX`:

- a implementação atual é uma baseline forte e útil;
- mas ela representa um `EfficientNetV2-S` adaptado ao projeto, não a versão exata `HistoDX` do paper;
- por isso os resultados daqui devem ser comparados aos do artigo apenas como aproximação experimental.

### Resumo Executivo Das Diferenças

Se for preciso resumir em uma frase por modelo:

- `SEConformer`: mantém a ideia `SE + atenção`, mas o bloco conformer do código é mais simples do que o paper sugere e opera sobre um único token global.
- `HFTNet`: preserva a noção de fusão entre CNNs e transformers, mas implementa isso como ensemble híbrido com concatenação e atenção o token.
- `HistoDX`: funciona como baseline `EfficientNetV2-S` inspirada no paper, enquanto o artigo aponta para uma formulação centrada em `EfficientNetV2-B3` e no dataset `IDC`.

Portanto, os nomes das pastas seguem os artigos, mas as implementações do repositório devem ser lidas como:

- reproduções parciais;
- simplificações pragmáticas para fins experimentais;
- extensões de escopo para novos datasets e cenários de transferência.

### Diferenças Entre Modelos Em Cada Dataset

Além das diferenças em relação aos papers, há uma segunda camada importante de diferença: os modelos não tratam todos os datasets exatamente do mesmo jeito.

Isso importa porque muda:

- o número de classes;
- o tipo de problema (`binário` vs `multiclasse`);
- as métricas disponíveis;
- a dificuldade real do experimento;
- o quão justa é a comparação direta entre dois resultados.

#### Visão rápida por dataset

| Dataset | Classes naturais do dataset | SEConformer | HFTNet | HistoDX | Impacto prático |
| --- | --- | --- | --- | --- | --- |
| `BreaKHis` | `2` classes binárias ou `8` subtipos | suporta `2` e `8` classes | suporta `2` e `8`, mas o fluxo principal está orientado a `8` classes | implementado no repositório apenas para `2` classes | nem sempre os três modelos estão resolvendo exatamente o mesmo problema |
| `INBreast` | `2` classes derivadas de `BI-RADS` ou `6` classes `BI-RADS 1-6` | suporta binário e multiclasse | suporta binário e multiclasse, com padrão mais orientado a multiclasse | baseline principal do repositório usa multiclasse, embora a base suporte binário | comparar resultados exige saber qual CSV e qual codificação de rótulo foram usados |
| `BACH` | `4` classes | `4` classes | `4` classes | `4` classes | aqui a comparação é mais direta entre arquiteturas |
| `BRACS` | `7` classes | `7` classes | `7` classes | `7` classes | aqui a comparação também é mais direta, embora o protocolo ainda varie por modelo |

#### BreaKHis

O `BreaKHis` é o dataset onde a diferença entre modelos mais muda o significado da tarefa.

- o dataset permite leitura binária: `benign` vs `malignant`;
- também permite leitura multiclasse por subtipo histológico, com `8` classes;
- portanto, dois resultados em `BreaKHis` podem parecer comparáveis no nome do cenário, mas na prática podem estar resolvendo problemas diferentes.

Como cada modelo trata o `BreaKHis`:

1. `SEConformer`
- o código suporta tanto `mode="binary"` quanto modo multiclasse em `build_breakhis_dataframe`;
- o baseline principal de notebook do repositório para `BreaKHis` está configurado em `mode="binary"`;
- o arquivo versionado `SEConformer/breakhis_binary_folds.csv` também aponta para esse uso binário.

2. `HFTNet`
- o parser suporta binário e multiclasse;
- porém o fluxo principal de `run_breakhis_baseline_holdout` usa `mode="multiclass"` por padrão;
- o próprio `HFTNet` nasce com `num_classes=8` no `model.py`;
- os notebooks principais de `BreaKHis` e de transferência também estão alinhados com a leitura multiclasse de `8` subtipos.

3. `HistoDX`
- no repositório, `collect_breakhis_images` foi implementado apenas para `mode="binary"`;
- portanto o baseline `HistoDX` em `BreaKHis` aqui é exclusivamente `2` classes.

Consequência prática em `BreaKHis`:

- `SEConformer` e `HistoDX` podem estar sendo comparados em binário;
- `HFTNet` pode estar sendo avaliado em um problema multiclasse de `8` classes;
- então, sem verificar o CSV ou o notebook exato usado em cada execução, uma comparação direta de `accuracy` entre os três pode ser injusta.

#### INBreast

O `INBreast` também muda bastante de significado conforme a codificação dos rótulos.

- o repositório suporta forma binária com limiar `BI-RADS >= 4` como classe positiva;
- também suporta forma multiclasse com `6` classes, correspondendo a `BI-RADS 1` até `BI-RADS 6`;
- isso altera não só a dificuldade, mas a própria interpretação clínica do problema.

Como cada modelo trata o `INBreast`:

1. `SEConformer`
- `build_inbreast_csv` em `SEConformer/data.py` tem `mode="binary"` por padrão;
- porém os notebooks principais de baseline em `INBreast` usam a geração do CSV em modo multiclasse;
- logo, o código suporta os dois cenários, mas o analisado foi o multiclasse.

#### BACH

No `BACH`, a comparação entre modelos fica mais limpa porque todos trabalham essencialmente sobre a mesma estrutura de rótulos.

- o dataset é tratado como problema de `4` classes;
- as classes naturais da pasta local são `Normal`, `Benign`, `InSitu` e `Invasive`;

Mesmo assim, ainda existem diferenças de protocolo:

1. `SEConformer`
- usa augmentação mais forte para `BACH`;
- usa `epoch_size_multiplier=2.0`, o que aumenta a amostragem efetiva por época;
- continua com seleção de melhor época pelo score apropriado.

2. `HFTNet`
- usa `pretrained=False` no baseline de `BACH`;
- treina a arquitetura híbrida inteira para `4` classes;
- continua com `AdamW + StepLR`.

3. `HistoDX`
- também usa `pretrained=False` no baseline de `BACH`;
- adapta `EfficientNetV2-S` para `4` classes.

Consequência prática em `BACH`:

- aqui a diferença principal não é o número de classes entre modelos;
- a diferença principal é arquitetura, augmentação e estratégia de treino.

#### BRACS

No `BRACS`, os três modelos estão mais alinhados em termos de rótulo final.

- o dataset local está organizado em `7` classes;
- as pastas são `0_N`, `1_PB`, `2_UDH`, `3_FEA`, `4_ADH`, `5_DCIS` e `6_IC`;
- os três pipelines consomem essa estrutura como problema multiclasse de `7` classes.

Diferenças entre modelos em `BRACS`:

1. `SEConformer`
- usa augmentação moderada;
- usa `epoch_size_multiplier=1.25`;
- otimiza com foco em score multiclasse.

2. `HFTNet`
- usa `pretrained=False` no baseline `BRACS`;
- mantém a fusão entre cinco backbones mesmo num dataset relativamente menor.

3. `HistoDX`
- também entra como `pretrained=False` em `BRACS`;
- funciona como baseline monolítica de backbone único para `7` classes.

Consequência prática em `BRACS`:

- aqui os três modelos estão, em princípio, resolvendo o mesmo problema de `7` classes;
- por isso, `BRACS` é um dos cenários mais apropriados para comparação arquitetural direta entre eles.

#### Efeito direto nas métricas

As diferenças de dataset e de número de classes alteram também o tipo de métrica calculada.

1. Em cenários binários:
- aparecem métricas como `specificity`, `ROC`, `precision-recall` binária e `AUC` binária;
- no `SEConformer` e no `HFTNet`, a lógica de avaliação separa explicitamente binário de multiclasse;
- o `HistoDX` calcula `roc_auc` binária no caso binário, mas não calcula `specificity` da mesma forma que o `SEConformer`.

2. Em cenários multiclasses:
- as métricas passam para `precision_macro`, `recall_macro`, `f1_macro` e `auc_macro_ovr`;
- essas métricas geralmente são mais severas e menos diretamente comparáveis com uma `accuracy` binária simples;
- portanto, um `92%` em binário não significa a mesma coisa que um `92%` em `8` classes.

#### Implicação para ler a tabela de resultados

A tabela consolidada de resultados deste repositório deve ser lida com esta cautela:

- `BACH` e `BRACS` são os cenários mais homogêneos para comparar arquiteturas;
- `BreaKHis` pode mudar bastante de dificuldade dependendo de estar em `2` ou `8` classes;
- `INBreast` pode mudar de problema binário para problema `BI-RADS` multiclasse de `6` classes;
- transferência `BreaKHis -> INBreast` adiciona, além da mudança de domínio, uma possível mudança no espaço de rótulos.

Em outras palavras:

- quando muda o dataset, muda o domínio visual;
- quando muda o número de classes, muda a própria tarefa;
- e, em alguns casos deste repositório, mudam as duas coisas ao mesmo tempo.

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
