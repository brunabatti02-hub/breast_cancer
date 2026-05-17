 Documentacao Tecnica Completa do Projeto


1. Histopatologia (BreaKHis): imagens de lamina histologica, com texturas complexas e padroes microscópicos.
2. Mamografia (INBreast): imagens radiologicas em DICOM, com contraste baixo, ruido e alto grau de variabilidade.

Esses dominios sao bastante diferentes. A histologia apresenta padroes locais ricos (celulas, estruturas glandulares), enquanto a mamografia apresenta padroes globais e sutis (masas, assimetrias, calcificacoes). A transferencia entre esses dominios exige cuidado para evitar sobreajuste e catastrofica degradacao das representacoes aprendidas.


2. Datasets

2.1 BreaKHis (Histologia)
Dominio: microscopia de tecido mamario.
Classes:
  - Binario: benigno vs maligno.
  - Multiclasse: 8 subtipos histologicos.
Desafio principal: desbalanceamento entre classes e alta variabilidade intra-classe.

2.2 INBreast (Mamografia)
Dominio: mamografia (DICOM).
Labels principais:
  - BI-RADS: 1–6, escala clinica de suspeita.
  - ACR: densidade mamaria (usado em trabalhos especificos).
Aqui foi adotado:
  - Binario: BI-RADS >= 4 como maligno.
  - Multiclasse: BI-RADS 1–6 (mapeado para 0–5).

Desafio principal: dataset pequeno, dominio distinto e imagens DICOM que precisam de normalizacao.


3. Preprocessamento

3.1 Histologia (BreaKHis)
Imagens ja em RGB.
Normalizacao simples (mean/ std).\n- Augmentations para robustez: flips, rotacoes, affine.

3.2 Mamografia (INBreast DICOM)
Conversao DICOM:
  - Leitura do pixel array.
  - Rescale slope/intercept (se existir).
  - Inversao se MONOCHROME1.
  - Normalizacao para 0–255.
  - Conversao para 3 canais (RGB fake) para compatibilidade com CNN.


4. Balanceamento de Classes

O desbalanceamento e comum em datasets medicos. Foram aplicadas duas estrategias principais:

1. WeightedRandomSampler (PyTorch)
   - Superamostra classes minoritarias.
   - Ajuda a expor o modelo a exemplos raros durante o treino.

2. CrossEntropyLoss com pesos
   - Penaliza mais erros em classes minoritarias.
   - Complementa o sampler e estabiliza o gradiente.

Essas estrategias foram aplicadas nos seguintes modelos:
SEConformer (histologia e INBreast)
SEConformer Transfer Learning
HFTNet
HistoDX


5. Augmentation

Augmentations aumentam a diversidade do dataset e reduzem overfitting. As principais transformacoes aplicadas:

Flip horizontal e vertical: simula variacoes de orientacao.
Rotacoes pequenas (10–15 graus): invariancia a pequenos angulos.
Affine (shear + translate): simula deformacoes anatomicas e variacoes no posicionamento.

Em histologia, essas transformacoes sao altamente efetivas porque a orientacao da lamina e arbitraria. Em mamografia, sao usadas com cautela, mas ainda ajudam a melhorar a robustez do classificador.


6. Modelos e Arquiteturas

6.1 SEConformer

Objetivo: classificar imagens binario/multiclasse em histologia e mamografia.

Arquitetura:
Blocos convolucionais com Squeeze-and-Excitation (SE).
Conformer integra:
  - CNN: padroes locais (textura, borda).
  - Multihead Attention: padroes globais e dependencias longas.

Racional tecnico:
SE melhora discriminacao ao realcar canais importantes.
Conformer reduz dependencia exclusiva de convolucao e permite atencao global.

Treinamento:
Loss: CrossEntropy com pesos.
Sampler: WeightedRandomSampler.
Augmentation ativa.


6.2 SEConformer Transfer Learning (Histologia -> Mamografia)

Objetivo: aproveitar representacoes aprendidas em histologia para mamografia.

Pipeline:
1. Treina SEConformer em histologia.
2. Carrega pesos no modelo para mamografia.
3. Congela backbone (SE + Conformer).
4. Ajusta somente o classificador final.

Justificativa:
O backbone captura padroes genericos (texturas, contraste, bordas) que podem ser transferidos.
Congelar reduz risco de overfitting e catastrofica forgetting.
Somente a camada final se adapta a distribuicao do dataset alvo.

Configuracao atual:
Multiclasse BI-RADS 1–6.
Balanceamento e augmentation ativos.


6.3 HFTNet

Objetivo: classificar 8 subtipos de histologia (multiclasse).

Arquitetura:
Ensemble/ fusao de backbones heterogeneos:
  - DenseNet, Xception (CNNs)
  - ViT, DeiT, Swin (Transformers)

Racional tecnico:
CNNs capturam textura local.
Transformers capturam relacoes globais.
Fusao permite maior poder representacional.

Treinamento:
Augmentation forte.
Loss com pesos para balanceamento.


6.4 HistoDX 

Objetivo: classificar benigno vs maligno em histologia.

Arquitetura:
EfficientNetV2-S como backbone.
Classificador final custom.

Treinamento:
Augmentation basica.
Balanceamento via sampler + class weights.



7. Transferencia de Aprendizado (Resumo Tecnico)

Transfer learning foi aplicado no SEConformer, com estrategia conservadora:
Congelamento do backbone.
Ajuste apenas do classificador.

Beneficios:
Menor risco de overfitting.
Estabilidade em datasets pequenos (INBreast).
Preserva representacoes robustas aprendidas no dominio fonte.

Riscos:
Se dominios forem muito distintos, pode limitar a adaptacao.
Solucao: fine-tuning em duas fases (congelado -> descongelado).


8. Desenvolvimento Tecnico (Como o Codigo Foi Construido)

Esta secao descreve o desenho do codigo e as decisoes de implementacao.

8.1 Organizacao em pacotes
Cada modelo foi estruturado em uma pasta com modulos `.py` separados:
`data.py` -> leitura e preparo dos dados.
`model.py` -> definicao da arquitetura.
`train.py` -> loop de treino e validacao.
`eval.py` (quando aplicavel) -> metricas e graficos.
`io_utils.py` -> salvamento de logs, metricas e imagens.

Essa separacao facilita manutencao e permite reaproveitar codigo entre modelos.

8.2 Padrao de Treinamento
Todos os treinamentos seguem o mesmo fluxo:
1. Carregar dataset e gerar splits.
2. Criar DataLoaders com sampler balanceado.
3. Instanciar modelo.
4. Treinar por epoca (loss + atualizacao de pesos).
5. Validar e registrar metricas.
6. Salvar checkpoints e graficos.

8.3 Design de augmentations
As transformações foram escolhidas para aumentar robustez sem distorcer semântica clínica.
Em mamografia, augmentations foram mantidas moderadas (rotação pequena).

8.4 Balanceamento e impacto no treinamento
O sampler garante que todas as classes aparecem com frequência similar.
A loss ponderada evita que o modelo aprenda a sempre prever a classe majoritária.

8.5 Transfer Learning
A implementação carrega pesos com `strict=False` para compatibilidade.
Camadas congeladas com `requires grad=False`.
Apenas o classificador final e atualizado.







# Multimodal Representation Learning for Breast Cancer Risk

## Mamografia (Imagem) + Genética (PGS/PRS)

------------------------------------------------------------------------
## 📊 Datasets Utilizados

### 🧬 Polygenic Score (PGS) Catalog – EBI
🔗 https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/

Este dataset é mantido pelo **European Bioinformatics Institute (EBI)** e contém **Polygenic Risk Scores (PRS/PGS)** para diversas doenças e características complexas. Porém nesse estudo, selecionamos somente o breast cancer. 
Os scores são calculados a partir de variantes genéticas (SNPs) e são amplamente utilizados em estudos de:
- Genômica
- Epidemiologia genética
- Predição de risco de doenças complexas, incluindo câncer

Os dados incluem pesos genéticos, identificadores de variantes e metadados associados a cada score.

---

### 🩻 CBIS-DDSM – Cancer Imaging Archive
🔗 https://www.cancerimagingarchive.net/wp-content/uploads/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia

O **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** é um dataset público de imagens médicas focado em **mamografias para detecção de câncer de mama**.  
Ele é uma versão curada do dataset DDSM original e inclui:
- Imagens de mamografia em alta resolução
- Anotações de lesões (benignas e malignas)
- Segmentações e metadados clínicos

Este dataset é amplamente utilizado em pesquisas de:
- Visão computacional
- Deep Learning
- Diagnóstico assistido por computador (CAD) em câncer de mama

## 1. Motivação Científica

Em cenários reais, é comum que diferentes modalidades de dados
biomédicos provenham de **cohorts distintos**, impossibilitando o
pareamento direto de indivíduos. Este projeto propõe uma solução
metodologicamente correta:

> **Elevar a multimodalidade ao nível de representação, não ao nível do
> paciente.**

Cada modalidade aprende um espaço latente próprio, e a integração ocorre
por **alinhamento geométrico entre espaços latentes**.

Objetivo científico: \> Investigar se diferentes modalidades capturam
padrões estruturais convergentes \> de risco para câncer de mama.

Outra possível abordagem encontrada na literatura:https://www.nature.com/articles/s42256-025-01052-4#Sec12

------------------------------------------------------------------------

## 2. Experimento 1 --- Mamografia

### Dataset

-   CBIS-DDSM (TCIA)
-   Formato: DICOM
-   Organização: séries em pastas como `Calc-Test_P_00038_LEFT_CC`

### Pipeline

1.  Varredura das pastas CBIS-DDSM\
2.  Localização dos arquivos `.dcm`\
3.  Leitura DICOM com:
    -   VOI LUT
    -   Correção de MONOCHROME1
    -   RescaleSlope / Intercept
4.  Normalização por percentis
5.  Pré-processamento:
    -   Resize para 224×224
    -   Conversão para tensor \[3,224,224\]
6.  Extração de embedding por CNN (ResNet50 sem cabeça)
7.  Agregação por série (média)
8.  Saídas:
    -   `mammo_embeddings.npy`
    -   `mammo_metadata.csv`

### Produto

Um vetor de alta dimensão por série de mamografia, representando padrões
visuais relacionados a risco.

------------------------------------------------------------------------

## 3. Experimento 2 --- Genética (PGS/PRS)

### Dataset

-   PGS Catalog (scores poligênicos)

### Pipeline

1.  Leitura dos arquivos PGS
2.  Extração de estatísticas por score:
    -   número de SNPs
    -   média, desvio, soma dos pesos, etc.
3.  Construção de matriz tabular
4.  Treinamento de Autoencoder
5.  Compressão para espaço latente

### Produto

Um vetor latente por score genético, capturando padrões globais de
risco.

------------------------------------------------------------------------

## 4. Compressão Latente

Tanto imagem quanto genética passam por Autoencoders independentes:

-   Entrada: embedding original
-   Saída: espaço latente compacto (ex.: 3--32 dimensões)
-   Arquivos finais:
    -   `autoencoder_latent_space.csv` (imagem)
    -   `genetic_latent_space.csv` (genética)

Formato:

  join_id   z1   z2   z3   ...
  --------- ---- ---- ---- -----

------------------------------------------------------------------------

## 5. Integração Multimodal sem Pareamento

Como não há pacientes em comum, **não é possível** usar CCA por
indivíduo.

Solução adotada:

1.  Padronizar cada espaço separadamente
2.  Criar protótipos via KMeans em cada espaço
3.  Calcular similaridade entre protótipos
4.  Matching ótimo (Hungarian Algorithm)
5.  Estimar transformação geométrica (Procrustes)
6.  Alinhar genética → imagem

Produto:

-   `shared_space_images.csv`
-   `shared_space_genetics.csv`
-   `alignment_matrix_R.npy`

Agora ambas modalidades vivem em um **espaço compartilhado**.

------------------------------------------------------------------------

## 6. Testes Realizados

### Diagnóstico de cada espaço

-   Histogramas por dimensão
-   Correlação interna
-   PCA 2D
-   KMeans + Silhouette
-   Detecção de outliers

### Integração estrutural

-   Similaridade entre protótipos (antes/depois)
-   Teste de permutação
-   Visualização em PCA do espaço alinhado

Esses testes avaliam se existe **estrutura alinhável** entre
modalidades.

------------------------------------------------------------------------

## 7. O que seria possível com pacientes pareados

Se existissem dados de imagem e genética para os **mesmos pacientes**:

1.  CCA / Deep CCA por indivíduo
2.  Retrieval cross-modal (imagem → genética)
3.  Concordância de risco
4.  Modelos supervisionados multimodais
5.  AUC, Recall@K, MRR
6.  Testes de permutação por paciente

O notebook inclui células prontas para esses testes, documentando
claramente o caminho futuro.

------------------------------------------------------------------------

## 8. Conclusão

Este projeto:

-   Evita pareamento artificial
-   Mantém validade estatística
-   É eticamente defensável
-   Demonstra convergência estrutural entre modalidades
-   Prepara o terreno para integração real quando dados pareados
    existirem

> Multimodalidade por representação é a solução correta quando cohorts
> são disjuntos.
# Multimodal Representation Learning for Breast Cancer Risk

## Mamografia (Imagem) + Genética (PGS/PRS)

------------------------------------------------------------------------

## 1. Motivação Científica

Em cenários reais, é comum que diferentes modalidades de dados
biomédicos provenham de **cohorts distintos**, impossibilitando o
pareamento direto de indivíduos. Este projeto propõe uma solução
metodologicamente correta:

> **Elevar a multimodalidade ao nível de representação, não ao nível do
> paciente.**

Cada modalidade aprende um espaço latente próprio, e a integração ocorre
por **alinhamento geométrico entre espaços latentes**.

Objetivo científico: \> Investigar se diferentes modalidades capturam
padrões estruturais convergentes \> de risco para câncer de mama.

------------------------------------------------------------------------

## 2. Experimento 1 --- Mamografia

### Dataset

-   CBIS-DDSM (TCIA)
-   Formato: DICOM
-   Organização: séries em pastas como `Calc-Test_P_00038_LEFT_CC`

### Pipeline

1.  Varredura das pastas CBIS-DDSM\
2.  Localização dos arquivos `.dcm`\
3.  Leitura DICOM com:
    -   VOI LUT
    -   Correção de MONOCHROME1
    -   RescaleSlope / Intercept
4.  Normalização por percentis
5.  Pré-processamento:
    -   Resize para 224×224
    -   Conversão para tensor \[3,224,224\]
6.  Extração de embedding por CNN (ResNet50 sem cabeça)
7.  Agregação por série (média)
8.  Saídas:
    -   `mammo_embeddings.npy`
    -   `mammo_metadata.csv`

### Produto

Um vetor de alta dimensão por série de mamografia, representando padrões
visuais relacionados a risco.

------------------------------------------------------------------------

## 3. Experimento 2 --- Genética (PGS/PRS)

### Dataset

-   PGS Catalog (scores poligênicos)

### Pipeline

1.  Leitura dos arquivos PGS
2.  Extração de estatísticas por score:
    -   número de SNPs
    -   média, desvio, soma dos pesos, etc.
3.  Construção de matriz tabular
4.  Treinamento de Autoencoder
5.  Compressão para espaço latente

### Produto

Um vetor latente por score genético, capturando padrões globais de
risco.

------------------------------------------------------------------------

## 4. Compressão Latente

Tanto imagem quanto genética passam por Autoencoders independentes:

-   Entrada: embedding original
-   Saída: espaço latente compacto (ex.: 3--32 dimensões)
-   Arquivos finais:
    -   `autoencoder_latent_space.csv` (imagem)
    -   `genetic_latent_space.csv` (genética)

Formato:

  join_id   z1   z2   z3   ...
  --------- ---- ---- ---- -----

------------------------------------------------------------------------

## 5. Integração Multimodal sem Pareamento

Como não há pacientes em comum, **não é possível** usar CCA por
indivíduo.

Solução adotada:

1.  Padronizar cada espaço separadamente
2.  Criar protótipos via KMeans em cada espaço
3.  Calcular similaridade entre protótipos
4.  Matching ótimo (Hungarian Algorithm)
5.  Estimar transformação geométrica (Procrustes)
6.  Alinhar genética → imagem

Produto:

-   `shared_space_images.csv`
-   `shared_space_genetics.csv`
-   `alignment_matrix_R.npy`

Agora ambas modalidades vivem em um **espaço compartilhado**.

------------------------------------------------------------------------

## 6. Testes Realizados

### Diagnóstico de cada espaço

-   Histogramas por dimensão
-   Correlação interna
-   PCA 2D
-   KMeans + Silhouette
-   Detecção de outliers

### Integração estrutural

-   Similaridade entre protótipos (antes/depois)
-   Teste de permutação
-   Visualização em PCA do espaço alinhado

Esses testes avaliam se existe **estrutura alinhável** entre
modalidades.

------------------------------------------------------------------------

## 7. O que seria possível com pacientes pareados

Se existissem dados de imagem e genética para os **mesmos pacientes**:

1.  CCA / Deep CCA por indivíduo
2.  Retrieval cross-modal (imagem → genética)
3.  Concordância de risco
4.  Modelos supervisionados multimodais
5.  AUC, Recall@K, MRR
6.  Testes de permutação por paciente

O notebook inclui células prontas para esses testes, documentando
claramente o caminho futuro.

------------------------------------------------------------------------

## 8. Conclusão

Este projeto:

-   Evita pareamento artificial
-   Mantém validade estatística
-   É eticamente defensável
-   Demonstra convergência estrutural entre modalidades
-   Prepara o terreno para integração real quando dados pareados
    existirem

> Multimodalidade por representação é a solução correta quando cohorts
> são disjuntos.



