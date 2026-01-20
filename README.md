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


