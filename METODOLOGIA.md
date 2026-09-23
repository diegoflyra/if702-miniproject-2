# Metodologia de experimentação empírica em blocos

Este documento descreve **o método**, não o projeto específico do CIFAR-10: a intenção é que ele sirva
de guia para montar um novo estudo empírico — outro algoritmo de treinamento, outros hiperparâmetros,
outra base de dados, outro objetivo — reaproveitando a mesma disciplina que fez este projeto funcionar
como um relatório executável e defensável. Onde ajuda, os exemplos usam o CIFAR-10/MLP/CNN deste
repositório, mas cada peça está descrita de forma genérica o suficiente para ser adaptada.

A pergunta de fundo, em qualquer domínio, é sempre a mesma: **o que funciona melhor, e por quê** —
descoberto por execuções reais, não por intuição. Tudo abaixo existe para responder essa pergunta sem
enganar a si mesmo no caminho (vazamento de dados, ruído confundido com sinal, "o melhor entre muitos"
tratado como se fosse "o melhor").

## 1. A estrutura em blocos

Em vez de otimizar tudo de uma vez (um grid gigante com todos os hiperparâmetros cruzados, caro e
difícil de interpretar) ou de escolher hiperparâmetros um a um "no olho", o estudo é dividido em
**blocos sequenciais**, cada um respondendo a **uma pergunta específica**:

- **Bloco 0 — Referência.** Antes de otimizar qualquer coisa, estabeleça um piso e um ponto de partida
  conhecido (um baseline trivial e a implementação "padrão"/ingênua do problema). Todo ganho dos blocos
  seguintes é medido contra essa referência.
- **Bloco 1 — Capacidade/estrutura do modelo.** A pergunta mais fundamental: que arquitetura ou
  estrutura básica absorve o problema? (Neste projeto: profundidade × largura da MLP; blocos
  convolucionais, kernel, padding e redução espacial da CNN.) Isso é decidido **antes** de otimização e
  regularização, porque a estrutura define a capacidade sobre a qual o resto atua.
- **Bloco 2 — Otimização.** Com a estrutura fixa, qual algoritmo de treinamento e qual taxa de
  aprendizagem convergem melhor? Testar otimizador × taxa lado a lado no mesmo grid (em vez de fixar um
  e variar o outro) revela a faixa útil de cada um e onde ele diverge/colapsa.
- **Checagem de interação.** Depois de mudar o otimizador, a escolha do Bloco 1 ainda é a melhor? A
  busca em blocos assume que cada decisão continua boa quando a próxima muda algo — essa suposição
  **precisa ser testada**, não assumida. Retreine os 2º e 3º colocados do bloco anterior com a
  configuração nova e compare.
- **Bloco 3 — Regularização (e o que mais controla generalização).** Com a rede convergindo bem, ataque
  o overfitting: dropout, penalização, normalização, função de erro. Aqui o gap treino–validação passa a
  ser uma métrica de primeira classe, não só a acurácia.
- **Blocos complementares.** Quando um bloco anterior deixa uma pergunta em aberto — um eixo que não pôde
  ser testado plenamente por causa de outra escolha (ex.: "a checagem foi feita sem regularização"), um
  resultado que pode estar limitado pelo orçamento de épocas, ou uma suspeita de interação — abra um
  bloco complementar dedicado a essa pergunta, com o mesmo protocolo. Não deixe a dúvida sem resposta
  só porque ela "não estava no plano original".
- **Bônus/eixos ortogonais.** Alguns fatores não são hiperparâmetros do modelo (ex.: data augmentation é
  pré-processamento dos dados, não da rede) e por isso ficam **fora** da sequência principal de blocos,
  mas seguem o mesmo protocolo e partem do campeão encontrado até ali.

**A regra que dá disciplina a tudo isso:** cada bloco herda automaticamente o campeão do bloco anterior
— nenhum valor de hiperparâmetro é escolhido manualmente. Isso é o que torna a cadeia de decisões
auditável: dá para apontar exatamente qual execução decidiu qual valor.

### Adaptando a sequência de blocos a outro problema

A sequência acima (capacidade → otimização → regularização) é uma ordem sensata para redes neurais, mas
o princípio generaliza: **ordene os blocos do fator mais fundamental (o que redefine o espaço de busca
dos demais) para o mais incremental**. Alguns exemplos:

- **Gradient boosting em dados tabulares:** Bloco 0 = modelo linear/árvore única; Bloco 1 = profundidade
  das árvores e número de estimadores (capacidade); Bloco 2 = taxa de aprendizagem e subsampling
  (otimização/regularização implícita do boosting); Bloco 3 = regularização explícita (L1/L2, min child
  weight); complementar = tratamento de classes desbalanceadas.
- **Fine-tuning de um modelo de linguagem:** Bloco 0 = zero-shot/few-shot sem fine-tuning; Bloco 1 =
  tamanho do adaptador/rank do LoRA (capacidade); Bloco 2 = taxa de aprendizagem e scheduler; Bloco 3 =
  regularização (weight decay, dropout, early stopping); bônus = augmentation dos dados de treino
  (paráfrase, back-translation).
- **Clustering ou modelos não supervisionados:** Bloco 0 = baseline (k-means com k arbitrário); Bloco 1 =
  número de clusters/estrutura (cotovelo, silhueta); Bloco 2 = inicialização e métrica de distância;
  Bloco 3 = regularização/robustez a outliers.

O que não muda: cada bloco varia **um grupo coeso de hiperparâmetros**, herda o resto do campeão
anterior, e a pergunta de cada bloco é explícita antes de rodar o grid.

## 2. Triagem e confirmação: evitando a "maldição do vencedor"

Testar todas as configurações de um grid inteiro em todos os folds/repetições é caro. O padrão usado
aqui resolve isso em duas etapas, dentro de cada bloco:

1. **Triagem:** todas as configurações do grid rodam apenas um subconjunto das partições/repetições
   (ex.: 3 de 5 folds). Isso é barato e já separa o que claramente não funciona do que é competitivo.
2. **Confirmação:** as N melhores da triagem (tipicamente 2–3) completam **todas** as partições/
   repetições restantes. O campeão do bloco é a maior média de validação nas repetições completas —
   **nunca** a média parcial da triagem.

Por que isso importa: o 1º colocado de uma triagem com muitas configurações tende a estar
**otimisticamente favorecido pelo ruído** daquele subconjunto específico de partições — a chamada
"maldição do vencedor" (winner's curse). Neste projeto isso foi flagrado ao vivo: uma configuração da
MLP tinha 53,1% de acurácia em 3 folds e caiu para 52,7% quando completou os 5. Sem a etapa de
confirmação, esse número inflado teria sido reportado como o resultado real. Ao comparar duas
configurações do topo, uma diferença menor que o desvio entre folds é ruído, não uma vitória.

**Generalização:** troque "fold" por qualquer unidade de repetição que faça sentido no seu domínio —
seeds de inicialização, sub-amostras do dataset, janelas temporais em série temporal, etc. O princípio é
sempre "decidir com pouco, confirmar com mais, nunca reportar como final o número que só passou pela
etapa barata".

## 3. Validação cruzada e o contrato validação × teste

- **Partições fixas em todo o estudo.** As mesmas K partições (ex.: 5-fold estratificado) são usadas em
  **todos** os experimentos, com a mesma seed. Isso permite comparar duas configurações **partição a
  partição** (ver seção 5), não só pela média.
- **Toda escolha usa só a validação.** O conjunto de teste nunca participa de nenhuma decisão de
  hiperparâmetro — nem para desempatar, nem "só para conferir". Usá-lo para escolher, mesmo uma vez,
  contamina o número final com um otimismo artificial (vazamento de informação do teste para o
  processo de decisão).
- **O teste é revelado uma única vez, no fim**, para os campeões de cada bloco e para as referências —
  nunca para configurações descartadas. A tabela final mostra a concordância entre validação e teste
  como evidência de que a metodologia não vazou informação (quando os dois números batem de perto, isso
  é o esperado; uma discrepância grande é um sinal de alerta a investigar, não a ignorar).
- **Early stopping e restauração dos melhores pesos** usam a métrica de validação (nunca a de teste), e
  as métricas de treino devem ser medidas em modo avaliação (sem dropout/augmentation) para que
  comparações de gap treino–validação sejam justas entre configurações com regularizações diferentes.

## 4. O que persistir de cada execução (e por quê)

Ferramentas de rastreamento em nuvem (W&B, MLflow, TensorBoard remoto, etc.) são úteis, mas **nunca
devem ser a única cópia dos resultados**: sessões podem cair, cotas expiram, contas mudam. A regra
usada aqui — e que vale para qualquer stack — é **local primeiro, espelho depois**:

Para cada execução, grave em disco, ao lado do restante do projeto:

- **Hiperparâmetros exatos** da execução (um JSON com todo argumento usado, incluindo os herdados do
  bloco anterior) — sem isso, uma execução não pode ser reproduzida nem citada com precisão no relatório.
- **Histórico por época/iteração**, com as métricas de treino e validação (loss, e todas as métricas de
  interesse — aqui, acurácia, precision, recall e F1, gerais e por classe). É dessa tabela que vêm as
  curvas de convergência e a época de melhor desempenho.
- **Pesos/estado do melhor modelo** (pela métrica de validação monitorada), não só o do fim do
  treinamento — especialmente relevante com early stopping.
- **Um resumo consolidado da execução**: métricas finais de validação e teste (média ± desvio entre
  partições), a época/iteração de parada, e o suficiente para alimentar rankings sem reabrir os
  históricos brutos.
- Torne o salvamento **robusto a interrupções**: grave incrementalmente (não só ao final), e faça a
  reexecução pular o que já está completo em disco em vez de recomeçar do zero. Um estudo de muitas
  horas de GPU/CPU que não sobrevive a uma queda de sessão é um estudo que será refeito do zero na pior
  hora.

Essa disciplina existe porque o **relatório final** — não o modelo treinado — costuma ser o entregável
real de um estudo de ablação: tabelas, gráficos e a explicação causal de cada resultado são construídos
a partir desses arquivos locais, meses depois de qualquer sessão de treino ter terminado.

## 5. Ferramental de análise (o que vale a pena automatizar)

Escrever um pequeno módulo de utilidades de relatório (aqui, `src/report_utils.py`) que leia
exclusivamente os arquivos locais do passo 4 compensa rapidamente o esforço, porque as mesmas operações
se repetem em todo bloco:

- **Ranking pela validação** de todas as configurações de um bloco (triagem e confirmação
  separadamente), nunca expondo o teste.
- **Heatmap de dois (ou três, com facetas) eixos do grid** contra a métrica de validação — a forma mais
  rápida de enxergar onde fica a região boa do espaço de busca, incluindo marcar combinações inválidas
  descartadas antes do treino.
- **Curvas de treino/validação por época**, sobrepondo algumas configurações-chave, para embasar
  argumentos como "esta configuração decora mais cedo" ou "esta regularização atrasa o overfitting".
- **Comparação pareada por partição.** Como as partições são as mesmas em todo o estudo, a diferença
  entre duas configurações pode ser calculada **partição a partição**, não só entre médias. Isso revela
  efeitos pequenos mas consistentes ("venceu em 5 de 5 folds por uma margem pequena") que a sobreposição
  de intervalos de média ± desvio esconderia, e é a ferramenta certa para responder "essa mudança
  realmente ajudou, ou é ruído?".
- **Métricas por classe (ou por segmento relevante do problema)**, não só a métrica agregada — é onde
  aparecem efeitos que a média esconde (uma classe/segmento que puxa a média para baixo, uma arquitetura
  que só ganha em certos casos).
- **Relatório final consolidado**, revelando o teste apenas para os papéis "campeão" e "referência".

**Generalização:** em outro domínio, troque "heatmap de hiperparâmetros" e "métrica por classe" pelo
equivalente que fizer sentido (ex.: erro por segmento temporal numa série, por grupo demográfico numa
tarefa de classificação, por comprimento de sequência numa tarefa de NLP) — o padrão de construir esse
ferramental **uma vez** e reusá-lo em todo bloco é o que importa, não os nomes das funções.

## 6. Escrevendo as conclusões: a causa, não só o número

A parte mais fácil de fazer mal num relatório de ablação é parar em "a configuração X teve Y% de
acurácia". Isso é uma observação, não uma conclusão. Para cada resultado relevante, a pergunta a
responder é **por que isso aconteceu**, ancorada em algo verificável:

- Prefira números que sustentam mecanismo a números soltos: "o gap treino–validação subiu de 0,134 para
  0,144, e a melhor época foi de 7 para 15" é mais forte que "o dropout ajudou".
- Quando um resultado é contra-intuitivo ou nulo (ex.: "a otimização não rendeu ganho"), diga isso
  explicitamente como um achado legítimo, não como uma falha do experimento.
- Separe estágios: **o que aconteceu** (o número), **por que aconteceu** (o mecanismo, apoiado em outra
  métrica ou numa curva), e **o que isso implica** para a próxima decisão do estudo.
- Marque explicitamente as ressalvas: um resultado limitado pelo orçamento de épocas/iterações, uma
  configuração que não convergiu, um efeito pequeno demais para distinguir de ruído entre execuções
  repetidas. Essas ressalvas são parte do rigor do estudo, não uma fraqueza a esconder.
- Ao comparar duas redes/algoritmos diferentes (não só duas configurações do mesmo), leve a comparação
  ao nível de subgrupo (classe, segmento, tipo de erro) — é aí que normalmente mora a explicação de
  *por que* um é melhor, não só *quanto* é melhor.

## 7. Checklist para começar um novo estudo com este método

1. **Defina o objetivo e a métrica de decisão** (a que otimiza a seleção de hiperparâmetros) e a(s)
   métrica(s) de relato (as que aparecem no resultado final, que podem ser mais ricas que a de decisão).
2. **Defina a unidade de repetição** (fold, seed, sub-amostra) e fixe as partições/seeds antes do
   primeiro experimento, para toda comparação pareada ser possível depois.
3. **Escreva os blocos**: para cada um, uma pergunta, os hiperparâmetros fixos herdados, o grid a testar,
   e o que ele herda do bloco anterior.
4. **Decida quantas repetições vão para a triagem e quantas finalistas avançam para a confirmação.**
5. **Implemente (ou reaproveite) a persistência local** dos quatro artefatos da seção 4, robusta a
   interrupção e retomável.
6. **Implemente (ou reaproveite) o ferramental de análise** da seção 5 antes de rodar o primeiro bloco
   grande — ele paga o investimento já no Bloco 0/1.
7. **Rode os blocos em sequência**, escrevendo a análise causal de cada um (seção 6) antes de passar ao
   próximo — a interpretação de um bloco frequentemente motiva o bloco complementar seguinte.
8. **Reserve o teste** para o final: revele-o só para os campeões e as referências, uma única vez.
9. **Feche com uma checagem de interação** entre as decisões mais distantes no tempo (ex.: a estrutura
   escolhida no Bloco 1 ainda é a melhor sob a regularização do Bloco 3?) sempre que a suspeita fizer
   sentido — não assuma que decisões antigas continuam válidas.

## 8. Erros que este método existe para evitar

| Erro | Como este método evita |
|---|---|
| Escolher hiperparâmetro "no olho" | Todo valor vem de um grid testado; o campeão de um bloco é lido automaticamente pelo seguinte. |
| Vazamento de teste na seleção | Toda escolha usa só validação; teste revelado uma única vez, no fim. |
| Maldição do vencedor | Triagem barata + confirmação completa das finalistas antes de declarar campeão. |
| Achar que uma decisão antiga continua ótima | Checagem de interação explícita entre blocos distantes. |
| Confundir ruído com efeito real | Partições fixas + comparação pareada por partição ("venceu em N de K"). |
| Resultado limitado pelo orçamento (poucas épocas/iterações) disfarçado de "essa configuração é pior" | Reexecutar com mais orçamento configurações suspeitas antes de concluir. |
| Perder os resultados por causa de uma ferramenta externa (nuvem, sessão) | Persistência local completa em primeiro lugar; ferramenta remota é só espelho. |
| Relatório que só lista números | Toda conclusão relevante busca o mecanismo causal por trás do número. |
