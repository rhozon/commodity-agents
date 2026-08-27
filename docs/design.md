# Analista de Commodities Multiagente — design técnico

**Data:** 2026-08-25
**Autor:** Rodrigo Hermont Ozon

LLM e econometria aplicada resolvem problemas diferentes: um lê texto e decide
o próximo passo, o outro ajusta modelo e mede incerteza. Juntar os dois exige
que o LLM nunca vire a fonte do número — ele decide qual função do núcleo
chamar e como contar o resultado, nunca calcula nada sozinho. Este sistema
responde a uma pergunta de negócio sobre preço de milho ou soja com um
relatório em três camadas (previsão, drivers, implicação de decisão),
produzido por quatro agentes que decidem, e por um núcleo econométrico em R
que calcula.

## Objetivos

1. Sistema que responde a uma pergunta de negócio sobre preço de commodity com um relatório
   em três camadas: previsão, drivers e implicação de decisão
2. Servir a dois usos sem que um sabote o outro: **ferramenta real** de análise (dados,
   modelo, backtest) e **demonstração técnica** legível e executável por terceiros
3. Rodar do zero, em máquina limpa, seguindo apenas o README

## Fora de escopo

- Dado de cliente, de qualquer natureza. Só fonte pública
- Execução em produção, agendada ou com serviço web. É biblioteca mais linha de comando
- Decisão automatizada de compra, venda ou hedge. O sistema informa; quem decide é humano

## Decisões tomadas

| Questão | Decisão |
|---|---|
| Propósito | Ferramenta real **e** vitrine |
| Saída | Relatório em três camadas: previsão, drivers, nota de decisão |
| Runtime | Python orquestra, R modela |
| Autonomia | Híbrido: pipeline fixo com laço de crítica e reprovação |
| Escopo | Milho e soja |
| Primeiro modelo | MSGARCH |

## Arquitetura: duas camadas

A dupla finalidade só não vira armadilha se as duas metades não competirem pelo mesmo código.

**Núcleo** — biblioteca Python comum, sem nada de agente. Baixa série, ajusta modelo, roda
backtest, gera gráfico. Determinística, testável, chamável de um notebook. É o que se reusa
fora deste projeto.

**Camada agêntica** — fina, por cima. Os agentes não sabem econometria: decidem **qual função
do núcleo chamar e com quais argumentos**, leem o resultado e escolhem o próximo passo.

Três ganhos: o núcleo se testa sem gastar API; a camada de agentes se lê em dez minutos
porque é pequena; e quando o framework de agentes envelhecer, troca-se a casca sem tocar no
que importa.

## Componentes

| Agente | Decide | Chama do núcleo |
|---|---|---|
| **Coletor** | quais séries a pergunta exige, e qual janela | `fetch_series()` |
| **Econometrista** | qual família de modelo e qual transformação | `fit_model()` (via R) |
| **Crítico** | se o ajuste presta — e **reprova, com motivo** | `diagnose()`, `backtest()` |
| **Redator** | como contar, em três camadas, para três leitores | `render_report()` |

O laço vive entre Econometrista e Crítico, com teto de **3 tentativas**, recuando pela escada
`msgarch → garch → arima` a cada reprovação. O relatório declara o recuo **sempre que houve
reprovação**, não só quando o teto estoura — cada tentativa reprovada entra na seção "Recuo de
modelo" com a família tentada e o motivo. Quando o teto **também** estoura, o relatório
acrescenta um aviso explícito de que o resultado é o melhor ajuste obtido, sem comparação de
AIC ou verossimilhança entre tentativas, e deve ser lido com reserva.

**Modelo inicial:** MSGARCH. Quando o Crítico reprova por não convergência (ou por premissa de
resíduo violada — ver "Testes" abaixo), o Econometrista recua para famílias mais simples
(GARCH, depois ARIMA).

## Fluxo de dados

| Série | Fonte | Papel |
|---|---|---|
| Milho CBOT (`ZC=F`) | Yahoo Finance | referência internacional |
| Soja CBOT (`ZS=F`) | Yahoo Finance | referência internacional |
| Milho ESALQ, Soja Paraná | CEPEA | preço doméstico |
| USD/BRL (`BRL=X`) | Yahoo Finance | ponte entre os dois |

O câmbio não é enfeite: sem ele, análise de preço doméstico de grão no Brasil está errada.

**A coleta CEPEA não está implementada nesta versão.** A função correspondente levanta erro de
propósito; o Coletor cai para a série internacional (Yahoo Finance) e registra a troca de
fonte no relatório. O contrato (fonte doméstica com câmbio como ponte) está desenhado e testado
do lado do consumidor — é a implementação da chamada ao CEPEA que falta.

**Cache em disco, em parquet, com carimbo de data.** Roda uma vez, congela. Garante
reprodutibilidade — o gráfico publicado não muda sozinho quando o mercado mexe —, zera o
custo de re-execução e permite que o teste de ponta a ponta rode sem rede.

Uma execução por commodity. A segunda série é entrada de configuração, não código novo.

## Contratos entre as peças

Cada agente recebe e devolve **objeto tipado**, não texto solto. O LLM preenche campos de um
esquema; fora do esquema, é rejeitado e re-pedido uma vez.

Na fronteira com o R, o contrato é **JSON por subprocess** — entrada com série e parâmetros,
saída com coeficientes, diagnósticos e métricas de backtest.

Sem `rpy2`, deliberadamente: é a fonte número um de dor de instalação em projeto R+Python, e
há objetivo explícito de "qualquer um clona e roda". Com subprocess, o contrato entre as
linguagens é um arquivo de texto, e o script R roda sozinho no RStudio quando for preciso
depurar o modelo.

## Backtest

Origem fixa, não rolling origin: o modelo é ajustado até `T - h` e projeta `h` passos de uma
vez (`h = 20` por padrão), medido lado a lado com o passeio aleatório (mesmo preço repetido).
MAPE sem referência não diz se o modelo presta — em preço de commodity o passeio aleatório é
um adversário difícil de bater. A cobertura e o MAPE medidos vêm de um único corte da série e
ilustram o comportamento do modelo, não o estimam com precisão estatística.

MSGARCH e GARCH têm média zero na especificação: a previsão de retorno é zero em qualquer
horizonte, por desenho. Isso empata o MAPE/RMSE do ponto previsto com o passeio aleatório; a
contribuição real de um modelo de volatilidade está na largura do intervalo de confiança,
calculada a partir da volatilidade condicional do último instante — não na variância
multi-passo do modelo (que reverteria à média de longo prazo num GARCH; essa reversão não está
capturada aqui). O campo `Backtest.nota` distingue por escrito esse empate por construção de um
refit que simplesmente não convergiu, que são causas opostas para o mesmo silêncio no número.

## Tratamento de erro

| Falha | Comportamento |
|---|---|
| Fonte fora do ar (ou não implementada, caso do CEPEA) | Coletor cai para a alternativa e **registra a troca** no relatório |
| Modelo não converge, ou premissa de resíduo violada | Crítico reprova; conta como tentativa |
| Teto de 3 tentativas | Relatório com o melhor ajuste **e aviso explícito** |
| LLM devolve esquema inválido | Um novo pedido; na segunda falha, erro explícito |

## Trava anti-alucinação

**O Redator não pode escrever nenhum número que não exista no objeto de resultados.**

Uma verificação varre os números do texto gerado e confere contra os valores do núcleo. Número
inventado faz a execução falhar. A tolerância é pela **precisão escrita**, não por epsilon
fixo: um número com `k` casas decimais só passa se for arredondamento correto de algum valor
autorizado *nessa* precisão — escrever mais casas é mais exigente, não menos.

Num artefato que serve de prova de competência em IA, alucinação de número não é bug — é o fim
da credibilidade. A checagem é barata e cobre a maioria das alucinações reais; ela não cobre
número real com significado trocado (ex.: citar o RMSE como se fosse o MAPE), nem confere
direção ou unidade — só proveniência de dígito.

## Testes

Três níveis, nenhum gasta API:

1. **Núcleo** — série sintética de propriedades conhecidas. Gera-se processo com mudança de
   regime de volatilidade e verifica-se se o MSGARCH recupera o regime
2. **Camada agêntica** — LLM mockado, respostas fixas, cobrindo caminho feliz e laço de
   reprovação
3. **Fumaça, ponta a ponta** — cache congelado, sem rede. É o teste que garante que quem
   clonar consegue rodar

## Verificação

- `pytest` verde nos três níveis, sem rede e sem chave de API
- Execução completa para milho e para soja, a partir do cache congelado, gerando os dois
  relatórios de exemplo
- Forçar reprovação do Crítico (série degenerada) e conferir que o laço tenta de novo, recua
  de modelo e declara o recuo no relatório
- Injetar número falso na saída do Redator e conferir que a trava anti-alucinação derruba a
  execução
- Clonar em pasta limpa e rodar seguindo apenas o README
