"""Escreve o corpo do relatorio, sob a trava anti-alucinacao.

O prompt e a PRIMEIRA linha de defesa, nao a trava (`agro.guard`): a trava
so confere se cada digito citado tem PROVENIENCIA em algum resultado do
nucleo -- ela nao confere se a grandeza que o texto atribui a esse digito e
a mesma grandeza de onde ele veio (ver secao LIMITACOES na docstring de
`agro.guard`). "o preco subiu 3%" passa a trava se `backtest.rmse = 3.21`
for o unico valor autorizado perto de 3, mesmo RMSE e "preco subiu X%" nao
tendo nenhuma relacao entre si. E exatamente esse buraco -- vinculo
semantico, nao proveniencia de digito -- que o prompt precisa fechar antes
de o texto chegar na trava: por isso os numeros vao para o LLM COM ROTULO
(`RunResult.valores_rotulados()`), nao como uma lista crua de floats.

O esquema tem UM CAMPO POR CAMADA (previsao, drivers, implicacao). Com um
campo unico, o relatorio tinha tres titulos e uma camada so: as secoes
"Drivers" e "Implicacao de decisao" eram carimbos fixos apontando de volta
para a primeira. O LLM ja escrevia os tres paragrafos; separa-los no esquema
e o que faz cada um chegar na sua secao.

A trava roda sobre `CorpoRelatorio.texto_completo()`: as tres camadas
juntas, porque as tres sao texto de LLM. O agente chama
`guard.verificar_numeros` e deixa `NumeroInventado` / `NumeroAmbiguo`
subirem sem capturar: quem decide o que fazer com a falha (nova tentativa,
aborto, etc.) e o orquestrador, nao este agente.
"""
from agents.llm import LLM
from agro import guard
from agro.types import CorpoRelatorio, RunResult

ESQUEMA = {"previsao": str, "drivers": str, "implicacao": str, "confianca": str}


class Redator:
    def __init__(self, llm: LLM):
        self._llm = llm

    def escrever(self, res: RunResult) -> CorpoRelatorio:
        rotulados = res.valores_rotulados()
        lista_numeros = "\n".join(f"  - {rotulo} = {valor:g}" for rotulo, valor in rotulados)
        prompt = (
            f"Voce e o agente Redator.\nCommodity: {res.commodity}. "
            f"Pergunta: {res.pergunta!r}\n"
            f"Modelo ajustado: {res.fit.familia}, convergiu={res.fit.convergiu}.\n"
            f"Tentativas: {res.tentativas}. Teto estourado: {res.teto_estourado}.\n\n"
            "Numeros disponiveis, com o rotulo de onde cada um vem "
            "(use o rotulo para saber a QUE grandeza o numero se refere; "
            "nao troque o rotulo nem atribua um numero a uma grandeza diferente "
            "da que ele rotula):\n"
            f"{lista_numeros}\n\n"
            "Escreva o corpo do relatorio em portugues, em TRES CAMPOS "
            "SEPARADOS, um paragrafo cada:\n"
            "- previsao: o que o modelo ajustado diz sobre o preco, com o "
            "alcance e os limites do que ele consegue afirmar.\n"
            "- drivers: o que move a serie -- quais series entraram, o que o "
            "diagnostico e a volatilidade estimada indicam.\n"
            "- implicacao: o que isso significa para uma decisao, lembrando "
            "que o sistema informa e quem decide e humano.\n\n"
            "REGRAS ABSOLUTAS (valem para os tres campos):\n"
            "1. Use SOMENTE numeros da lista acima, e somente com o rotulo "
            "correspondente -- nunca cite um numero como se fosse outra grandeza.\n"
            "2. Nunca invente valor, percentual ou projecao numerica que nao "
            "esteja na lista. Se a lista nao trouxer um numero que voce "
            "gostaria de citar (por exemplo uma variacao percentual futura "
            "de preco que ninguem calculou), NAO o escreva -- descreva a "
            "direcao ou o fato qualitativamente, sem inventar o digito.\n"
            "3. Nao projete valores futuros de preco, retorno ou volatilidade "
            "que nao estejam entre os numeros disponiveis: o modelo pode nao "
            "produzir previsao pontual (familias de volatilidade tem media "
            "zero por construcao), e citar uma projecao que nao existe e "
            "alucinacao."
        )
        resposta = self._llm.perguntar(prompt, ESQUEMA)
        corpo = CorpoRelatorio(previsao=resposta["previsao"],
                               drivers=resposta["drivers"],
                               implicacao=resposta["implicacao"])
        # Levanta NumeroInventado/NumeroAmbiguo. Roda sobre as tres camadas
        # juntas: nenhuma escapa da trava por estar em outro campo.
        guard.verificar_numeros(corpo.texto_completo(), res)
        return corpo
