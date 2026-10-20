from model import Model, ModelHF
import pandas as pd


# ============================================================
# CONFIGURAÇÃO DOS MODELOS
# ============================================================

MODELOS = {
    "transformer": Model(
        "/media/data/matheusvieira/candeia-transformer-350M/model.pt",
        "/media/data/matheusvieira/candeia-transformer-350M/tokenizer.model",
        "transformer"
    ),

    "xlstm": Model(
        "/media/data/matheusvieira/candeia-xlstm-350M/model.pt",
        "/media/data/matheusvieira/candeia-xlstm-350M/tokenizer.model",
        "xlstm"
    ),

    "manaca": ModelHF(
        "/media/work/matheusvieira/models/models/menezesbruno/manaca-1b-base"
    ),
    
    "tucano": ModelHF(
        "/media/work/matheusvieira/models/models/TucanoBR/Tucano-2b4"
    ),
    
    "sabia": ModelHF(
        "/media/work/matheusvieira/models/models/maritaca-ai/sabia-7b"
    ),
    
    "llama": ModelHF(
        "/media/work/matheusvieira/models/models/meta-llama/Meta-Llama-3.1-8B"
    ),
    
    "qwen": ModelHF(
        "/media/work/matheusvieira/models/models/Qwen/Qwen3.5-9B-Base"
    )
    
}


# ============================================================
# CONFIGURAÇÕES DO EXPERIMENTO
# ============================================================

ARQUIVO_ENTRADA = "/media/work/matheusvieira/xLSTM_vs_Transformer/corpus/simple_information_recovery/recuperacao_informacao_20.tsv"
ARQUIVO_SAIDA = "resultados_recuperacao_informacao.tsv"

MAX_NEW_TOKENS = 20


# ============================================================
# GERAÇÃO DO PROMPT
# ============================================================

def gerar_prompt(frase):
    return f"""{frase}

Responda somente com a resposta solicitada, sem explicações.
Resposta:"""


# ============================================================
# EXECUÇÃO DO EXPERIMENTO
# ============================================================
import re

def normalizar_resposta(resposta):
    if resposta is None:
        return ""

    return re.sub(r"\s+", " ", str(resposta)).strip()


def executar_experimento():

    df = pd.read_csv(
        ARQUIVO_ENTRADA,
        sep="\t",
        encoding="utf-8"
    )

    for indice, row in enumerate(df.itertuples(index=False)):

        prompt = gerar_prompt(row.frase)

        resultado = {
            "id": row.id,
            "dificuldade": row.dificuldade,
            "quantidade_itens": row.quantidade_itens,
            "frase": normalizar_resposta(row.frase),
            "resposta_esperada": normalizar_resposta(row.resposta)
        }

        for nome, modelo in MODELOS.items():

            resposta = modelo.make_inference(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS
            )

            resultado[nome] = normalizar_resposta(resposta)

        pd.DataFrame([resultado]).to_csv(
            ARQUIVO_SAIDA,
            mode="w" if indice == 0 else "a",
            header=(indice == 0),
            index=False,
            encoding="utf-8",
            sep="\t"
        )

        print(f"Concluído: {indice + 1}/{len(df)}")

    print("\nExperimento finalizado!")


if __name__ == "__main__":
    executar_experimento()