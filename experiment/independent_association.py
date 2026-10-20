
from model import Model, ModelHF
import pandas as pd
import re


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

DIRETORIO_ENTRADA = "../corpus/independent_association"

VALORES_K = [8, 32, 64]

MAX_NEW_TOKENS = 16


# ============================================================
# GERAÇÃO DO PROMPT
# ============================================================

def gerar_prompt(chave, associacoes):
    return f"""{associacoes}

Recupere a palavra associada ao código solicitado.
Responda somente com a palavra.

Código: {chave}
Palavra:"""


# ============================================================
# NORMALIZAÇÃO DAS RESPOSTAS
# ============================================================

def normalizar_resposta(resposta):
    if resposta is None:
        return ""

    return re.sub(r"\s+", " ", str(resposta)).strip()


# ============================================================
# EXECUÇÃO DO EXPERIMENTO
# ============================================================

def executar_experimento():

    for k in VALORES_K:

        print(f"\n{'=' * 60}")
        print(f"Executando experimento com K = {k}")
        print(f"{'=' * 60}")

        arquivo_entrada = (
            f"{DIRETORIO_ENTRADA}/samples_K={k}.csv"
        )

        arquivo_saida = (
            f"independent_association_K={k}.tsv"
        )

        # Carrega associações
        df = pd.read_csv(
            arquivo_entrada,
            encoding="utf-8",
            dtype={"keys": str, "words": str}
        )

        # Constrói a lista de associações
        associacoes = "\n".join(
            f"{row.keys}: {row.words}"
            for row in df.itertuples(index=False)
        )

        # Executa o experimento
        for indice, row in enumerate(df.itertuples(index=False)):

            prompt = gerar_prompt(
                row.keys,
                associacoes
            )

            resultado = {
                "chave": row.keys,
                "resposta_esperada": normalizar_resposta(row.words)
            }

            # Executa todos os modelos cadastrados
            for nome, modelo in MODELOS.items():

                resposta = modelo.make_inference(
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS
                )

                resultado[nome] = normalizar_resposta(resposta)

            # Salva resultados parciais em TSV
            pd.DataFrame([resultado]).to_csv(
                arquivo_saida,
                mode="w" if indice == 0 else "a",
                header=(indice == 0),
                index=False,
                encoding="utf-8",
                sep="\t"
            )

            print(
                f"K={k} | "
                f"Concluído: {indice + 1}/{len(df)}"
            )

        print(f"\nResultados salvos em: {arquivo_saida}")

    print("\nExperimento finalizado!")


if __name__ == "__main__":
    executar_experimento()
