from model import Model, ModelHF
import pandas as pd

def gerar_prompt(chave):
    return f"""
{associacoes}

Recupere a palavra associada ao código solicitado.
Responda somente com a palavra.

Código: {chave}
Palavra:"""



modelo_transformer = Model(
    "/media/data/matheusvieira/candeia-transformer-350M/model.pt",
    "/media/data/matheusvieira/candeia-transformer-350M/tokenizer.model",
    "transformer"
)

modelo_xlstm = Model(
    "/media/data/matheusvieira/candeia-xlstm-350M/model.pt",
    "/media/data/matheusvieira/candeia-xlstm-350M/tokenizer.model",
    "xlstm"
)

manaca = ModelHF("/media/work/matheusvieira/models/models/menezesbruno/manaca-1b-base")

for k in [8, 32, 64]:
    print(f"K = {k}")
    
    df = pd.read_csv(f"../corpus/independent_association/samples_K={k}.csv")

    associacoes = "\n".join(
        f"{row.keys}: {row.words}"
        for row in df.itertuples(index=False)
    )



    resultados = []

    for row in df.itertuples(index=False):
        prompt = gerar_prompt(row.keys)

        resposta_xlstm = modelo_xlstm.make_inference(
            prompt, max_new_tokens=16
        )

        resposta_transformer = modelo_transformer.make_inference(
            prompt, max_new_tokens=16
        )
        
        respsota_manaca = manaca.make_inference(
            prompt, max_new_tokens=16
        )

        resultados.append({
            "chave": row.keys,
            "resposta_esperada": row.words,
            "xlstm": resposta_xlstm,
            "transformer": resposta_transformer,
            "manaca": respsota_manaca
        })

    pd.DataFrame(resultados).to_csv(
        f"resultados_associacao_k={k}.csv",
        index=False,
        encoding="utf-8"
    )
