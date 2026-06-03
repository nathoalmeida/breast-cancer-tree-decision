import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



colunas = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

df = pd.read_csv(
    "wdbc.data",
    header=None,
    names=colunas
)

## ETAPA 1 - classificação dos dados 
df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})

## X = variáveis de entrada, todos os atributos menos o diagnóstico
## y = diagnóstico, variável de saída que estamos tentando prever

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

## TREINO E TESTE
## divide em 80% do dataset pra treino e 20% pra teste

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

## ÁRVORES DE DECISÃO 
arvores = [
    ## Mais complexa
{
"nome": "Arvore 1",
"criterion": "gini",
"max_depth": None,
"min_samples_leaf": 5,
"min_samples_split": 2 
},
{
"nome": "Arvore 2",
"criterion": "gini",
"max_depth": 5,
"min_samples_leaf": 10,
"min_samples_split": 5
},
{
"nome": "Arvore 3",
"criterion": "entropy",
"max_depth": None,
"min_samples_leaf": 10,
"min_samples_split": 5
},
    ## Muito simples
{
"nome": "Arvore 4",
"criterion": "entropy",
"max_depth": 3,
"min_samples_leaf": 20,
"min_samples_split": 10
}
]

resultados = []

for config in arvores:

    arvore = DecisionTreeClassifier(
    criterion=config["criterion"],
    max_depth=config["max_depth"],
    min_samples_leaf=config["min_samples_leaf"],
    min_samples_split=config["min_samples_split"],
    random_state=42
    )

    arvore.fit(X_train, y_train)

    y_pred = arvore.predict(X_test)

    resultados.append({
    "Árvore": config["nome"],
    "Critério": config["criterion"],
    "Profundidade": config["max_depth"],
    "Min Leaf": config["min_samples_leaf"],
    "Min Split": config["min_samples_split"],
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred)
    })

resultados_df = pd.DataFrame(resultados)

print(resultados_df)

resultados_df.to_csv("resultados.csv", index=False) 

## Compara resultados em um gráfico 

resultados_df.set_index("Árvore")[
["Accuracy", "Precision", "Recall"]
].plot(kind="bar")

plt.title("Comparação das Árvores")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.show() 


## plotar
for config in arvores:

    arvore = DecisionTreeClassifier(
        criterion=config["criterion"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        min_samples_split=config["min_samples_split"],
        random_state=42
    )

    arvore.fit(X_train, y_train)

    
    dot_data = export_graphviz(
        arvore,
        out_file=None,
        feature_names=X.columns,
        class_names=["Benigno", "Maligno"],
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)

    nome_arquivo = config["nome"].replace(" ", "_")

    graph.render(
        filename=nome_arquivo,
        format="pdf",
        cleanup=True
    )

    y_pred = arvore.predict(X_test)

    resultados.append({
        "Árvore": config["nome"],
        "Critério": config["criterion"],
        "Profundidade": config["max_depth"],
        "Min Leaf": config["min_samples_leaf"],
        "Min Split": config["min_samples_split"],
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred)
    })


## matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benigno", "Maligno"]
    )

    disp.plot()

    plt.title(f"Matriz de Confusão - {config['nome']}")

    plt.savefig(
        f"matriz_{config['nome'].replace(' ', '_')}.png",
        bbox_inches="tight"
    )

    plt.close()

