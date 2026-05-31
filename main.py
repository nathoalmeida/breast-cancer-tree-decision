import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
resultados = []

for criterion in ["gini", "entropy"]:
    for max_depth in [3, 5, 7, None]:
        for min_samples_leaf in [1, 5]:
            for min_samples_split in [2, 10]:

                arvore = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    random_state=42
                )

                arvore.fit(X_train, y_train)

                y_pred = arvore.predict(X_test)

                resultados.append({
                    "criterion": criterion,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "min_samples_split": min_samples_split,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred)
                })

resultados_df = pd.DataFrame(resultados)

print(
    resultados_df.sort_values(
        by="accuracy",
        ascending=False
    )
)

resultados_df.to_csv("resultados.csv", index=False)
