import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_score
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

iris = load_iris()
x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, 
                                                    stratify=y)
folds = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

best_depth = None
best_score = -1
depths = list(range(1, 11))
training_means = []
validation_means = []
validation_folds = []

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    results = cross_validate(
        model, X_train, y_train,
        cv=folds,
        scoring="accuracy",
        return_train_score=True
    )

    scores = results["test_score"]  # Validation folds, not the held-out test set.
    average = scores.mean()
    training_means.append(results["train_score"].mean())
    validation_means.append(average)
    validation_folds.append(scores)
    print(f"Depth {depth}: average validation accuracy = {average:.1%}")

    if average > best_score:
        best_score = average
        best_depth = depth

# Train the chosen model using all the training data.
model = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42
)

model.fit(X_train, y_train)

# Finally, evaluate on the held-out test data.
predictions = model.predict(X_test)

for measurements, actual, predicted in zip(X_test, y_test, predictions):
    if actual != predicted:
        print("Measurements:", measurements)
        print("Actual species:", iris.target_names[actual])
        print("Predicted species:", iris.target_names[predicted])
        print()

forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

# Evaluate using the same folds as the decision tree.
forest_scores = cross_val_score(
    forest,
    X_train,
    y_train,
    cv=folds,
    scoring="accuracy"
)

print(f"Tree validation accuracy:   {best_score:.1%}")
print(f"Forest validation accuracy: {forest_scores.mean():.1%}")

# Train on all 120 training flowers.
forest.fit(X_train, y_train)

forest_predictions = forest.predict(X_test)


print(f"Tree test accuracy:   {accuracy_score(y_test, predictions):.1%}")
print(f"Forest test accuracy: {accuracy_score(y_test, forest_predictions):.1%}")

print([tree.get_depth() for tree in forest.estimators_])

print(np.array_equal(predictions, forest_predictions))

# Scale measurements so larger-valued features don't dominate distance.
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train)
scaled_test = scaler.transform(X_test)

# Find each species' average measurements.
species_averages = np.array([
    scaled_train[y_train == species].mean(axis=0)
    for species in range(3)
])

for i in range(len(y_test)):
    if predictions[i] != y_test[i]:
        distances = np.linalg.norm(
            species_averages - scaled_test[i],
            axis=1
        )

        print("Actual species:", iris.target_names[y_test[i]])

        for name, distance in zip(iris.target_names, distances):
            print(f"Distance to {name} average: {distance:.2f}")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

best_k = None
best_knn_score = -1

for k in range(1, 21):
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )

    scores = cross_val_score(
        knn, X_train, y_train,
        cv=folds,
        scoring="accuracy"
    )

    average = scores.mean()
    print(f"k={k}: validation accuracy = {average:.1%}")

    if average > best_knn_score:
        best_knn_score = average
        best_k = k

# Train the chosen model on all training flowers.
knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=best_k)
)
knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)

print(f"Chosen k: {best_k}")
print(f"Test accuracy: {accuracy_score(y_test, knn_predictions):.1%}")

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

print(f"KNN accuracy: {accuracy_score(y_test, knn_predictions):.1%}")

# Compare depths using only training data and its cross-validation folds.
fig, ax = plt.subplots(figsize=(10, 6))
fold_scores = np.array(validation_folds)
for fold in range(5):
    ax.plot(
        depths, fold_scores[:, fold], color="orange", alpha=0.25,
        label="Individual validation folds" if fold == 0 else None,
    )
ax.plot(depths, training_means, "o-", color="royalblue",
        label="Average training accuracy")
ax.plot(depths, validation_means, "o-", color="darkorange",
        label="Average validation accuracy")
ax.axvline(best_depth, color="gray", linestyle="--",
           label=f"Chosen depth: {best_depth}")
ax.scatter(best_depth, accuracy_score(y_test, predictions),
           color="crimson", marker="*", s=200, zorder=5,
           label="Final held-out test accuracy")
ax.set(title="Choosing tree depth with 5-fold cross-validation",
       xlabel="Maximum tree depth", ylabel="Accuracy")
ax.set_xticks(depths)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
