from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import learning_curve

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    cm_df = pd.DataFrame(cm, index=pipeline.named_steps["clf"].classes_, columns=pipeline.named_steps["clf"].classes_)
    return report_dict, cm_df


def compute_learning_curve(
        pipeline,
        X,
        y,
        cv=5,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        scoring="f1_macro"
        ):
    
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring
        )
    
    return pd.DataFrame({
        "train_size": train_sizes,
        "train_score_mean": train_scores.mean(axis=1),
        "train_score_std": train_scores.std(axis=1),
        "val_score_mean": val_scores.mean(axis=1),
        "val_score_std": val_scores.std(axis=1)
    })