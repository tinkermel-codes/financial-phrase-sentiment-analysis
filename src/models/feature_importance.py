import pandas as pd


def get_feature_importance(pipeline, top_n):
    clf = pipeline.named_steps["clf"]
    tfidf = pipeline.named_steps["tfidf"]

    if not hasattr(clf, "coef_"):
        return None 

    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]

    top_idx = coefs.argsort()[-top_n:]

    return pd.DataFrame({"feature": feature_names[top_idx], "coefficient": coefs[top_idx]}).sort_values("coefficient", ascending=False)
