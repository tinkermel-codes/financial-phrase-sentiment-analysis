import matplotlib.pyplot as plt


def get_feature_importance(pipeline, top_n, figsize, plot_title):
    clf = pipeline.named_steps["clf"]
    tfidf = pipeline.named_steps["tfidf"]

    if not hasattr(clf, "coef_"):
        return None 

    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]

    top_idx = coefs.argsort()[-top_n:]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(feature_names[top_idx], coefs[top_idx])
    ax.set_title(plot_title)
    plt.tight_layout()

    return fig
