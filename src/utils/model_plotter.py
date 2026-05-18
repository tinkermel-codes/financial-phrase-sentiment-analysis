import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

class ModelPlotter:
    def __init__(self, repo):
        self.repo = repo

    
    def plot_macro_metrics(self,
                           df_metrics,
                           model_colors,
                           figsize=(10,6)):
        df_melted = df_metrics.melt(
            id_vars="model",
            value_vars=["macro_precision", "macro_recall", "macro_f1"],
            var_name="metric",
            value_name="score"
            )
        
        df_melted["metric"] = (df_melted["metric"].str.replace("_", " ").str.title())

        plt.figure(figsize=figsize)
        sns.barplot(data=df_melted, x="model", y="score", hue="metric", palette=model_colors)
        plt.title("Macro Metrics per Model")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.ylim(0,1)
        plt.tight_layout()
        plt.show()

    
    def compute_angles(self, n_metrics):
        angles = [n / float(n_metrics) * 2 * np.pi for n in range(n_metrics)]
        return angles + angles[:1]
    

    def pretty_labels(self, metric_cols):
        return [col.replace("macro_", "").replace("_", " ").title() for col in metric_cols]
    

    def place_labels(self, angles, pretty_labels, ax):
        for angle, label in zip(angles[:-1], pretty_labels):
            angle_deg = np.degrees(angle)
            r_max = ax.get_rmax()

            ha = "left" if -90 <= angle_deg <= 90 else "right"
            va = "bottom" if 0 < angle_deg < 180 else "top"

            ax.text(angle, r_max * 1.05, label, ha=ha, va=va)


    def plot_radar_chart(self,
                         df_metrics,
                         model_colors,
                         metric_cols=["macro_precision", "macro_recall", "macro_f1"],
                         figsize=(5,5),
                         title="Macro Metrics - Radar Chart"
                         ):
        angles = self.compute_angles(len(metric_cols))
        pretty_labels = self.pretty_labels(metric_cols)

        plt.figure(figsize=figsize)

        for _, row in df_metrics.iterrows():
            values = row[metric_cols].tolist()
            values += values[:1]

            plt.polar(
                angles,
                values,
                color= model_colors[row["model"]],
                linewidth=2,
                label=row["model"]
            )

        ax = plt.gca()
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])

        self.place_labels(angles, pretty_labels, ax)

        plt.legend(loc='lower right', bbox_to_anchor=(1.2, -0.05))
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


    def plot_class_performances(self, df_class_metrics, model_colors, figsize=(10,6)):
        df_melted = df_class_metrics.melt(
            id_vars=["model", "class"],
            value_vars=["precision", "recall", "f1"],
            var_name="metric",
            value_name="score"
            )
        
        plt.figure(figsize=figsize)
        sns.barplot(data=df_melted[df_melted["metric"] == "f1"], x="class", y="score", hue="model", palette=model_colors)
        plt.title("Class-Level Performance per Model")
        plt.xlabel("Class")
        plt.ylabel("F1 Score")
        plt.ylim(0,1.1)
        plt.tight_layout()
        plt.show()


    def create_gird(self, n_plots, min_per_row=3):
        cols = min(min_per_row, n_plots)
        rows = math.ceil(n_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        return fig, axes, rows, cols
    

    def plot_single_confusion_matrix(self, ax, model_name, cm_df, cmap):
        sns.heatmap(cm_df, annot=True, cmap=cmap, ax=ax, vmin=0, vmax=1)
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    def plot_confusion_matrices(self, cm_list, cmap="magma"):
        fig, axes, rows, cols = self.create_gird(len(cm_list))

        idx = 0
        for row in range(rows):
            for col in range(cols):
                ax = axes[row][col]

                if idx < len(cm_list):
                    model_name, cm_df = cm_list[idx]
                    self.plot_single_confusion_matrix(ax, model_name, cm_df, cmap)
                else:
                    ax.axis("off")

                idx += 1
        
        plt.suptitle("Confusion Matrices")
        plt.tight_layout()
        plt.show()


    def plot_single_learning_curve(self, ax, model_name, lc_df):
        ax.plot(lc_df["train_size"], lc_df["train_score_mean"], label="Train")
        ax.plot(lc_df["train_size"], lc_df["val_score_mean"], linestyle="--", label="Val")

        ax.set_title(model_name)
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("Macro F1 Score")
        ax.legend(loc= "lower right")
        ax.set_ylim(0,1.1)


    def plot_learning_curves(self, lc_list):
        fig, axes, rows, cols = self.create_gird(len(lc_list))

        idx = 0
        for row in range(rows):
            for col in range(cols):
                ax = axes[row][col]

                if idx < len(lc_list):
                    model_name, lc_df = lc_list[idx]
                    self.plot_single_learning_curve(ax, model_name, lc_df)
                else:
                    ax.axis("off")

                idx += 1
        plt.tight_layout()
        plt.show()


    def plot_single_misclassification(self, ax, model_name, mscls_df, normalize=True, vmin=None, vmax=None, cmap="Reds"):
        if normalize:
            error_matrix = pd.crosstab(mscls_df["true"], mscls_df["pred"], normalize="index")

        else:
            error_matrix = pd.crosstab(mscls_df["true"], mscls_df["pred"])

        sns.heatmap(
            error_matrix,
            annot=True,
            cmap=cmap,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            fmt="d" if not normalize else ".2f"
            )
        
        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")


    def plot_misclassifications(self, mscls_list, same_scale = True, normalize = True, cmap="Reds"):
        fig, axes, rows, cols = self.create_gird(len(mscls_list))

        if same_scale:
            if normalize:
                vmin, vmax = 0, 1
            else:
                vmin = 0
                vmax = vmax = max(pd.crosstab(mscls_df["true"], mscls_df["pred"]).values.max() for model_name, mscls_df in mscls_list)
        else:
            vmin = vmax = None

        idx = 0
        for row in range(rows):
            for col in range(cols):
                ax = axes[row][col]

                if idx < len(mscls_list):
                    model_name, mscls_df = mscls_list[idx]
                    self.plot_single_misclassification(ax, model_name, mscls_df, normalize=normalize, vmin=vmin, vmax=vmax, cmap=cmap)
                else:
                    ax.axis("off")

                idx += 1
        plt.tight_layout()
        plt.show()


    def calculate_error_rates(self, mscls_list, class_counts):
        rows = []
        
        for model_name, mscls_df in mscls_list:
            error_counts = mscls_df["true"].value_counts()
            error_rate = (error_counts / class_counts).fillna(0)

            for cls , rate in error_rate.items():
                rows.append({
                    "model":model_name,
                    "class":cls,
                    "error_rate":rate
                })
            
        return pd.DataFrame(rows)
        

    def plot_error_rates(self, mscls_list, class_counts, colors, figsize=(6,4)):
        error_rates_df = self.calculate_error_rates(mscls_list=mscls_list, class_counts=class_counts)

        plt.figure(figsize=figsize)
        sns.barplot(
            data=error_rates_df,
            x="class",
            y="error_rate",
            hue="model",
            palette=colors
            )

        plt.title("Class-wise Error Rates per Model")
        plt.xlabel("Class")
        plt.ylabel("Error Rate")
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.show()


    def plot_model_sizes(self, df, metric, ax, model_colors):
        sns.scatterplot(
            data=df,
            x="model_size_kb",
            y=metric,
            hue="model",
            palette=model_colors,
            ax=ax
            )
    
        ax.set_title("Model Size vs Performance")
        ax.set_xlabel("Model Size (KB)")
        ax.set_ylabel(metric)


    def plot_model_efficiency(self, df, metric, ax, model_colors):
        sns.barplot(
            data=df,
            x="model",
            y="efficiency",
            hue="model",
            palette=model_colors,
            ax=ax
            )
        ax.set_title("Model Efficiency: Performance per KB")
        ax.set_ylabel(f"Efficiency ({metric} / KB)")
        ax.set_xlabel("Model")


    def plot_model_sizes_and_efficiencies(self, metrics_df, sizes_df, metric, model_colors, figsize=(14,5)):
        df = metrics_df.merge(sizes_df, on="model")
        df["efficiency"] = df[metric] / df["model_size_kb"]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        self.plot_model_sizes(df, metric, axes[0], model_colors)
        self.plot_model_efficiency(df, metric, axes[1], model_colors)

        plt.tight_layout()
        plt.show()


        


