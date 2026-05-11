# File for visualisaitions of phase 2 results, including overall performance, per-language and per-emotion breakdowns, and class distributions.
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_bar_chart(df, x_col, y_col, title, xlabel, ylabel, output_path, rotation=0):
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, max(1.0, float(df[y_col].max()) + 0.1))
    plt.xticks(rotation=rotation, ha='right' if rotation else 'center')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_grouped_bar_chart(df, x_col, group_col, y_col, title, xlabel, ylabel, output_path, rotation=0):
    pivot = df.pivot(index=x_col, columns=group_col, values=y_col)
    ax = pivot.plot(kind='bar', figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, float(df[y_col].max()) + 0.1))
    ax.legend(title=group_col)
    plt.xticks(rotation=rotation, ha='right' if rotation else 'center')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_visualisations(results_dir: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)

    summary = pd.read_csv(results_dir / 'phase2_summary.csv')
    logreg_lang = pd.read_csv(results_dir / 'logreg_test_per_language.csv')
    svm_lang = pd.read_csv(results_dir / 'svm_test_per_language.csv')
    logreg_emotion = pd.read_csv(results_dir / 'logreg_test_per_emotion.csv')
    svm_emotion = pd.read_csv(results_dir / 'svm_test_per_emotion.csv')

    summary['model'] = summary['model'].replace({'logreg': 'Logistic Regression', 'svm': 'SVM'})

    logreg_lang['model'] = 'Logistic Regression'
    svm_lang['model'] = 'SVM'
    lang_combined = pd.concat([logreg_lang, svm_lang], ignore_index=True)

    logreg_emotion['model'] = 'Logistic Regression'
    svm_emotion['model'] = 'SVM'
    emotion_combined = pd.concat([logreg_emotion, svm_emotion], ignore_index=True)

    save_grouped_bar_chart(
        summary,
        x_col='model',
        group_col='split',
        y_col='macro_f1',
        title='Overall Macro F1: Validation vs Test',
        xlabel='Model',
        ylabel='Macro F1',
        output_path=output_dir / 'overall_macro_f1_val_test.png'
    )

    test_summary = summary[summary['split'] == 'test'].copy()
    save_bar_chart(
        test_summary,
        x_col='model',
        y_col='macro_f1',
        title='Test Macro F1 by Baseline Model',
        xlabel='Model',
        ylabel='Macro F1',
        output_path=output_dir / 'test_macro_f1_by_model.png'
    )

    save_grouped_bar_chart(
        lang_combined,
        x_col='language',
        group_col='model',
        y_col='macro_f1',
        title='Test Macro F1 per Language',
        xlabel='Language',
        ylabel='Macro F1',
        output_path=output_dir / 'test_macro_f1_per_language.png'
    )

    save_grouped_bar_chart(
        lang_combined,
        x_col='language',
        group_col='model',
        y_col='micro_f1',
        title='Test Micro F1 per Language',
        xlabel='Language',
        ylabel='Micro F1',
        output_path=output_dir / 'test_micro_f1_per_language.png'
    )

    possible_emotion_cols = ['emotion', 'label', 'class']
    emotion_col = next((c for c in possible_emotion_cols if c in emotion_combined.columns), None)
    if emotion_col is None:
        raise ValueError(f'Could not find emotion column. Found: {list(emotion_combined.columns)}')

    metric_col = 'f1' if 'f1' in emotion_combined.columns else 'macro_f1'

    save_grouped_bar_chart(
        emotion_combined,
        x_col=emotion_col,
        group_col='model',
        y_col=metric_col,
        title='Test F1 per Emotion Class',
        xlabel='Emotion',
        ylabel='F1-score',
        output_path=output_dir / 'test_f1_per_emotion.png',
        rotation=30
    )

    dist_path = results_dir / 'class_distribution_by_split_language.csv'
    if dist_path.exists():
        dist = pd.read_csv(dist_path)
        if {'split', 'language', 'emotion', 'positive_count'}.issubset(dist.columns):
            test_dist = dist[dist['split'] == 'test'].copy()
            for language in sorted(test_dist['language'].unique()):
                language_dist = test_dist[test_dist['language'] == language]
                save_bar_chart(
                    language_dist,
                    x_col='emotion',
                    y_col='positive_count',
                    title=f'Test Class Distribution: {language}',
                    xlabel='Emotion',
                    ylabel='Positive Label Count',
                    output_path=output_dir / f'class_distribution_test_{language}.png',
                    rotation=30
                )

    lang_combined.to_csv(output_dir / 'combined_test_per_language.csv', index=False)
    emotion_combined.to_csv(output_dir / 'combined_test_per_emotion.csv', index=False)

    print(f'Visualisations saved to: {output_dir}')
    for file in sorted(output_dir.glob('*')):
        print(f' - {file.name}')


def main():
    parser = argparse.ArgumentParser(description='Create Phase 2 result visualisations.')
    parser.add_argument('--results-dir', default='results/phase2_baselines')
    parser.add_argument('--output-dir', default='results/phase2_visualisations')
    args = parser.parse_args()
    create_visualisations(Path(args.results_dir), Path(args.output_dir))


if __name__ == '__main__':
    main()
