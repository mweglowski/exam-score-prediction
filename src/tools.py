from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import os

def load_data():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    return df_train, df_test

def split_data(df):
    X = df.drop(['exam_score'], axis=1)
    y = df['exam_score']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=303)
    return X_train, X_val, y_train, y_val

def get_features_and_labels(df, target_col):
    X = df.drop([target_col, 'id'], axis=1)
    y = df[target_col]
    return X, y

def generate_experiment_history_graph(experiments_file_name, save_path):
    if os.path.exists(experiments_file_name):
        scores = []
        time_durations = []
        descriptions = []
        
        with open(experiments_file_name, 'r') as file:
            for line in file.readlines():
                line = line.strip()
                if line.startswith('**Mean'):
                    scores.append(float(line[16:]))
                elif line.startswith('**Time'):
                    time_durations.append(float(line[18:line.rindex('s')]))
                elif line.startswith('>'):
                    descriptions.append(line[2:])

        df = pd.DataFrame({
            'Experiment Index': range(len(scores)),
            'Score': scores,
            'Duration (s)': time_durations,
            'Description': descriptions
        })

        sns.set_theme('paper')
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.lineplot(data=df, x='Experiment Index', y='Score', color='gray', 
                     linestyle='--', alpha=0.5, ax=ax, legend=False, zorder=1)

        scatter = sns.scatterplot(
            data=df, 
            x='Experiment Index', 
            y='Score', 
            size='Duration (s)', 
            sizes=(50, 500),
            color='#3b82f6', 
            edgecolor='black',
            alpha=0.8,
            ax=ax,
            zorder=2,
            legend='brief'
        )

        for i in range(df.shape[0]):
            txt = df['Description'][i]
            x_pos = df['Experiment Index'][i]
            y_pos = df['Score'][i]
            
            label_text = (txt[:40] + '...') if len(txt) > 40 else txt
            
            ax.annotate(label_text, 
                        (x_pos, y_pos),
                        textcoords='offset points', 
                        xytext=(0, 12), 
                        ha='left', 
                        rotation=45,
                        fontsize=9,
                        color='#333333')

        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1.01, 1), 
                        title='Duration (Seconds)', frameon=True)

        ax.set_title('Experiments History', fontsize=15, pad=20)
        plt.tight_layout()
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f'Graph saved to {save_path}')
    else:
        raise FileNotFoundError(f'File {experiments_file_name} does not exist')


if __name__ == '__main__':
    experiments_file_name = 'experiments.md'
    save_path = Path('./images/experiment_history.jpg')
    generate_experiment_history_graph(experiments_file_name, save_path)