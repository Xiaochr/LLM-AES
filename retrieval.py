import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split



def find_top_k_similar_ids(df_1, df_2, k=5):
    similarities = cosine_similarity(list(df_2['gpt3_emb']), list(df_1['gpt3_emb']))
    print(similarities.shape)
    
    top_k_indices = (-similarities).argsort(axis=1)[:, :k]
    print(len(top_k_indices))
    
    top_k_ids = [[df_1['essay_id'].iloc[i] for i in row] for row in top_k_indices]
    
    return top_k_ids


def get_sim_df(essay_set=1):
    data = pd.read_csv("./results/ASAP_emb.csv", index_col=0)
    data = data[data["essay_set"] == essay_set]
    data.index = range(len(data))

    data["gpt3_emb"] = data["gpt3_emb"].apply(lambda x: eval(x))
    print(data.head())

    df_1, df_2 = train_test_split(data, test_size=0.2, random_state=42)
    df_2['sim_id'] = find_top_k_similar_ids(df_1, df_2)
    
    print(df_2.head())
    df_1.to_csv("./split_data/set{}_emb_train.csv".format(str(essay_set)))
    df_2.to_csv("./split_data/set{}_emb_test.csv".format(str(essay_set)))




if __name__ == '__main__':
    for i in range(1, 9):
        get_sim_df(essay_set=i)


