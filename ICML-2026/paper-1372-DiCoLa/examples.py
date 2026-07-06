
import json
import pandas as pd
import io
from DiCoLa.Recursive_PAG import DiCola_learner
from compare_algs.fci_alg import my_fci
from DiCoLa.utils import f1_score_edges


def base_example(graph_name):
    print("="*10,f"{graph_name} examples:","="*10)
    with open(f"Data/{graph_name}/pags_{graph_name}.json", "r") as f:
        all_pag = json.load(f)
    
    for id_num in [1,2,3]:
        
        pag_df_csv =  all_pag[str(id_num)]["pag"]
        true_pag_df = pd.read_csv(io.StringIO(pag_df_csv), index_col=0)


        data = pd.read_csv(f"Data/{graph_name}/data_2000_{id_num}.csv")
        # drop latent variables
        latent_nodes = [node for node in data.columns if node.startswith('L')]
        observed_data = data.drop(columns=latent_nodes)

        res_fci = my_fci(data=observed_data, alpha=0.01)
        print("FCI results:")
        score_fci = f1_score_edges(true_pag_df, res_fci['PAG.DataFrame'])
        print('number of CI tests:', res_fci['CI_num'], 'runtime:', f"{res_fci['runtime_sec']:.3f}")
        print('f1-score:', f"{score_fci['f1']:.3f}")


        res_DiCoLa_fci = DiCola_learner(observed_data=observed_data, leaf_node_learner=my_fci, alpha=0.01, ci_type="Fisher_Z")
        print("DiCoLa+fci results:")
        score_DiCoLa = f1_score_edges(true_pag_df, res_DiCoLa_fci['PAG.DataFrame'])
        print('number of CI tests:', res_DiCoLa_fci['CI_num'], 'runtime:', f"{res_DiCoLa_fci['runtime_sec']:.3f}")
        print('f1-score:', f"{score_DiCoLa['f1']:.3f}")


        print("-"*40)


def oracle_example(graph_name):
    print("="*10,f"{graph_name} oracle examples:","="*10)
    with open(f"Data/{graph_name}/dags_{graph_name}.json", "r") as f:
        all_dag = json.load(f)
    with open(f"Data/{graph_name}/pags_{graph_name}.json", "r") as f:
        all_pag = json.load(f)


    for id_num in [1,2,3]:
        
        dag_df_csv =  all_dag[str(id_num)]["graph"]
        adj_df = pd.read_csv(io.StringIO(dag_df_csv), index_col=0)

        pag_df_csv =  all_pag[str(id_num)]["pag"]
        true_pag_df = pd.read_csv(io.StringIO(pag_df_csv), index_col=0)

        # latent variables
        latent_nodes = [node for node in adj_df.columns if node.startswith('L')]
        
        # setting the latent variables 
        res_DiCoLa_oracle = DiCola_learner(observed_data=adj_df, ci_type="D_sep", latent_nodes=latent_nodes)
        print("DiCoLa+oracle results:")
        score_DiCoLa = f1_score_edges(true_pag_df, res_DiCoLa_oracle['PAG.DataFrame'])
        print('number of CI tests:', res_DiCoLa_oracle['CI_num'], 'runtime:', f"{res_DiCoLa_oracle['runtime_sec']:.3f}")
        print('f1-score:', f"{score_DiCoLa['f1']:.3f}")


        print("-"*40)



if __name__ == "__main__":

    ''' mildew examples '''
    base_example('mildew')

    ''' andes examples '''
    base_example('andes')



    ''' oracle examples '''
    # Suppose we have the Oracle CI tests, only for the purpose of Oracle testing

    oracle_example('mildew')
    oracle_example('andes')