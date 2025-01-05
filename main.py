import numpy as np
from plotly_viz import *
from pytftk.dicts import dict_join
from pytftk.sequence import obs2seqs
from pytftk.gpu_tools import use_devices, await_avail_memory

# Dependencies
# plotly kaleido reportlab pyvista

dataset_config = {
    "beijing-multisite-airquality": {
        "timesteps": 6,
        "nodes": 12,
        "features": 11,
    },
    "lightsource": {
        "timesteps": 19,
        "nodes": 7,
        "features": 11,
        "test_dates": 36,
    },
    "pems-sf-weather": {
        "timesteps": 6,
        "nodes": 163,
        "features": 16,
    },
    "pv-italy": {
        "timesteps": 19,
        "nodes": 17,
        "features": 12,
        "test_dates": 85,
    },
    "wind-nrel": {
        "timesteps": 24,
        "nodes": 5,
        "features": 8,
        "test_dates": 73,
    },
}

def create_timeseries(dataset, dataset_name):
    horizon = dataset_config[dataset_name]["timesteps"]
    X, Y = obs2seqs(dataset, horizon, horizon, horizon)
    return X, Y


path = "data/"

datasets_seq = {
    "lightsource": "324",
    "beijing-multisite-airquality": "3426",
    "pv-italy": "671",
    "wind-nrel": "496",
    "pems-sf-weather": "1524"
}

models = ["LSTM", "CNN-LSTM", "GCN-LSTM"]

methods = [volume_plot,
           matplotlib_scatter, plotly_3d_scatter, parallel_coordinates,
           sankey_diagram, facet_grid_diagram, dynamic_graph_visualization,
           heatmap_animation,
           chord_diagram_v3,
           hive_plot,
           stream_graph,
           three_dim_point_cloud,
           treemap,
           timeline_cluster_visualization,
           force_directed_graph]


for dataset in datasets_seq:
    for model in models:
        print()
        print(dataset, model, "...")
        input_data = np.load(f"{path}/{dataset}/{dataset}.npz")["data"]
        print(np.shape(input_data))
        seqs, _ = create_timeseries(input_data, dataset)
        # Element-wise product per mostrare valore della feature quando l'entry è selezionata nella maschera
        print(np.shape(seqs))
        seq = seqs[int(datasets_seq[dataset])]

        raw_data = np.load(f"masks/{model}-{dataset}/tmp/{datasets_seq[dataset]}.npy")[0]
        print("*", np.shape(raw_data))  # e.g. (19, 7, 11)

        df = preprocess_data(seq, raw_data, False, continuous_scores=True)

        for method in methods:
            try:
                if method == volume_plot:
                    method(raw_data * seq, dataset, model)
                else:
                    method(df, dataset, model)
            except Exception as e:
                print("Error in method:", method, str(e))

#TODO: fix circular_packing, dynamic_node_link_diagrams, sunburst_chart
