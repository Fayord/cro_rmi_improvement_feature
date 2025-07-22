# load the graph data library from /Users/ford/Documents/coding_trae/cro_rmi_improvement_feature/src/cro_rmi_improvement_feature/network_analyzer/data/graph/graph_data_library.pkl with pickle
import pickle
import os
from create_graph_data_library import GraphDataLibrary

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    graph_data_library_path = os.path.join(
        dir_path,
        "../data/graph/graph_data_library.pkl",
    )
    graph_data_library: GraphDataLibrary = pickle.load(
        open(graph_data_library_path, "rb")
    )
    print(graph_data_library.company_graph_datas)
    # from graph_data_library
