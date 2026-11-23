import pandas as pd
import os
from src import DATA
import plotly.graph_objects as go
from src.lumen.scramblers.aggregate_spectra import L2

def compare(file: os.PathLike, names: list[str]) -> None:
    data = pd.read_csv(os.path.join(DATA, file), header=None, dtype=str)
    entry_names = data.iloc[0].to_numpy()
    entry_names = entry_names[pd.notna(entry_names)]
    
    fig = go.Figure()
    absorbtions = []
    for idx, entry in enumerate(entry_names):
        if entry in names:
            wn = pd.to_numeric(data.iloc[1:, 2 * idx], errors='coerce')
            abs = pd.to_numeric(data.iloc[1:, 2 * idx + 1], errors='coerce')
            absorbtions.append(abs[pd.notna(abs)].to_numpy())
            
            fig.add_trace(go.Scatter(
                x=wn,
                y=abs,
                mode='markers',
                name=entry
            ))

    fig.update_layout(
        xaxis_title="X Values",
        yaxis_title="Y Values"
    )
    
    print(L2(absorbtions[0], absorbtions[1]))
    
    fig.show()

if __name__ == '__main__':
    file_name = "Primpke_FTIR_Microplastics_converted.csv"
    names = ["PP - Subway_Straw", "PP - White_Castle_Straw"]

    compare(file_name, names)
