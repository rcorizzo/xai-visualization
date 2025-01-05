import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from PIL import Image
import sys


def preprocess_data(input_data, mask_data, include_zero, continuous_scores=False):
    n_timestamps, n_nodes, n_features = np.shape(mask_data)  # Number of nodes

    if np.shape(input_data) != np.shape(mask_data):
        raise ValueError("Input data and mask data should have the same structure")

    if continuous_scores:
        mask_data = mask_data * input_data
        print(mask_data)

    # Create a DataFrame for the 3D scatter plot
    data = []

    for timestamp in range(n_timestamps):
        for node in range(n_nodes):
            for feature in range(n_features):
                relevance = mask_data[timestamp, node, feature]  # (0-1)

                data.append({
                    'Timestep': timestamp,
                    'Node': node,
                    'Feature': feature,
                    'Relevance': relevance
                })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(df)

    if not include_zero:
        df = df[df['Relevance'] != 0]

    return df

def matplotlib_scatter(df, dataset, model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['Node'], df['Timestep'], df['Feature'], c=df['Relevance'], cmap='viridis')
    ax.set_xlabel('Node')
    ax.set_ylabel('Timestamp')
    ax.set_zlabel('Feature')
    plt.colorbar(sc)
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/{dataset}-{model}-3d_scatter_plot_matplotlib.pdf")


def plotly_3d_scatter(df, dataset, model):

    # Plotting with Plotly 3D Scatter
    fig = px.scatter_3d(
        df,
        x='Node',
        y='Timestep',
        z='Feature',
        color='Relevance',
        size='Relevance',
        opacity=0.5,
        color_continuous_scale='Viridis',
        title='3D Scatter Plot of Node, Timestep, and Feature Relevance'
    )

    # Customize layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='Node',
            yaxis_title='Timestep',
            zaxis_title='Feature'
        ),
        coloraxis_colorbar=dict(
            title='Relevance Score'
        ),
        template='plotly_white'
    )

    # Set the image size for A4 page dimensions (in pixels)
    # A4 size is approximately 595x842 points, which is roughly 842x1191 pixels at 96 DPI
    image_width = 1200  # Width in pixels
    image_height = 1600  # Height in pixels

    # Save the plot as a PNG image
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    fig.write_image(f"{path}/{dataset}-{model}-3d_scatter_plot_plotly.pdf", format='pdf', engine='kaleido')


def parallel_coordinates(df, dataset, model):
    # Filter to remove zero relevance entries (if needed)
    df_non_zero = df[df['Relevance'] != 0]

    # Generate the parallel coordinates plot
    fig = px.parallel_coordinates(
        df_non_zero,
        dimensions=['Node', 'Feature', 'Timestep'],  # Axes for parallel coordinates
        color='Relevance',  # Color based on relevance score
        color_continuous_scale='Viridis',  # Color scale
        title="Parallel Coordinates Plot of Nodes, Features, and Timesteps"
    )

    # Customize layout
    fig.update_layout(
        template='plotly_white',
        coloraxis_colorbar=dict(title='Relevance Score')
    )

    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)

    # Save plot to an interactive HTML file
    output_path_html = f"{path}/{dataset}-{model}-parallel_coordinates_plot.html"
    fig.write_html(output_path_html)

    print(f"Interactive plot saved at: {output_path_html}")

    # To save as a static image (optional)
    output_path_pdf = f"{path}/{dataset}-{model}-parallel_coordinates_plot.pdf"
    fig.write_image(output_path_pdf, format='pdf', engine='kaleido')

    print(f"Static image saved at: {output_path_pdf}")


def sankey_diagram(df, dataset, model):

    # The Node, Feature, and Timestamp categories are assigned unique indices for the Sankey diagram's source and target.
    # Flows are created between:
    # Node → Feature (relevance flow from node to feature).
    # Feature → Timestamp (relevance flow from feature to timestamp).

    # Filter to remove zero relevance entries (if needed)
    df_non_zero = df[df['Relevance'] != 0]

    # Create lists for Sankey diagram
    sources = []  # Source nodes
    targets = []  # Target nodes
    values = []  # Relevance scores (flow widths)
    labels = []  # Labels for blocks
    colors = []  # Colors for blocks

    # Map categories to unique indices for Sankey
    nodes = df_non_zero['Node'].unique()
    features = df_non_zero['Feature'].unique()
    timestamps = df_non_zero['Timestep'].unique()

    print()
    print(df_non_zero.head())
    print()

    # Combine all unique labels
    all_labels = list(nodes) + list(features) + list(timestamps)

    # Add prefixes to labels for clarity
    labeled_nodes = [f"Node: {label}" for label in nodes]
    labeled_features = [f"Feature: {label}" for label in features]
    labeled_timestamps = [f"Timestep: {label}" for label in timestamps]

    # Final list of labeled nodes for Sankey
    labels = labeled_nodes + labeled_features + labeled_timestamps

    # Create a color palette for categories
    color_nodes = "rgba(31, 119, 180, 0.8)"  # Blue for nodes
    color_features = "rgba(44, 160, 44, 0.8)"  # Green for features
    color_timestamps = "rgba(255, 127, 14, 0.8)"  # Orange for timestamps

    colors = (
            [color_nodes] * len(nodes) +
            [color_features] * len(features) +
            [color_timestamps] * len(timestamps)
    )

    # Map the labeled blocks to unique indices
    label_to_index = {label: i for i, label in enumerate(labels)}
    print(label_to_index)
    print()

    # Populate Sankey data
    for _, row in df_non_zero.iterrows():
        # Flow: Node -> Feature
        sources.append(label_to_index[f"Node: {int(row['Node'])}"])
        targets.append(label_to_index[f"Feature: {int(row['Feature'])}"])
        values.append(row['Relevance'])

        # Flow: Feature -> Timestamp
        sources.append(label_to_index[f"Feature: {int(row['Feature'])}"])
        targets.append(label_to_index[f"Timestep: {int(row['Timestep'])}"])
        values.append(row['Relevance'])

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,  # Padding between nodes
            thickness=20,  # Node thickness
            line=dict(color="black", width=0.5),
            label=labels,  # Labels with prefixes
            color=colors  # Assign category-specific colors
        ),
        link=dict(
            source=sources,  # Source nodes
            target=targets,  # Target nodes
            value=values,  # Flow widths
            color="rgba(0, 0, 0, 0.3)"  # Optional: Uniform flow color
        )
    )])

    # Customize layout
    fig.update_layout(
        title_text="Sankey Diagram with Emphasized Categories",
        font_size=10,
        template="plotly_white"
    )

    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)

    # Save the interactive HTML file
    output_path_html = f"{path}/{dataset}-{model}-sankey_diagram.html"
    fig.write_html(output_path_html)

    print(f"Sankey Diagram saved at: {output_path_html}")

    # Optionally save as a static PNG
    output_path_pdf = f"{path}/{dataset}-{model}-sankey_diagram.pdf"
    fig.write_image(output_path_pdf, format='pdf', engine='kaleido')

    print(f"Static Sankey Diagram saved at: {output_path_pdf}")


def facet_grid_diagram(df, dataset, model):
    #  Each row represents a specific node.
    #  The scatter plot in each row visualizes timestamp vs. feature, with relevance scores shown as the marker color.
    #  Dynamic Color Scale:
    #       The relevance score determines the marker color using the Viridis color scale.
    #       The color bar appears only on the first plot for a cleaner layout.

    # Choose which dimension to split by (e.g., Node)
    unique_nodes = sorted(df['Node'].unique())

    # Create subplots: One row per node
    fig = make_subplots(
        rows=len(unique_nodes),
        cols=1,
        subplot_titles=[f"Node {node}" for node in unique_nodes]
    )

    # Add 2D scatter plots for each node
    for i, node in enumerate(unique_nodes, start=1):
        # Filter data for the current node
        node_data = df[df['Node'] == node]

        # Create a scatter plot for Timestamp vs. Feature, colored by Relevance
        scatter = go.Scatter(
            x=node_data['Timestep'],
            y=node_data['Feature'],
            mode='markers',
            marker=dict(
                size=10,
                color=node_data['Relevance'],
                colorscale='Viridis',
                showscale=(i == 1),  # Show color scale only for the first plot
                colorbar=dict(title="Relevance")
            ),
            name=f"Node {node}"
        )

        # Add scatter plot to the corresponding subplot
        fig.add_trace(scatter, row=i, col=1)

    # Customize layout
    fig.update_layout(
        height=300 * len(unique_nodes),  # Adjust height for number of subplots
        width=800,
        title="Facet Grid of Timestep vs. Feature (Split by Node)",
        xaxis_title="Timestep",
        yaxis_title="Feature",
        showlegend=False  # Hide legends to declutter
    )

    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)

    # Save the plot as an interactive HTML file
    output_path_html = f"{path}/{dataset}-{model}-facet_grid.html"
    fig.write_html(output_path_html)

    print(f"Facet Grid saved at: {output_path_html}")

    # Optionally save as a static PNG
    output_path_pdf = f"{path}/{dataset}-{model}-facet_grid.pdf"
    fig.write_image(output_path_pdf, format='pdf', engine='kaleido')

    print(f"Static Facet Grid saved at: {output_path_pdf}")


def dynamic_graph_visualization(df, dataset, model):
    # Nodes and edges evolve over time, reflecting how relationships and relevance scores change.
    # Node colors provide an immediate visual cue for their importance at each timestamp.

    # Filter data to remove zero relevance scores
    df_non_zero = df[df['Relevance'] != 0]

    # Build the graph
    G = nx.Graph()

    # Add edges for each timestamp and track node types
    node_types = {}
    for timestamp in df_non_zero['Timestep'].unique():
        df_time = df_non_zero[df_non_zero['Timestep'] == timestamp]
        for _, row in df_time.iterrows():
            node = row['Node']
            feature = row['Feature']
            node_types[node] = "Node"
            node_types[feature] = "Feature"
            G.add_edge(node, feature, weight=row['Relevance'], timestamp=row['Timestep'])

    # Extract unique timestamps for animation frames
    unique_timestamps = sorted(df_non_zero['Timestep'].unique())

    # Prepare frames for animation
    frames = []
    for timestamp in unique_timestamps:
        edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['timestamp'] == timestamp]

        # Build node positions
        pos = nx.spring_layout(G, seed=42)

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_labels = []  # Collect edge labels
        for u, v, d in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_labels.append(((x0 + x1) / 2, (y0 + y1) / 2, f"{d['weight']:.2f}"))

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color="rgba(0, 0, 0, 0.5)"),
            hoverinfo="none",
            mode="lines"
        )

        # Create node traces
        node_x = []
        node_y = []
        node_color = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            # Use the node type (Node or Feature) from node_types
            label_prefix = node_types[node]
            node_text.append(f"{label_prefix} - {node}")
            relevance_sum = sum(d['weight'] for u, v, d in edges if u == node or v == node)
            node_color.append(relevance_sum)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(
                size=15,
                color=node_color,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                colorbar=dict(title="Relevance"),
                showscale=True
            ),
            hoverinfo="text"
        )

        # Add edge intensity annotations
        edge_annotations = [
            dict(
                x=x,
                y=y,
                xref="x",
                yref="y",
                text=label,
                showarrow=False,
                font=dict(size=10, color="darkblue")
            )
            for x, y, label in edge_labels
        ]

        # Add frame for the current timestamp
        frames.append(go.Frame(
            data=[edge_trace, node_trace],
            name=f"Timestamp {timestamp}",
            layout=go.Layout(
                annotations=edge_annotations + [
                    dict(
                        x=0.5,
                        y=1.2,
                        xref="paper",
                        yref="paper",
                        text=f"Timestep: {timestamp}",
                        showarrow=False,
                        font=dict(size=16, color="black")
                    )
                ]
            )
        ))

    # Create initial traces for the first timestamp
    initial_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['timestamp'] == unique_timestamps[0]]
    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    edge_labels = []
    for u, v, d in initial_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_labels.append(((x0 + x1) / 2, (y0 + y1) / 2, f"{d['weight']:.2f}"))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="rgba(0, 0, 0, 0.5)"),
        hoverinfo="none",
        mode="lines"
    )

    node_x = []
    node_y = []
    node_color = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label_prefix = node_types[node]
        node_text.append(f"{label_prefix} - {node}")
        relevance_sum = sum(d['weight'] for u, v, d in initial_edges if u == node or v == node)
        node_color.append(relevance_sum)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(
            size=15,
            color=node_color,
            colorscale="Viridis",
            cmin=0,
            cmax=1,
            colorbar=dict(title="Relevance"),
            showscale=True
        ),
        hoverinfo="text"
    )

    # Add edge annotations
    initial_edge_annotations = [
        dict(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=label,
            showarrow=False,
            font=dict(size=10, color="darkblue")
        )
        for x, y, label in edge_labels
    ]

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Dynamic Graph Visualization",
            titlefont_size=16,
            annotations=[
                            dict(
                                x=0.5,
                                y=1.3,
                                xref="paper",
                                yref="paper",
                                text="Nodes represent entities; edges show relationships with intensity.",
                                showarrow=False,
                                font=dict(size=14, color="black")
                            ),
                            dict(
                                x=0.5,
                                y=1.2,
                                xref="paper",
                                yref="paper",
                                text=f"Timestep: {unique_timestamps[0]}",
                                showarrow=False,
                                font=dict(size=16, color="black"),
                                name="timestep_label"
                            )
                        ] + initial_edge_annotations,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=60),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                    ]
                )
            ]
        ),
        frames=frames
    )

    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)

    # Save as an interactive HTML file
    output_path_html = f"{path}/{dataset}-{model}-dynamic_graph.html"
    fig.write_html(output_path_html)

    print(f"Dynamic graph saved at: {output_path_html}")


def heatmap_animation(df, dataset, model):
    # Prepare frames for animation
    timestamps = sorted(df['Timestep'].unique())
    frames = []
    for timestamp in timestamps:
        df_time = df[df['Timestep'] == timestamp]
        pivot_time = df_time.pivot_table(index='Node', columns='Feature', values='Relevance', aggfunc='mean',
                                         fill_value=0)
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=pivot_time.values,
                x=pivot_time.columns,
                y=pivot_time.index,
                colorscale='Viridis',
                zmin=0,
                zmax=1
            )],
            name=f"Timestep {timestamp}"
        ))

    # Initial heatmap for the first timestamp
    initial_data = df[df['Timestep'] == timestamps[0]]
    pivot_initial = initial_data.pivot_table(index='Node', columns='Feature', values='Relevance', fill_value=0)

    heatmap = go.Heatmap(
        z=pivot_initial.values,
        x=pivot_initial.columns,
        y=pivot_initial.index,
        colorscale='Viridis',
        zmin=0,
        zmax=1
    )

    # Create figure
    fig = go.Figure(
        data=[heatmap],
        layout=go.Layout(
            title="Heatmap Animation of Feature Relevance Over Time",
            titlefont=dict(size=16),
            xaxis=dict(title='Features'),
            yaxis=dict(title='Nodes'),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=1.05,  # Adjust position
                    y=1.15,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                    ]
                )
            ]
        ),
        frames=frames
    )

    # Add frame slider
    fig.update_layout(
        sliders=[
            dict(
                steps=[
                    dict(method="animate",
                         args=[[f"Timestep {timestamp}"],
                               dict(mode="immediate",
                                    frame=dict(duration=1000, redraw=True),
                                    transition=dict(duration=0))],
                         label=f"{timestamp}")
                    for timestamp in timestamps
                ],
                active=0,
                transition=dict(duration=0),
                x=0.1,
                xanchor="left",
                y=0,
                yanchor="top"
            )
        ]
    )

    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)

    # Save as an interactive HTML file
    output_path_html = f"{path}/{dataset}-{model}-heatmap-animation.html"
    fig.write_html(output_path_html)

    print(f"Heatmap animation saved at: {output_path_html}")


def chord_diagram_v3(df, dataset, model):

    print(df)

    # Combine all entities into a single set of labels
    nodes = df["Node"].unique()
    print(nodes)
    nodes = ["N" + str(node) for node in nodes]
    print(nodes)

    timesteps = df["Timestep"].unique()
    print(timesteps)
    timesteps = ["T" + str(time) for time in timesteps]

    features = df["Feature"].unique()
    print(features)
    features = ["F" + str(f) for f in features]

    all_labels = list(nodes) + list(timesteps) + list(features)

    print(all_labels)

    # Create a mapping from label to index
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    print(label_to_index)

    # Initialize the adjacency matrix (size = number of unique labels)
    n_labels = len(all_labels)
    adj_matrix = np.zeros((n_labels, n_labels))


    # Populate the adjacency matrix based on relevance
    for _, row in df.iterrows():
        node_idx = label_to_index["N" + str(int(row["Node"]))]
        timestep_idx = label_to_index["T" + str(int(row["Timestep"]))]
        feature_idx = label_to_index["F" + str(int(row["Feature"]))]

        # Add relevance to the respective connections
        adj_matrix[node_idx, timestep_idx] += row["Relevance"]
        adj_matrix[timestep_idx, feature_idx] += row["Relevance"]
        adj_matrix[node_idx, feature_idx] += row["Relevance"]  # Ensure direct Node-Feature connection is added

    # Normalize the adjacency matrix
    adj_matrix = adj_matrix / adj_matrix.max()

    print(adj_matrix)
    sys.exit(1)

    # Position labels around a circle
    theta = np.linspace(0, 2 * np.pi, n_labels, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    # Assign colors based on type
    colors = (
            ["blue"] * len(nodes) +  # Nodes
            ["purple"] * len(timesteps) +  # Timesteps
            ["green"] * len(features)  # Features
    )

    # Create traces for nodes (scatter points)
    node_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        marker=dict(size=15, color=colors),
        text=all_labels,
        textposition="top center",
        hoverinfo="text"
    )

    # Create traces for the arcs (connections)
    connections = []
    for i in range(n_labels):
        for j in range(n_labels):
            if i != j and adj_matrix[i, j] > 0.1:  # Only include significant flows
                xi, yi = x[i], y[i]
                xj, yj = x[j], y[j]

                # Draw a Bezier curve between points
                mid_x = (xi + xj) / 2
                mid_y = (yi + yj) / 2
                bezier_curve = go.Scatter(
                    x=[xi, mid_x, xj],
                    y=[yi, mid_y, yj],
                    mode="lines",
                    line=dict(width=adj_matrix[i, j] * 10, color="rgba(0, 0, 255, 0.5)"),
                    # Line width proportional to relevance
                    hoverinfo="text",
                    text=f"{all_labels[i]} → {all_labels[j]}: {adj_matrix[i, j]:.2f}"
                )
                connections.append(bezier_curve)

    # Add a static legend
    legend_trace = go.Scatter(
        x=[1.1, 1.1, 1.1],
        y=[0.8, 0.6, 0.4],
        mode="markers+text",
        marker=dict(size=15, color=["blue", "purple", "green"]),
        text=["Nodes", "Timesteps", "Features"],
        textposition="middle right",
        showlegend=False
    )

    # Combine all traces
    fig = go.Figure(data=[node_trace] + connections + [legend_trace])

    # Add layout and annotations
    fig.update_layout(
        title="Chord Diagram: Node-Timestep-Feature Relationships",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(
                text="<b>Legend:</b>",
                x=1.1, y=1,
                showarrow=False,
                font=dict(size=16),
                xref="paper", yref="paper"
            )
        ]
    )
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)

    # Save as an interactive HTML file
    output_path_html = f"{path}/{dataset}-{model}-chord-diagram.html"
    fig.write_html(output_path_html)

    print(f"Heatmap animation saved at: {output_path_html}")


def hive_plot(df, dataset, model):
    # Aggregate relevance scores across timestamps (or focus on a specific timestamp)
    df_aggregated = df.groupby(['Node', 'Feature'])['Relevance'].mean().reset_index()

    # Hive Plot Setup
    # Define axes for nodes and features
    node_axis = 0  # Axis 0: Nodes
    feature_axis = 1  # Axis 1: Features

    # Create a graph from the data
    G = nx.Graph()

    # Add nodes to the graph
    nodes = df_aggregated['Node'].unique()
    features = df_aggregated['Feature'].unique()

    for node in nodes:
        G.add_node(node, axis=node_axis)

    for feature in features:
        G.add_node(feature, axis=feature_axis)

    # Add edges with relevance as weight
    for _, row in df_aggregated.iterrows():
        G.add_edge(row['Node'], row['Feature'], weight=row['Relevance'])

    # Create distinct colors for nodes and features using colormaps
    node_colors = plt.cm.Blues(np.linspace(0.4, 1, len(nodes)))  # Shades of blue
    feature_colors = plt.cm.Greens(np.linspace(0.4, 1, len(features)))  # Shades of green

    # Map colors to nodes and features
    node_color_map = dict(zip(nodes, node_colors))
    feature_color_map = dict(zip(features, feature_colors))

    # Plot the Hive Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define axes positions
    axes_positions = {
        node_axis: (0, 1),  # Axis 0 (Nodes) at the top
        feature_axis: (1, 0)  # Axis 1 (Features) at the right
    }

    # Draw nodes on their respective axes
    for axis, (x, y) in axes_positions.items():
        axis_nodes = [n for n, d in G.nodes(data=True) if d['axis'] == axis]
        angles = np.linspace(0, 2 * np.pi, len(axis_nodes), endpoint=False)
        for angle, node in zip(angles, axis_nodes):
            G.nodes[node]['pos'] = (x + np.cos(angle), y + np.sin(angle))

    # Draw edges connecting nodes
    for u, v, data in G.edges(data=True):
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        ax.plot(
            [pos_u[0], pos_v[0]],
            [pos_u[1], pos_v[1]],
            color=plt.cm.viridis(data['weight']),
            linewidth=data['weight'] * 5,  # Scale edge thickness
            alpha=0.7
        )

    # Draw nodes with unique colors
    for n, d in G.nodes(data=True):
        color = node_color_map[n] if d['axis'] == node_axis else feature_color_map[n]
        ax.scatter(*d['pos'], s=100, color=color)

    # Add captions
    ax.text(0, 1.5, "Nodes", ha='center', fontsize=12, fontweight='bold', color='blue')
    ax.text(1.5, 0, "Features", ha='center', fontsize=12, fontweight='bold',
            color='green')

    # Add legends
    node_legend = plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color=node_color_map[n], label=n, linestyle='None')
            for n in nodes
        ],
        loc='upper left', fontsize=10, title="Nodes"
    )

    feature_legend = plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color=feature_color_map[f], label=f, linestyle='None')
            for f in features
        ],
        loc='lower left', fontsize=10, title="Features"
    )

    ax.add_artist(node_legend)

    # Customization
    ax.set_title("Hive Plot for Node-Feature Relevance", fontsize=16)
    ax.axis('off')
    plt.tight_layout()

    # Save or show the plot
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-hive-plot.pdf"
    plt.savefig(output_path, dpi=300, format="pdf")
    #plt.show()

    print(f"Hive plot saved at: {output_path}")


def stream_graph(df, dataset, model):
    # Aggregating relevance over nodes and features for each timestamp
    stream_data_nodes = df.groupby(['Timestep', 'Node'])['Relevance'].sum().unstack(fill_value=0)
    stream_data_features = df.groupby(['Timestep', 'Feature'])['Relevance'].sum().unstack(fill_value=0)

    # Define a colormap for nodes and features
    node_colors = plt.cm.Blues(np.linspace(0.4, 1, len(stream_data_nodes.columns)))
    feature_colors = plt.cm.Greens(np.linspace(0.4, 1, len(stream_data_features.columns)))

    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot for Nodes
    axes[0].stackplot(
        stream_data_nodes.index,
        stream_data_nodes.T,
        labels=stream_data_nodes.columns,
        colors=node_colors,
        alpha=0.8
    )
    axes[0].set_title("Stream Graph: Relevance by Nodes Over Time", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Aggregated Relevance", fontsize=12)
    axes[0].legend(
        loc='upper left',
        title="Nodes",
        fontsize=10,
        title_fontsize=12,
        bbox_to_anchor=(1.02, 1)
    )

    # Captions for Nodes
    axes[0].text(
        0.5, 1.1,
        "Each area represents the relevance score of a Node over time.",
        fontsize=10,
        ha='center',
        transform=axes[0].transAxes
    )

    # Plot for Features
    axes[1].stackplot(
        stream_data_features.index,
        stream_data_features.T,
        labels=stream_data_features.columns,
        colors=feature_colors,
        alpha=0.8
    )
    axes[1].set_title("Stream Graph: Relevance by Features Over Time", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Timesteps", fontsize=12)
    axes[1].set_ylabel("Aggregated Relevance", fontsize=12)
    axes[1].legend(
        loc='upper left',
        title="Features",
        fontsize=10,
        title_fontsize=12,
        bbox_to_anchor=(1.02, 1)
    )

    # Captions for Features
    axes[1].text(
        0.5, -0.2,
        "Each area represents the relevance score of a Feature over time.",
        fontsize=10,
        ha='center',
        transform=axes[1].transAxes
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legends on the right

    # Save or show the plot
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-stream-graph.pdf"
    plt.savefig(output_path, dpi=300, format="pdf")

    print(f"Stream Graph saved to: {output_path}")


def three_dim_point_cloud(df, dataset, model):
    # Extract unique values for nodes, features, and timestamps
    unique_nodes = df['Node'].unique()
    unique_features = df['Feature'].unique()
    unique_timestamps = df['Timestep'].unique()

    # Map nodes and features to numerical values for 3D positioning
    node_map = {node: idx for idx, node in enumerate(unique_nodes)}
    feature_map = {feature: idx for idx, feature in enumerate(unique_features)}

    # Map DataFrame columns to numerical values
    df['Node_Num'] = df['Node'].map(node_map)
    df['Feature_Num'] = df['Feature'].map(feature_map)

    # Create 3D Point Cloud
    fig = go.Figure()

    # Add points to the plot
    fig.add_trace(go.Scatter3d(
        x=df['Node_Num'],
        y=df['Feature_Num'],
        z=df['Timestep'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['Relevance'],  # Color by relevance
            colorscale='Viridis',
            colorbar=dict(title='Relevance'),
            opacity=0.8
        ),
        text=[f"Node: {row['Node']}<br>Feature: {row['Feature']}<br>Timestep: {row['Timestep']}"
              for _, row in df.iterrows()],
        hoverinfo='text'
    ))

    # Add captions for Nodes and Features
    fig.add_trace(go.Scatter3d(
        x=[-1] * len(unique_nodes),
        y=list(node_map.values()),
        z=[-1] * len(unique_nodes),
        mode='text',
        text=[f"{node}" for node in unique_nodes],
        textfont=dict(color='blue', size=14),
        name="Nodes"
    ))

    fig.add_trace(go.Scatter3d(
        x=list(feature_map.values()),
        y=[-1] * len(unique_features),
        z=[-1] * len(unique_features),
        mode='text',
        text=[f"{feature}" for feature in unique_features],
        textfont=dict(color='green', size=14),
        name="Features"
    ))

    # Update layout
    fig.update_layout(
        title="3D Point Cloud Visualization of Nodes, Features, and Timesteps",
        scene=dict(
            xaxis=dict(title="Nodes"),
            yaxis=dict(title="Features"),
            zaxis=dict(title="Timesteps")
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1)
    )


    # Save or show the plot
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-3d-point-cloud.pdf"
    fig.write_html(output_path.replace(".pdf",".html"))
    fig.write_image(output_path, format='pdf', engine='kaleido')

    print(f"3D point cloud plot saved to: {output_path}")


def circular_packing(df, dataset, model):
    # Aggregate relevance by Node and Feature
    node_relevance = df.groupby('Node')['Relevance'].sum().reset_index()
    feature_relevance = df.groupby('Feature')['Relevance'].sum().reset_index()

    # Step 1: Flatten Data for Plotly
    # Create labels and parents for Sunburst
    labels = ['Root'] + \
             ['Nodes'] + list(node_relevance['Node']) + \
             ['Features'] + list(feature_relevance['Feature'])

    parents = [''] + \
              ['Root'] + ['Nodes'] * len(node_relevance) + \
              ['Root'] + ['Features'] * len(feature_relevance)

    values = [None] + \
             [None] + list(node_relevance['Relevance']) + \
             [None] + list(feature_relevance['Relevance'])

    print("TEST")
    print(parents)
    print()
    print(labels)
    print()
    print(values)
    print("TEST")

    # Step 2: Create Sunburst Plot
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colorscale='Viridis',
            line=dict(color='white', width=2)
        )
    ))

    # Step 3: Update Layout with Titles and Captions
    fig.update_layout(
        title="Circular Packing Visualization of Nodes and Features",
        annotations=[
            dict(
                text="Nodes and Features are represented as circular areas.<br>"
                     "Size of each area corresponds to relevance.",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font=dict(size=16),
                xref="paper",
                yref="paper"
            )
        ],
        margin=dict(t=50, l=0, r=0, b=0)
    )

    # Save or show the plot
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-circular-packing.pdf"
    fig.write_html(output_path.replace(".pdf",".html"))
    #fig.write_image(output_path, format='pdf', engine='kaleido')


def dynamic_node_link_diagrams(df, dataset, model):

    # Ensure Node and Feature columns are strings
    df['Node'] = df['Node'].astype(str)
    df['Feature'] = df['Feature'].astype(str)

    # Normalize relevance scores to use as edge weights
    df['Relevance'] = df['Relevance'] / df['Relevance'].max()

    n_nodes = df['Node'].unique()
    n_timestamps = df['Timestep'].unique()
    n_features = df['Feature'].unique()

    # Get unique nodes and features
    nodes = [f"Node {i}" for i in range(n_nodes)]
    features = [f"Feature {i}" for i in range(n_features)]

    # Create a dictionary to map entities to unique indices
    all_entities = list(df['Node'].unique()) + list(df['Feature'].unique())
    entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}

    # Build node positions for visualization
    node_positions = {node: (np.random.uniform(0, 1), np.random.uniform(0, 1)) for node in nodes}
    feature_positions = {feature: (np.random.uniform(0, 1), np.random.uniform(0, 1)) for feature in features}

    # Merge positions
    positions = {**node_positions, **feature_positions}

    # Function to extract edges for a given timestamp
    def get_edges_at_timestamp(timestamp):
        df_at_time = df[df['Timestamp'] == timestamp]
        edges = []
        for _, row in df_at_time.iterrows():
            edges.append({
                'source': entity_to_idx[row['Node']],
                'target': entity_to_idx[row['Feature']],
                'weight': row['Relevance']
            })
        return edges

    # Create frames for animation
    frames = []
    for timestamp in range(n_timestamps):
        edges = get_edges_at_timestamp(timestamp)
        frame_data = []
        for edge in edges:
            source_pos = positions[all_entities[edge['source']]]
            target_pos = positions[all_entities[edge['target']]]
            frame_data.append(
                go.Scatter(
                    x=[source_pos[0], target_pos[0]],
                    y=[source_pos[1], target_pos[1]],
                    mode='lines',
                    line=dict(width=edge['weight'] * 5, color='blue'),
                    showlegend=False
                )
            )
        frame_data.append(
            go.Scatter(
                x=[positions[node][0] for node in nodes],
                y=[positions[node][1] for node in nodes],
                mode='markers+text',
                text=[node for node in nodes],
                marker=dict(size=10, color='cyan'),
                name='Nodes'
            )
        )
        frame_data.append(
            go.Scatter(
                x=[positions[feature][0] for feature in features],
                y=[positions[feature][1] for feature in features],
                mode='markers+text',
                text=[feature for feature in features],
                marker=dict(size=10, color='green'),
                name='Features'
            )
        )
        frames.append(go.Frame(data=frame_data, name=f"Time {timestamp}"))

    # Initial edges for the first frame
    edges = get_edges_at_timestamp(0)
    initial_data = []
    for edge in edges:
        source_pos = positions[all_entities[edge['source']]]
        target_pos = positions[all_entities[edge['target']]]
        initial_data.append(
            go.Scatter(
                x=[source_pos[0], target_pos[0]],
                y=[source_pos[1], target_pos[1]],
                mode='lines',
                line=dict(width=edge['weight'] * 5, color='blue'),
                showlegend=False
            )
        )
    initial_data.append(
        go.Scatter(
            x=[positions[node][0] for node in nodes],
            y=[positions[node][1] for node in nodes],
            mode='markers+text',
            text=[node for node in nodes],
            marker=dict(size=10, color='cyan'),
            name='Nodes'
        )
    )
    initial_data.append(
        go.Scatter(
            x=[positions[feature][0] for feature in features],
            y=[positions[feature][1] for feature in features],
            mode='markers+text',
            text=[feature for feature in features],
            marker=dict(size=10, color='green'),
            name='Features'
        )
    )

    # Create the figure
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title="Dynamic Node-Link Diagram",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 1000, "redraw": True}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}}])
                ]
            )]
        ),
        frames=frames
    )

    # Save or show the plot
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-dynamic-node-link.pdf"
    fig.write_html(output_path.replace(".pdf",".html"))
    #fig.write_image(output_path, format='pdf', engine='kaleido')


def treemap(df, dataset, model):
    # Aggregate relevance scores for nodes and features
    node_relevance = df.groupby('Node')['Relevance'].sum().reset_index()
    feature_relevance = df.groupby('Feature')['Relevance'].sum().reset_index()

    # Combine data for the treemap
    labels = ['Root'] + list(node_relevance['Node']) + list(feature_relevance['Feature'])
    parents = [''] + ['Root'] * len(node_relevance) + list(node_relevance['Node'])
    values = [None] + list(node_relevance['Relevance']) + list(feature_relevance['Relevance'])

    # Add a "type" field to distinguish nodes and features
    types = ['Root'] + ['Node'] * len(node_relevance) + ['Feature'] * len(feature_relevance)

    # Assign colors based on type
    color_map = {'Root': 'lightgrey', 'Node': 'cyan', 'Feature': 'green'}
    colors = [color_map[t] for t in types]

    # Create the Treemap
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        textinfo="label+value",
        marker=dict(
            colors=colors,
            line=dict(color="white", width=2)
        )
    ))

    # Add captions and annotations
    fig.update_layout(
        title="Treemap Visualization of Nodes and Features",
        annotations=[
            dict(
                text="<b>Legend:</b><br>"
                     "<span style='color:cyan;'>■ Nodes</span><br>"
                     "<span style='color:green;'>■ Features</span>",
                x=1.05,  # Position to the right of the treemap
                y=0.8,
                showarrow=False,
                font=dict(size=16),
                align="left",
                xref="paper",
                yref="paper"
            ),
            dict(
                text="Nodes and Features are represented hierarchically.<br>"
                     "Size of each section corresponds to relevance.",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font=dict(size=16),
                xref="paper",
                yref="paper"
            )
        ],
        margin=dict(t=50, l=0, r=120, b=0)  # Adjust margin for annotations
    )

    # Save or show the plot
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-treemap.pdf"
    fig.write_html(output_path.replace(".pdf",".html"))
    #fig.write_image(output_path, format='pdf', engine='kaleido')


def sunburst_chart(df, dataset, model):
    # Aggregate relevance scores for the Sunburst Chart
    aggregated_data = (
        df.groupby(['Timestep', 'Node', 'Feature'])['Relevance']
        .sum()
        .reset_index()
    )

    # Prepare hierarchical data for Sunburst
    timestamps = aggregated_data['Timestep'].unique()
    nodes = aggregated_data['Node'].unique()
    features = aggregated_data['Feature'].unique()

    names = ['Root'] + list(timestamps) + list(nodes) + list(features)
    parents = [''] + ['Root'] * len(timestamps) + \
              [aggregated_data.loc[aggregated_data['Node'] == node, 'Timestep'].iloc[0] for node in nodes] + \
              [aggregated_data.loc[aggregated_data['Feature'] == feature, 'Node'].iloc[0] for feature in features]

    values = [None] + \
             [aggregated_data.loc[aggregated_data['Timestep'] == timestamp, 'Relevance'].sum() for timestamp in
              timestamps] + \
             [aggregated_data.loc[aggregated_data['Node'] == node, 'Relevance'].sum() for node in nodes] + \
             list(aggregated_data['Relevance'])

    values = [None] + [None] + \
             [aggregated_data.loc[aggregated_data['Timestep'] == timestamp, 'Relevance'].sum() for timestamp in
              timestamps] + \
             [aggregated_data.loc[aggregated_data['Node'] == node, 'Relevance'].sum() for node in nodes]

    print(len(timestamps))
    print(len(names))
    print(len(parents))
    print(len(values))

    data = dict(names=names, parents=parents, values=values)

    fig = px.sunburst(
        data,
        names='names',
        parents='parents',
        values='values',
    )

    # Create Sunburst Chart
    # fig = go.Figure(go.Sunburst(
    #     labels=labels,
    #     parents=parents,
    #     values=values,
    #     branchvalues="total",
    #     textinfo="label+value",
    #     marker=dict(
    #         colorscale="Viridis",
    #         line=dict(color="white", width=2)
    #     )
    # ))

    # Add captions and annotations
    # fig.update_layout(
    #     title="Sunburst Chart Visualization of Nodes and Features",
    #     annotations=[
    #         dict(
    #             text="<b>Legend:</b><br>"
    #                  "<span style='color:blue;'>■ Timesteps</span><br>"
    #                  "<span style='color:cyan;'>■ Nodes</span><br>"
    #                  "<span style='color:green;'>■ Features</span>",
    #             x=1.05,  # Position to the right of the chart
    #             y=0.8,
    #             showarrow=False,
    #             font=dict(size=16),
    #             align="left",
    #             xref="paper",
    #             yref="paper"
    #         ),
    #         dict(
    #             text="The Sunburst Chart represents the hierarchy:<br>"
    #                  "Root → Timesteps → Nodes → Features.<br>"
    #                  "The size corresponds to relevance.",
    #             x=0.5,
    #             y=-0.1,
    #             showarrow=False,
    #             font=dict(size=16),
    #             xref="paper",
    #             yref="paper"
    #         )
    #     ],
    #     margin=dict(t=50, l=0, r=120, b=0)  # Adjust margin for annotations
    # )

    # Save or show the plot
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-sunburst-chart.pdf"
    fig.write_html(output_path.replace(".pdf",".html"))
    #fig.write_image(output_path, format='pdf', engine='kaleido')

    fig.show()
    sys.exit(1)


def timeline_cluster_visualization(df, dataset, model):
    # Timeline-Based Cluster Visualization
    # We will create a scatter plot where the x-axis is the timestamp, the y-axis is the node, and the color represents the relevance of features.

    fig = px.scatter(
        df,
        x='Timestep',
        y='Node',
        color='Relevance',
        hover_data=['Feature'],
        labels={'Relevance': 'Feature Relevance', 'Node': 'Node ID', 'Timestep': 'Time Step'},
        title="Timeline-Based Cluster Visualization"
    )

    # Add additional annotations or captions for clarity
    fig.update_layout(
        title="Timeline-Based Cluster Visualization of Nodes and Features",
        annotations=[
            dict(
                text="<b>Legend:</b><br>"
                     "<span style='color:blue;'>■ Nodes</span><br>"
                     "<span style='color:green;'>■ Features</span><br>"
                     "<span style='color:purple;'>■ Relevance (color scale)</span>",
                x=1.05,  # Position to the right of the chart
                y=0.8,
                showarrow=False,
                font=dict(size=16),
                align="left",
                xref="paper",
                yref="paper"
            ),
            dict(
                text="Each point represents a node at a specific timestamp<br>"
                     "The color intensity corresponds to the relevance of the feature.",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font=dict(size=16),
                xref="paper",
                yref="paper"
            )
        ],
        margin=dict(t=50, l=50, r=150, b=50)  # Adjust margin for annotations
    )

    fig.update_traces(marker=dict(size=16))

    # Save or show the plot
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-timeline-cluster.pdf"
    fig.write_html(output_path.replace(".pdf",".html"))
    #fig.write_image(output_path, format='pdf', engine='kaleido')


def force_directed_graph(df, dataset, model):
    import networkx as nx
    import plotly.graph_objects as go
    import os

    # Create Network Graph
    G = nx.Graph()

    # Add nodes for each unique node, feature, and timestep
    for node in df['Node'].unique():
        G.add_node(node, type="node")
    for feature in df['Feature'].unique():
        G.add_node(feature, type="feature")
    for timestep in df['Timestep'].unique():
        G.add_node(timestep, type="timestep")

    # Add edges between nodes, features, and timesteps
    for _, row in df.iterrows():
        G.add_edge(row['Node'], row['Feature'], weight=row['Relevance'])
        G.add_edge(row['Node'], row['Timestep'], weight=row['Relevance'])
        G.add_edge(row['Feature'], row['Timestep'], weight=row['Relevance'])

    # Extract positions for force-directed layout
    pos = nx.spring_layout(G, seed=42)  # Force-directed layout

    # Create Edge Traces
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color="#888"),
        hoverinfo='none',
        mode='lines'
    )

    # Add edges to the trace
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # Create Node Traces
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=10,
            color=[],
            colorscale='Viridis',
            colorbar=dict(
                thickness=15,
                title='Node Type',
                xanchor='left',
                titleside='right'
            )
        )
    )

    # Update nodes with type-specific colors and hover text
    node_colors = []
    node_texts = []

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)

        if data['type'] == "node":
            color = "blue"
            text = f"Node: {node}"
        elif data['type'] == "feature":
            color = "green"
            text = f"Feature: {node}"
        elif data['type'] == "timestep":
            color = "purple"
            text = f"Timestep: {node}"

        node_colors.append(color)
        node_texts.append(text)

    # Update node marker colors and text
    node_trace.marker.color = node_colors
    node_trace.text = node_texts

    # Create the plot
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Force-Directed Graph with Temporal Coloring",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        annotations=[
                            dict(
                                text="<b>Legend:</b><br>"
                                     "<span style='color:blue;'>■ Nodes</span><br>"
                                     "<span style='color:green;'>■ Features</span><br>"
                                     "<span style='color:purple;'>■ Timesteps</span>",
                                x=1.05,  # Position to the right of the chart
                                y=0.8,
                                showarrow=False,
                                font=dict(size=16),
                                align="left",
                                xref="paper",
                                yref="paper"
                            ),
                            dict(
                                text="Nodes are connected to Features and Timesteps.<br>"
                                     "Color represents the type of entity.",
                                x=0.5,
                                y=-0.1,
                                showarrow=False,
                                font=dict(size=16),
                                xref="paper",
                                yref="paper"
                            )
                        ],
                        margin=dict(t=50, l=50, r=150, b=50)  # Adjust margin for annotations
                    ))

    # Save or show the plot
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-force-directed-graph.html"
    fig.write_html(output_path)
    print(f"Graph saved to {output_path}")


def volume_plot(raw_data, dataset, model):
    path = f"results/{dataset}/{model}"
    os.makedirs(path, exist_ok=True)
    output_path = f"{path}/{dataset}-{model}-volume-rendering.png"

    #zoomout = 1 # Ideal for few nodes
    zoomout = 6 # Ideal for many nodes

    pv.start_xvfb()

    T, N, F = raw_data.shape

    # Create a uniform grid, which is required for volume rendering
    grid = pv.ImageData(dimensions=(T, N, F))

    # Set the grid spacing (optional)
    grid.spacing = (1, 1, 1)

    # Add the data values to the grid
    grid.point_data["values"] = raw_data.flatten(
        order="F"
    )  # Flatten the data in Fortran order

    # Create a plotter object
    plotter = pv.Plotter(off_screen=True)

    # Add the volume rendering
    plotter.add_volume(
        grid,
        cmap="viridis",
        opacity="sigmoid",
        scalar_bar_args={
            "title": "Intensity",
            "vertical": True,  # Vertical orientation
            "position_x": 0.85,  # X position (closer to 1.0 is right)
            "position_y": 0.3,  # Y position (closer to 1.0 is top)
            "width": 0.1,  # Width of the scalar bar
            "height": 0.5,  # Height of the scalar bar
        },
    )

    plotter.show_grid(
        xtitle="Timesteps",
        ytitle="Nodes",
        ztitle="Features",
    )
    # plotter.add_axes(xtitle="Timesteps", ytitle="Nodes", ztitle="Features")

    plotter.camera.position = (zoomout * 30.0, zoomout * 36.0, zoomout * 30.0)
    plotter.screenshot(output_path)
    plotter.close()


# def plot_mask(x, mask, filename):
#     if not os.path.exists(os.path.dirname(filename)):
#         os.makedirs(os.path.dirname(filename))
#
#     F = x.shape[-1]
#     rows = 2
#     cols = math.ceil(F / 2)
#     _, ax = plt.subplots(rows, cols, figsize=(15, 5))
#     for f in range(F):
#         sns.heatmap(
#             x[0, ..., f],
#             cbar=False,
#             square=True,
#             cmap="bone",
#             ax=ax[f // cols][f % cols],
#         )
#         for i in range(mask[0, ..., f].shape[0]):
#             for j in range(mask[0, ..., f].shape[1]):
#                 if mask[0, ..., f][i, j] == 1:
#                     ax[f // cols][f % cols].add_patch(
#                         plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="cyan", lw=1)
#                     )
#
#         ax[f // cols][f % cols].set_xlabel("Nodes", fontsize=16)
#         ax[f // cols][f % cols].set_ylabel("Timesteps", fontsize=16)
#         ax[f // cols][f % cols].tick_params(axis="x", labelrotation=90, labelsize=14)
#         ax[f // cols][f % cols].tick_params(axis="y", labelrotation=90, labelsize=14)
#
#         ax[f // cols][f % cols].yaxis.set_major_locator(MultipleLocator(3))
#         ax[f // cols][f % cols].set_yticklabels([i * 3 for i in range(mask.shape[1])])
#         ax[f // cols][f % cols].xaxis.set_major_locator(MultipleLocator(2))
#         ax[f // cols][f % cols].set_xticklabels([i * 2 for i in range(mask.shape[1])])
#
#         plt.tight_layout()
#         plt.savefig(filename)