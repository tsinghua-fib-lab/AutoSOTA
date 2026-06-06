
import plotly.graph_objects as go
import numpy as np

def plot_3d_point_cloud(points, values, title="3D Point Cloud", output_file="plot.html", frame_duration=200):
    """
        points: (N, 3)
        values: (N,) or (T, N) for time sequence
        frame_duration: int, duration of each frame in milliseconds
    """
    values = np.array(values)
    
    # Handle single frame (N,) -> (1, N)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    
    T, N = values.shape
    
    # Create the initial trace
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=values[0],
            colorscale='Inferno',
            opacity=0.8,
            colorbar=dict(title='Value'),
            cmin=np.min(values),
            cmax=np.max(values)
        )
    )

    # If we only have one frame, just plot it statically
    if T == 1:
        markup_layout = dict(
            title=title,
            scene=dict(
                xaxis_title='X', 
                yaxis_title='Y', 
                zaxis_title='Z', 
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        fig = go.Figure(data=[trace], layout=markup_layout)
        fig.write_html(output_file)
        print(f"Plot saved to {output_file}")
        return

    # If multiple frames, create animation
    frames = []
    for t in range(T):
        frames.append(go.Frame(
            data=[go.Scatter3d(
                marker=dict(color=values[t])
            )],
            name=str(t)
        ))

    # Layout with sliders and play button
    fig = go.Figure(data=[trace], frames=frames)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X', 
            yaxis_title='Y', 
            zaxis_title='Z', 
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': frame_duration, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            }, {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }]
        }],
        sliders=[{
            'steps': [{
                'method': 'animate',
                'args': [[str(t)], {
                    'mode': 'immediate',
                    'frame': {'duration': frame_duration, 'redraw': True},
                    'transition': {'duration': 0}
                }],
                'label': str(t)
            } for t in range(T)],
            'currentvalue': {'prefix': 'Time: '}
        }]
    )

    fig.write_html(output_file)
    print(f"Animation saved to {output_file}")

