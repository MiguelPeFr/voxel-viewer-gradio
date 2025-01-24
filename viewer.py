import gradio as gr
import numpy as np
import plotly.graph_objects as go
from pyvox.models import Vox
from pyvox.writer import VoxWriter
from pyvox.custom_parser import CustomVoxParser

def load_vox_model(file_path):
    """Load and parse a .vox file"""
    try:
        print(f"Attempting to parse vox file: {file_path}")
        model = CustomVoxParser(file_path).parse()
        print(f"Model parsed successfully")
        
        voxels = model.to_dense()
        print(f"Voxel array shape: {voxels.shape}")
        print(f"Number of non-zero voxels: {np.count_nonzero(voxels)}")
        print(f"Palette size: {len(model.palette)}")
        
        if np.count_nonzero(voxels) == 0:
            print("Warning: No voxels found in the model")
            return None, None
            
        return voxels, model.palette
    except Exception as e:
        print(f"Error in load_vox_model: {str(e)}")
        return None, None

def create_3d_scatter(voxels, palette):
    """Create a 3D scatter plot from voxel data"""
    # Get coordinates and color indices of non-zero voxels
    x, y, z = np.nonzero(voxels)
    color_indices = voxels[x, y, z] - 1  # Subtract 1 since palette indices in .vox start at 1
    
    # Apply rotations: first 90 degrees in X, then 180 degrees in Z
    # Convert to radians
    theta_x = np.pi / 2  # 90 degrees
    theta_z = np.pi      # 180 degrees
    
    # First rotation around X axis
    y_rot = y * np.cos(theta_x) - z * np.sin(theta_x)
    z_rot = y * np.sin(theta_x) + z * np.cos(theta_x)
    y, z = y_rot, z_rot
    
    # Then rotation around Z axis
    x_rot = x * np.cos(theta_z) - y * np.sin(theta_z)
    y_rot = x * np.sin(theta_z) + y * np.cos(theta_z)
    x, y = x_rot, y_rot
    
    
    # Convert palette indices to RGB colors using direct palette indexing
    rgb_colors = [f'rgb({int(palette[c][0])}, {int(palette[c][1])}, {int(palette[c][2])})' 
                 if c < len(palette) else 'rgb(255, 255, 255)' for c in color_indices]
    
    # Create 3D scatter plot with improved voxel representation
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=6, 
            color=rgb_colors,
            opacity=1.0,
            symbol='square',  # Using square symbol (supported by Plotly)
            line=dict(width=0),  # Remove border lines completely
            sizemode='diameter', 
            sizeref=1.0  # Reference scale for consistent size
        )
    )])
    
    # Calculate center and range for better camera positioning
    center_x = (x.max() + x.min()) / 2
    center_y = (y.max() + y.min()) / 2
    center_z = (z.max() + z.min()) / 2
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    
    # better visualization
    fig.update_layout(
        scene=dict(
            aspectmode='cube',  # Force cubic aspect ratio
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=0.9, z=0.9) 
            ),
            xaxis=dict(range=[center_x - max_range/1.5, center_x + max_range/1.5], showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
            yaxis=dict(range=[center_y - max_range/1.5, center_y + max_range/1.5], showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
            zaxis=dict(range=[center_z - max_range/1.5, center_z + max_range/1.5], showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        template='plotly_dark',  # Dark theme for better contrast
        paper_bgcolor='rgba(0,0,0,1)',  # Pure black background
        plot_bgcolor='rgba(0,0,0,1)'
    )
    
    return fig

def display_vox_model(vox_file):
    """Main function to display the voxel model"""
    try:
        # Load the vox model
        if not vox_file:
            fig = go.Figure()
            fig.add_annotation(
                text="Please upload a .vox file",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="white")
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,1)',
                plot_bgcolor='rgba(0,0,0,1)'
            )
            return fig
        
        if not vox_file.name.endswith('.vox'):
            fig = go.Figure()
            fig.add_annotation(
                text="Please upload a valid .vox file",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="white")
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,1)',
                plot_bgcolor='rgba(0,0,0,1)'
            )
            return fig
        
        print(f"Loading vox file: {vox_file.name}")
        voxels, palette = load_vox_model(vox_file.temp_path if hasattr(vox_file, 'temp_path') else vox_file.name)
        
        if voxels is None or palette is None:
            fig = go.Figure()
            error_message = "Error: Could not load voxel data from file\n"
            error_message += "This might be due to version incompatibility.\n"
            error_message += "The viewer currently supports .vox files up to version 200."
            fig.add_annotation(
                text=error_message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="white")
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,1)',
                plot_bgcolor='rgba(0,0,0,1)'
            )
            return fig
        
        if voxels.size == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Error: No voxels found in the model",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="white")
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,1)',
                plot_bgcolor='rgba(0,0,0,1)'
            )
            return fig
        
        print(f"Model loaded successfully. Shape: {voxels.shape}, Palette size: {len(palette)}")
        
        # Create the 3D visualization
        fig = create_3d_scatter(voxels, palette)
        
        return fig
    except Exception as e:
        print(f"Error details: {str(e)}")
        # Create an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading model: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,1)',
            plot_bgcolor='rgba(0,0,0,1)'
        )
        return fig

# Create Gradio interface
interface = gr.Interface(
    fn=display_vox_model,
    inputs=gr.File(label="Upload .vox file"),
    outputs=gr.Plot(label="3D Voxel Model"),  # Remove the type parameter
    title="Voxel Model Viewer",
    description="Upload a .vox file to view the 3D voxelized model.",
    examples=[
        ["examples/modelo_optimizado.vox"],
        ["examples/Poster.vox"],
        ["examples/Horse.vox"]
    ],
    cache_examples=True  # Enable caching to ensure examples work properly
)

if __name__ == "__main__":
    interface.launch()
