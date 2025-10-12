import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.fft import fft2, ifft2, fftfreq, fftshift
import trimesh

def mynormal2depth(normals, Id, object_name_list):
    """
    This function implements the normal to depth.
    ## Parameters:
    -   Normal:
            normal map of object

    -   Id:
            current object's ID

    -   object_name_list:
            the list of objects' names
    ## Returns:
    tuple of ->
    - X:
        X mesh grid of pixels
    - y:
        Y mesh grid of pixels
    - Surface:
        Surface depth mesh grid 
    """
    height, width, _ = normals.shape
    p = np.clip( normals[:, :, 0] / (normals[:, :, 2] + 1e-6 ), -30, 30)
    q = np.clip( -normals[:, :, 1] / (normals[:, :, 2] + 1e-6 ), -30, 30)
    surface = frankot_chellappa(p, q)
    x, y = np.meshgrid(np.linspace(0, width, int(width)),
                     np.linspace(0, height, int(height)))
    fig = plt.figure(figsize=(6, 6*height/width))
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-90, azim=-90)
    ax.plot_surface(x, y, surface, cmap=cm.coolwarm) # type: ignore
    plt.xlabel("pixel x")
    plt.ylabel("pixel y")
    plt.title(f"{object_name_list[Id]} Reconstructed Surface")
    fig.savefig(f"../results/Object {Id+1} - 3D surface.png",
                dpi = 400)
    return (x, y, surface)

    

def frankot_chellappa(p, q, padding=0):
    """
    Frankot-Chellappa algorithm for surface reconstruction from gradients.
    
    ## Parameters:
    p : numpy.ndarray
        Gradient in x-direction (df/dx)
    q : numpy.ndarray
        Gradient in y-direction (df/dy)
    padding : int
        Zero padding to reduce boundary effects
    
    ## Returns:
    z : numpy.ndarray
        Reconstructed surface height
    """
    
    # Get dimensions
    m, n = p.shape
    
    # Apply padding if requested
    if padding > 0:
        p_padded = np.pad(p, padding, mode='constant')
        q_padded = np.pad(q, padding, mode='constant')
        m_pad, n_pad = p_padded.shape
    else:
        p_padded = p
        q_padded = q
        m_pad, n_pad = m, n
    
    # Compute Fourier transforms of gradients
    P = fft2(p_padded)
    Q = fft2(q_padded)
    
    # Create frequency grids
    wx = 2 * np.pi * fftfreq(m_pad)
    wy = 2 * np.pi * fftfreq(n_pad)
    Wx, Wy = np.meshgrid(wy, wx)  # Note: meshgrid uses (cols, rows)
    
    # Avoid division by zero at DC component
    denominator = Wx**2 + Wy**2
    denominator[0, 0] = 1  # Set DC component to avoid division by zero
    
    # Compute Fourier transform of surface
    Z = (-1j * Wx * P - 1j * Wy * Q) / denominator
    
    # Set DC component to zero (arbitrary constant offset)
    Z[0, 0] = 0
    
    # Inverse Fourier transform to get surface
    z_padded = np.real(ifft2(Z))
    
    # Remove padding if applied
    if padding > 0:
        z = z_padded[padding:-padding, padding:-padding]
    else:
        z = z_padded
    
    return z





def surface2mesh(X,Y,Z,Id):
    """Converts surface 2D representation to mesh and saves the results

    ## Parameters:
    - X:
        X mesh grid of pixels
    - y:
        Y mesh grid of pixels
    - Z:
        Z mesh grid of pixels
    - Id:
        Objects Id
    """
    height, width = Z.shape
    vertices = np.zeros((height*width, 3))
    vertices[:, 0] = -X.flatten()  # x-coordinates
    vertices[:, 1] = -Y.flatten()  # y-coordinates
    vertices[:, 2] = -Z.flatten()  # z-coordinates (depth)

    # Step 3: Create faces (triangles) for the mesh
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            # Index of the current vertex in the flattened array
            v0 = i * width + j
            v1 = v0 + 1
            v2 = (i + 1) * width + j
            v3 = v2 + 1
            # Create two triangles per grid cell
            faces.append([v0, v1, v2])  # First triangle
            faces.append([v1, v3, v2])  # Second triangle

    faces = np.array(faces)
    # Create the 3D mesh using trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Export or visualize the mesh
    # Save the mesh to a file (e.g., STL format)
    mesh.export(f'../results/Object-{Id+1} surface_mesh.stl')

