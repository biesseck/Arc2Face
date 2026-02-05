import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime



def parse_args():
    parser = argparse.ArgumentParser(description="Generate point clouds and compute PCA.")
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--points-per-class", type=int, default=100)
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    return args


def generate_data(n_classes, points_per_class, dim, seed):
    np.random.seed(seed)
    all_points = []
    labels = []
    
    # Generate random centers and then points around them
    centers = np.random.uniform(-10, 10, (n_classes, dim))
    for i in range(n_classes):
        # Create a cluster with some variance
        cluster = centers[i] + np.random.normal(0, 1.5, (points_per_class, dim))
        all_points.append(cluster)
        labels.extend([i] * points_per_class)
        
    return np.vstack(all_points), np.array(labels)


def generate_angular_data(n_classes, points_per_class, dim, seed):
    np.random.seed(seed)
    all_points = []
    labels = []
    
    # concentration controls the 'width' (opening angle) of the cone
    # angle_sigma = 0.1 
    angle_sigma = 0.05
    # magnitude controls the 'length' and thickness of the cone
    mag_min, mag_max = 0.5, 1.5

    for i in range(n_classes):
        # 1. Generate a random unit vector for the class axis
        center = np.random.normal(0, 1, dim)
        center /= np.linalg.norm(center)
        
        # 2. Create orthogonal basis for the spread
        if dim == 3:
            ref = np.array([1, 0, 0]) if abs(center[0]) < 0.9 else np.array([0, 1, 0])
            v1 = np.cross(center, ref); v1 /= np.linalg.norm(v1)
            v2 = np.cross(center, v1); v2 /= np.linalg.norm(v2)
            
            # Angular noise (the "flare" of the cone)
            noise = np.random.normal(0, angle_sigma, (points_per_class, 2))
            directions = (np.outer(np.ones(points_per_class), center) + 
                          np.outer(noise[:, 0], v1) + 
                          np.outer(noise[:, 1], v2))
        else: # 2D
            v1 = np.array([-center[1], center[0]])
            noise = np.random.normal(0, angle_sigma, (points_per_class, 1))
            directions = (np.outer(np.ones(points_per_class), center) + 
                          np.outer(noise[:, 0], v1))

        # 3. Normalize directions to unit vectors first...
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        
        # 4. ...then multiply by random magnitudes to fill the cone's volume
        magnitudes = np.random.uniform(mag_min, mag_max, (points_per_class, 1))
        conical_cluster = directions * magnitudes
        
        all_points.append(conical_cluster)
        labels.extend([i] * points_per_class)
        
    return np.vstack(all_points), np.array(labels)


def save_2d_chart(filename, nclasses, X, y, mean_point, pc_vectors, dpi=200):
    plt.figure(figsize=(8, 6), dpi=dpi)
        
    # 1. Scatter plot the data
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.6, label='Data Points')
    
    # 2. Plot Eigenvectors and collect their tip coordinates
    vector_tips = []
    for v in pc_vectors:
        tip = mean_point + v
        vector_tips.append(tip)
        plt.quiver(mean_point[0], mean_point[1], v[0], v[1], color='red', 
                    angles='xy', scale_units='xy', scale=1, width=0.01)
    
    # 3. Calculate new limits
    all_coords = np.vstack([X, vector_tips, mean_point])
    x_min, y_min = np.min(all_coords, axis=0)
    x_max, y_max = np.max(all_coords, axis=0)
    
    # Add a 10% padding so things aren't touching the edge
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"PCA 2D - {nclasses} Classes (Scaled Vectors)")
    plt.savefig(filename)
    print("2D Visualization saved as pca_output.png")
    plt.close('all')


def save_3d_ply(filename, points, labels, pc_origin, pc_vectors):
    """Saves points and Eigenvectors as a skeleton in a PLY file."""
    # 1. Prepare Data Points
    n_data = len(points)
    colors = plt.cm.get_cmap('tab10')(labels / (labels.max() + 1))[:, :3] * 255
    
    # 2. Prepare Vector Vertices (Origin + 3 Tips)
    vector_vertices = np.vstack([pc_origin, [pc_origin + v for v in pc_vectors]])
    # Color the vector vertices white
    vector_colors = np.tile([255, 255, 255], (4, 1))
    
    all_vertices = np.vstack([points, vector_vertices])
    all_colors = np.vstack([colors, vector_colors])
    
    # 3. Define Edges (connecting origin to each tip)
    # Origin is at index: n_data
    # Tips are at indices: n_data+1, n_data+2, n_data+3
    edges = [
        (n_data, n_data + 1),
        (n_data, n_data + 2),
        (n_data, n_data + 3)
    ]

    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        
        # Write all vertices
        for p, c in zip(all_vertices, all_colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            
        # Write edges
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")
            
    print(f"3D Point cloud with eigenvector skeleton saved to {filename}")


def main():
    args = parse_args()

    # 1. Generate Data
    # X, y = generate_data(args.classes, args.points_per_class, args.dim, args.seed)
    X, y = generate_angular_data(args.classes, args.points_per_class, args.dim, args.seed)

    # 2. Compute PCA
    pca = PCA(n_components=args.dim)
    pca.fit(X)
    
    # Vectors scaled by eigenvalues: v_i = eigenvector_i * eigenvalue_i
    # Note: pca.explained_variance_ contains the eigenvalues
    pc_vectors = pca.components_ * pca.explained_variance_[:, np.newaxis]
    mean_point = pca.mean_

    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
    filename = f"pca_output_{args.dim}D_classes={args.classes}_points={args.points_per_class}_date={formatted_time}_seed={args.seed}"

    # 3. Visualization/Export
    if args.dim == 2:
        save_2d_chart(f"{filename}.png", args.classes, X, y, mean_point, pc_vectors, dpi=args.dpi)
    
    else: # 3D
        save_3d_ply(f"{filename}.ply", X, y, mean_point, pc_vectors)
        # Output the vectors to console so you can draw them as 'Lines' in MeshLab
        print(f"\nPrincipal Components (Origin: {mean_point})")
        for i, v in enumerate(pc_vectors):
            print(f"PC{i+1} (Length={pca.explained_variance_[i]:.2f}): {v}")

if __name__ == "__main__":
    main()