import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity




def parse_args():
    parser = argparse.ArgumentParser(description="Generate point clouds and compute PCA.")
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--points-per-class", type=int, default=50)
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--k", type=int, default=3)
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


def save_2d_chart(filename, nclasses, X, y, centroids, dpi=200):
    plt.figure(figsize=(8, 6), dpi=dpi)
        
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.1, label='Data Points')
    scatter = plt.scatter(centroids[:, 0], centroids[:, 1], c=np.unique(y), cmap='tab10', alpha=1.0, label='Centroids')
    
    all_coords = X
    x_min, y_min = np.min(all_coords, axis=0)
    x_max, y_max = np.max(all_coords, axis=0)
    
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



def find_tangent_void_direction(target_centroid, neighbor_centroids):
    """
    Finds the direction in the Target's tangent space that points 
    furthest away from the weighted center of the neighbors.
    """
    # 1. Project Neighbors onto the Tangent Plane of the Target
    # Formula: v_proj = v - (v . target) * target
    # (Removes the component parallel to the target)
    
    # Dot products of neighbors with target
    dots = np.dot(neighbor_centroids, target_centroid)
    
    # Calculate projections
    # shape (k, dim)
    projections = neighbor_centroids - np.outer(dots, target_centroid)
    
    # 2. Find the "Center of Mass" of these projections
    # This vector points straight towards the "crowd" of neighbors in the tangent plane
    crowd_direction = np.mean(projections, axis=0)
    
    # 3. The Void is the OPPOSITE of the crowd
    void_vector = -crowd_direction
    
    # 4. Normalize to unit vector
    norm = np.linalg.norm(void_vector)
    if norm < 1e-9:
        # Fallback: If neighbors surround target perfectly (mean=0), 
        # use random orthogonal direction
        print("Warning: Neighbors surround target perfectly. Using random void.")
        void_vector = np.random.normal(0, 1, target_centroid.shape)
        void_vector -= np.dot(void_vector, target_centroid) * target_centroid
        void_vector /= np.linalg.norm(void_vector)
    else:
        void_vector /= norm
        
    return void_vector



def save_void_visualization_2d(centroids, isolated_indices, target_idx, neighbor_indices, void_vector, filename="void_viz_2d.png"):
    # 2. Setup Plot
    plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.gca()
    
    # Draw Unit Circle (Context)
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
    ax.add_artist(circle)
    
    # 3. Plot All Centroids (Background)
    # Create a mapping from index -> isolation rank
    rank_map = {idx: rank for rank, idx in enumerate(isolated_indices)}
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='lightgray', s=50, label='Other Classes', alpha=0.5)
    
    # 4. Highlight Neighbors
    neigh_2d = centroids[neighbor_indices]
    plt.scatter(neigh_2d[:, 0], neigh_2d[:, 1], c='blue', s=100, label='k-Nearest Neighbors')
    
    # 5. Highlight Target
    t_pt = centroids[target_idx]
    plt.scatter(t_pt[0], t_pt[1], c='red', s=150, edgecolors='black', label=f'Target (Rank {rank_map[target_idx]})')
    
    # 6. Draw Void Vector
    # We draw it starting from the target centroid
    plt.quiver(t_pt[0], t_pt[1], void_vector[0], void_vector[1], 
               angles='xy', scale_units='xy', scale=1, color='lime', width=0.015, label='Void Direction')

    # 7. Annotate Ranks
    # Only annotate the relevant ones to avoid clutter
    indices_to_annotate = np.append(neighbor_indices, target_idx)
    for idx in indices_to_annotate:
        plt.text(centroids[idx, 0]+0.02, centroids[idx, 1]+0.02, f"#{rank_map[idx]}", fontsize=9, fontweight='bold')

    plt.legend(loc='upper right')
    plt.title(f"Void Direction Visualization (Target Rank: {rank_map[target_idx]})")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.axis('equal')
    
    plt.savefig(filename, facecolor='white')
    plt.close()
    print(f"2D Void Visualization saved to {filename}")



def save_void_visualization_3d_wireframe(centroids, target_idx, neighbor_indices, void_vector, filename="void_wireframe_viz.ply"):
    """
    Saves a 3D PLY where:
    - Centroids are standard vertices (dots).
    - Void Vector is a WIREFRAME Pyramid (Edges) representing an arrow.
    """
    
    # 1. PREPARE CENTROIDS
    n_centroids = len(centroids)
    
    # Colors
    colors = np.full((n_centroids, 3), 200, dtype=int)
    colors[neighbor_indices] = [0, 100, 255] # Neighbors Blue
    colors[target_idx] = [255, 0, 0] # Target Red

    # 2. GENERATE PYRAMID VERTICES
    start_pt = centroids[target_idx]
    end_pt = start_pt + void_vector
    
    # Calculate orientation for the base
    vec = end_pt - start_pt
    length = np.linalg.norm(vec)
    z_axis = vec / length
    
    if np.abs(z_axis[0]) < 0.9: x_axis = np.cross(z_axis, [1, 0, 0])
    else: x_axis = np.cross(z_axis, [0, 1, 0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    thickness = 0.005 
    
    # 4 corners around the start point to form the pyramid base
    offsets = [
        (x_axis + y_axis)*thickness, (x_axis - y_axis)*thickness,
        (-x_axis - y_axis)*thickness, (-x_axis + y_axis)*thickness
    ]
    
    pyramid_vertices = []
    # Base Ring (at start_pt) - Indices 0, 1, 2, 3
    for off in offsets: pyramid_vertices.append(start_pt + off)
    # The Tip (at end_pt) - Index 4
    pyramid_vertices.append(end_pt)
    
    pyramid_vertices = np.array(pyramid_vertices)
    # Make wireframe Green
    pyramid_colors = np.tile([50, 205, 50], (5, 1))
    
    # 3. DEFINE EDGES
    # Base: 0,1,2,3 | Tip: 4
    offset = n_centroids 
    
    wireframe_edges = [
        # Base Ring (Square)
        (0,1), (1,2), (2,3), (3,0),
        # Connecting Lines (Base corners to the single Tip vertex)
        (0,4), (1,4), (2,4), (3,4)
    ]
    
    final_edges = [(e[0]+offset, e[1]+offset) for e in wireframe_edges]

    # 4. COMBINE AND SAVE
    all_vertices = np.vstack([centroids, pyramid_vertices])
    all_colors = np.vstack([colors, pyramid_colors])

    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(final_edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        
        # Write Vertices
        for p, c in zip(all_vertices, all_colors):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")
            
        # Write Edges
        for e in final_edges:
            f.write(f"{e[0]} {e[1]}\n")
            
    print(f"Wireframe Pyramid Void visualization saved to {filename}")




def main():
    args = parse_args()

    X, y = generate_angular_data(args.classes, args.points_per_class, args.dim, args.seed)
    centroids = np.array([np.mean(X[y == i], axis=0) for i in np.unique(y)])

    centroids = np.array([np.mean(X[y == i], axis=0) for i in np.unique(y)])
    centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids_norm

    sim_matrix = cosine_similarity(centroids_norm)
    # print('sim_matrix:', sim_matrix)

    k = args.k
    k_sim_sums = np.sort(sim_matrix, axis=1)[:, -(k+1):-1].sum(axis=1)
    print('k_sim_sums:', k_sim_sums)
    isolated_indices = np.argsort(k_sim_sums)
    print('isolated_indices:', isolated_indices)

    target_idx = isolated_indices[0]
    target_centroid = centroids[target_idx]
    target_sims = sim_matrix[target_idx]
    sorted_indices = np.argsort(target_sims)[::-1]
    neighbor_indices = sorted_indices[1:k+1]
    neighbor_centroids = centroids[neighbor_indices]
    # void_vector = find_void_direction(target_centroid, neighbor_centroids)
    void_vector = find_tangent_void_direction(target_centroid, neighbor_centroids)

    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
    filename = f"knn+pca_output_{args.dim}D_classes={args.classes}_points={args.points_per_class}_knn={args.k}_date={formatted_time}_seed={args.seed}"

    # 3. Visualization/Export
    if args.dim == 2:
        filename = '2D_embeddings.png'
        print('Saving chart of void spaces')
        save_2d_chart(filename, args.classes, X, y, centroids, dpi=200)

        print('Saving 2D chart of void spaces')
        save_void_visualization_2d(centroids, isolated_indices, target_idx, neighbor_indices, void_vector, f"{filename}.png")
    
    else: # 3D
        save_void_visualization_3d_wireframe(centroids, target_idx, neighbor_indices, void_vector, f"{filename}.ply")
        print('Saving 3D point cloud of void spaces')

    print("\nFinished!\n")


if __name__ == "__main__":
    main()