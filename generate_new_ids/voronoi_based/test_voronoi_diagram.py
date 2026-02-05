from __future__ import annotations

import os, sys
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.spatial import Voronoi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Voronoi diagram for random points.")
    p.add_argument("--npoints", type=int, default=50, help="Number of random points.")
    p.add_argument("--dim", type=int, default=2, help="Dimension (>=2).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--out", type=str, default="voronoi_output", help="Output basename (without extension).")
    p.add_argument("--dpi", type=int, default=200, help="PNG DPI (dim=2 only).")
    return p.parse_args()


# Only import matplotlib if needed (dim=2).
# (Keeps 3D/ND usage lightweight.)
def _import_matplotlib():
    import matplotlib.pyplot as plt  # noqa: F401
    from scipy.spatial import voronoi_plot_2d  # noqa: F401
    return plt, voronoi_plot_2d


@dataclass
class VoronoiExport:
    points: np.ndarray          # (N, D)
    vertices: np.ndarray        # (M, D)
    ridge_points: np.ndarray    # (R, 2)
    ridge_vertices: List[List[int]]
    regions: List[List[int]]
    point_region: np.ndarray    # (N,)


def compute_voronoi(points: np.ndarray) -> VoronoiExport:
    v = Voronoi(points)
    return VoronoiExport(
        points=v.points,
        vertices=v.vertices,
        ridge_points=v.ridge_points,
        ridge_vertices=v.ridge_vertices,
        regions=v.regions,
        point_region=v.point_region,
    )


def save_npz(vexp: VoronoiExport, path: str) -> None:
    # Regions and ridge_vertices are ragged lists; store as object arrays.
    np.savez_compressed(
        path,
        points=vexp.points,
        vertices=vexp.vertices,
        ridge_points=vexp.ridge_points,
        ridge_vertices=np.array(vexp.ridge_vertices, dtype=object),
        regions=np.array(vexp.regions, dtype=object),
        point_region=vexp.point_region,
    )


def save_2d_png(vexp: VoronoiExport, path_png: str, dpi: int = 200) -> None:
    plt, voronoi_plot_2d = _import_matplotlib()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)

    # Re-create a SciPy Voronoi object for plotting convenience
    v = Voronoi(vexp.points)
    voronoi_plot_2d(v, ax=ax, show_vertices=False, line_width=1.0)

    ax.scatter(vexp.points[:, 0], vexp.points[:, 1], s=12)
    ax.set_aspect("equal", "box")
    ax.set_title(f"Voronoi diagram (N={vexp.points.shape[0]})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(path_png, dpi=dpi)
    plt.close(fig)


def _vtk_legacy_polydata(points: np.ndarray,
                         polygons: List[List[int]],
                         comments: str = "Voronoi ridge facets (finite only)") -> str:
    """
    Create a legacy VTK ASCII POLYDATA string.
    polygons: list of faces, each is list of vertex indices (>=3)
    """
    lines: List[str] = []
    lines.append("# vtk DataFile Version 3.0")
    lines.append(comments)
    lines.append("ASCII")
    lines.append("DATASET POLYDATA")

    # Points
    lines.append(f"POINTS {points.shape[0]} float")
    for p in points:
        # Ensure 3D coords for VTK
        if p.shape[0] == 3:
            x, y, z = p
        else:
            # Shouldn't happen here, but be safe
            x, y = p[:2]
            z = 0.0
        lines.append(f"{x:.9g} {y:.9g} {z:.9g}")

    # Polygons
    # VTK POLYGONS section needs: "POLYGONS numPolys totalIndexCount"
    # where each poly contributes (1 + nverts) integers.
    total_ints = sum(1 + len(face) for face in polygons)
    lines.append(f"POLYGONS {len(polygons)} {total_ints}")
    for face in polygons:
        lines.append(" ".join([str(len(face))] + [str(i) for i in face]))

    return "\n".join(lines) + "\n"


def save_3d_vtk_ridge_facets(vexp: VoronoiExport, path_vtk: str) -> None:
    """
    Exports Voronoi ridge facets as VTK PolyData polygons.

    Notes:
    - SciPy/ Qhull can produce "infinite" ridges: ridge_vertices contains -1.
      Those are skipped.
    - Vertex order for facets is provided by Qhull; it is generally usable for
      visualization (ParaView tolerates minor inconsistencies).
    """
    if vexp.vertices.shape[1] != 3:
        raise ValueError("save_3d_vtk_ridge_facets expects 3D vertices")

    polygons: List[List[int]] = []
    for rv in vexp.ridge_vertices:
        # rv is a list of vertex indices making up a polygonal facet; -1 means infinity.
        if not rv or any(idx < 0 for idx in rv):
            continue
        if len(rv) < 3:
            continue
        polygons.append(rv)

    vtk_text = _vtk_legacy_polydata(vexp.vertices, polygons)
    with open(path_vtk, "w", encoding="utf-8") as f:
        f.write(vtk_text)


def _vtk_legacy_polydata_with_points_and_polys(
    vertices3: np.ndarray,
    polygons: List[List[int]],
    sites3: np.ndarray,
    comments: str = "Voronoi facets (finite) + input sites",
) -> str:
    """
    Build a VTK legacy ASCII POLYDATA containing:
      - Voronoi vertices as POINTS (used by POLYGONS)
      - Input sites appended as additional POINTS and referenced by VERTICES

    Adds CELL_DATA arrays so you can color:
      cell_type = 0 for polygons (facets)
      cell_type = 1 for vertex-cells (sites)

    Also adds POINT_DATA array 'pscale' for sites (1.0 for sites, 0.0 for vertices),
    useful with ParaView 'Glyph' filter.
    """
    if vertices3.ndim != 2 or vertices3.shape[1] != 3:
        raise ValueError("vertices3 must be (M,3)")
    if sites3.ndim != 2 or sites3.shape[1] != 3:
        raise ValueError("sites3 must be (N,3)")

    M = vertices3.shape[0]
    N = sites3.shape[0]

    # Combine points: first Voronoi vertices, then sites
    all_pts = np.vstack([vertices3, sites3])

    lines: List[str] = []
    lines.append("# vtk DataFile Version 3.0")
    lines.append(comments)
    lines.append("ASCII")
    lines.append("DATASET POLYDATA")

    # POINTS
    lines.append(f"POINTS {M + N} float")
    for p in all_pts:
        x, y, z = p
        lines.append(f"{x:.9g} {y:.9g} {z:.9g}")

    # POLYGONS (Voronoi facets)
    total_poly_ints = sum(1 + len(face) for face in polygons)
    lines.append(f"POLYGONS {len(polygons)} {total_poly_ints}")
    for face in polygons:
        lines.append(" ".join([str(len(face))] + [str(i) for i in face]))

    # VERTICES for sites
    # Each vertex cell: "1 <ptId>"
    # ptId must reference the appended site points: M .. M+N-1
    lines.append(f"VERTICES {N} {2 * N}")
    for i in range(N):
        lines.append(f"1 {M + i}")

    # CELL_DATA: polygons + site-vertices
    n_cells = len(polygons) + N
    lines.append(f"CELL_DATA {n_cells}")
    lines.append("SCALARS cell_type int 1")
    lines.append("LOOKUP_TABLE default")
    # 0 = Voronoi facet polygons
    for _ in polygons:
        lines.append("0")
    # 1 = site points (vertex cells)
    for _ in range(N):
        lines.append("1")

    # POINT_DATA: mark which points are sites (for glyph scaling, etc.)
    lines.append(f"POINT_DATA {M + N}")
    lines.append("SCALARS is_site int 1")
    lines.append("LOOKUP_TABLE default")
    for _ in range(M):
        lines.append("0")
    for _ in range(N):
        lines.append("1")

    # Optional: glyph scale hint (0 for Voronoi vertices, 1 for sites)
    lines.append("SCALARS pscale float 1")
    lines.append("LOOKUP_TABLE default")
    for _ in range(M):
        lines.append("0")
    for _ in range(N):
        lines.append("1")

    return "\n".join(lines) + "\n"


def save_3d_vtk_with_sites(vexp: VoronoiExport, path_vtk: str) -> None:
    """
    Exports:
      - Finite Voronoi ridge facets as POLYGONS
      - Input sites as VERTICES
    In the same legacy VTK POLYDATA file.

    Visualize in ParaView:
      - Color by 'cell_type' (facets vs sites)
      - For point size:
          * simplest: Properties -> 'Point Size'
          * or use Filters -> 'Glyph' and scale by 'pscale'
    """
    if vexp.points.shape[1] != 3:
        raise ValueError("save_3d_vtk_with_sites expects 3D points")
    if vexp.vertices.shape[1] != 3:
        raise ValueError("save_3d_vtk_with_sites expects 3D vertices")

    polygons: List[List[int]] = []
    for rv in vexp.ridge_vertices:
        if not rv or any(idx < 0 for idx in rv):  # skip infinite ridges
            continue
        if len(rv) < 3:
            continue
        polygons.append(rv)

    vtk_text = _vtk_legacy_polydata_with_points_and_polys(
        vertices3=vexp.vertices,
        polygons=polygons,
        sites3=vexp.points,
        comments="3D Voronoi (finite facets) + input sites",
    )
    with open(path_vtk, "w", encoding="utf-8") as f:
        f.write(vtk_text)



import xml.etree.ElementTree as ET
import numpy as np


def save_3d_vtp_with_sites(vexp, path_vtp):
    """
    Writes a VTK XML PolyData (.vtp) with:
      - Voronoi facets as Polygons
      - Input sites as Vertices
      - CellData 'cell_type': 0=facet, 1=site
    Fully ParaView-filter compatible.
    """

    verts = vexp.vertices
    sites = vexp.points

    # Collect finite facets
    polygons = []
    for rv in vexp.ridge_vertices:
        if not rv or any(i < 0 for i in rv) or len(rv) < 3:
            continue
        polygons.append(rv)

    M = verts.shape[0]
    N = sites.shape[0]

    points = np.vstack([verts, sites])

    root = ET.Element("VTKFile", type="PolyData", version="0.1", byte_order="LittleEndian")
    polydata = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(
        polydata, "Piece",
        NumberOfPoints=str(M + N),
        NumberOfVerts=str(N),
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys=str(len(polygons)),
    )

    # ---- POINTS ----
    pts = ET.SubElement(piece, "Points")
    da = ET.SubElement(pts, "DataArray",
                       type="Float32", NumberOfComponents="3", format="ascii")
    da.text = "\n" + " ".join(f"{x} {y} {z}" for x, y, z in points) + "\n"

    # ---- POLYGONS ----
    polys = ET.SubElement(piece, "Polys")

    conn = ET.SubElement(polys, "DataArray", type="Int32", Name="connectivity", format="ascii")
    conn.text = "\n" + " ".join(str(i) for face in polygons for i in face) + "\n"

    offs = ET.SubElement(polys, "DataArray", type="Int32", Name="offsets", format="ascii")
    offset = 0
    offs_list = []
    for face in polygons:
        offset += len(face)
        offs_list.append(str(offset))
    offs.text = "\n" + " ".join(offs_list) + "\n"

    # ---- VERTICES (sites) ----
    verts_cell = ET.SubElement(piece, "Verts")

    conn_v = ET.SubElement(verts_cell, "DataArray",
                           type="Int32", Name="connectivity", format="ascii")
    conn_v.text = "\n" + " ".join(str(M + i) for i in range(N)) + "\n"

    offs_v = ET.SubElement(verts_cell, "DataArray",
                           type="Int32", Name="offsets", format="ascii")
    offs_v.text = "\n" + " ".join(str(i + 1) for i in range(N)) + "\n"

    # ---- CELL DATA ----
    cell_data = ET.SubElement(piece, "CellData", Scalars="cell_type")
    ct = ET.SubElement(cell_data, "DataArray",
                       type="Int32", Name="cell_type", format="ascii")
    ct.text = "\n" + " ".join(["0"] * len(polygons) + ["1"] * N) + "\n"

    ET.ElementTree(root).write(path_vtp, encoding="utf-8", xml_declaration=True)



def main() -> None:
    args = parse_args()

    if args.npoints < 2:
        raise SystemExit("--npoints must be >= 2")
    if args.dim < 2:
        raise SystemExit("--dim must be >= 2")

    rng = np.random.default_rng(args.seed)
    pts = rng.random((args.npoints, args.dim), dtype=np.float64)
    print('pts.shape:', pts.shape)
    # sys.exit(0)

    print('Computing Voronoi diagram...')
    vexp = compute_voronoi(pts)
    print('    Done!')

    base = f"{args.out}_npoints={args.npoints}_dim={args.dim}"
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    if args.dim == 2:
        png_path = base + ".png"
        save_2d_png(vexp, png_path, dpi=args.dpi)
        # Also save raw data
        save_npz(vexp, base + ".npz")
        print(f"Saved: {png_path}")
        print(f"Saved: {base}.npz (raw Voronoi data)")

    elif args.dim == 3:
        vtk_path = base + ".vtk"
        # save_3d_vtk_ridge_facets(vexp, vtk_path)
        save_3d_vtk_with_sites(vexp, vtk_path)

        # vtk_path = base + ".vtp"
        # save_3d_vtp_with_sites(vexp, vtk_path)
        
        save_npz(vexp, base + ".npz")
        print(f"Saved: {vtk_path}")
        print(f"Saved: {base}.npz (raw Voronoi data)")
        print("\nOpen the .vtk offline in ParaView (File → Open) or another VTK-capable viewer.")
        print("Tip: In ParaView, try 'Surface With Edges' to see facet boundaries clearly.")

    else:
        npz_path = base + ".npz"
        save_npz(vexp, npz_path)
        print(f"Saved: {npz_path}")
        print("For dim != 2/3, visualization is not included; the .npz contains vertices/regions/ridges.")


if __name__ == "__main__":
    main()
