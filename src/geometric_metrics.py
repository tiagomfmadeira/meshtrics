from copy import deepcopy
from itertools import combinations

import numpy as np
import open3d as o3d
import pyrender
import trimesh


class Plane:

    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, first_guess, thresh=0.05, max_iteration=1):
        """ 
        Find the best equation for a plane.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param first_guess: Initial guess for the 3 points defining the plane.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param max_iteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers

        ---
        """
        best_eq = []
        best_inliers = []
        first_time = True

        for it in range(max_iteration):

            if first_time:
                first_time = False
                pt_samples = first_guess
            else:
                # Samples 3 random points
                id_samples = np.random.choice(best_inliers, 3)
                pt_samples = pts[id_samples]
                # print(pt_samples)

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane 
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            pt_id_inliers = []  # list of inliers ids
            dist_pt = (plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[
                3]) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is bigger than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            self.inliers = best_inliers
            self.equation = best_eq

        return self.equation, self.inliers


def find_plane(pcd, picked_points, max_iter):
    plane1 = Plane()
    plane_model, inliers = plane1.fit(np.asarray(pcd.points), picked_points, 0.01, max_iter)
    a, b, c, d = plane_model
    inlier_cloud = pcd.select_by_index(inliers)

    # print(f'Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0')
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([outlier_cloud, inlier_cloud])
    # o3d.visualization.draw_geometries([inlier_cloud])

    return {'equation': [a, b, c, d], 'inliers': inlier_cloud}


def get_roughness(mesh, visualize=False):
    print("Computing mesh curvature...")
    mesh = deepcopy(mesh)
    ori_curv = mesh.vertex_defects
    # q_min, q_max = np.percentile(q, [6, 94])
    # q = np.array([max(min(x, q_max), q_min) for x in q])
    #
    # vect_col_map = trimesh.visual.color.interpolate(q, color_map='jet')
    # mesh.visual.vertex_colors = vect_col_map
    # mesh.show(viewer='gl')

    smooth_mesh = deepcopy(mesh)
    trimesh.smoothing.filter_taubin(smooth_mesh)
    smooth_curv = smooth_mesh.vertex_defects
    # q_min, q_max = np.percentile(q, [6, 94])
    # q = np.array([max(min(x, q_max), q_min) for x in q])
    # vect_col_map = trimesh.visual.color.interpolate(q, color_map='jet')
    # mesh.visual.vertex_colors = vect_col_map
    # mesh.show(viewer='gl')

    roughness = [np.sqrt((ori - smooth) ** 2) for (ori, smooth) in zip(ori_curv, smooth_curv)]

    if visualize:
        q_min, q_max = np.percentile(roughness, [10, 90])
        q = np.array([max(min(x, q_max), q_min) for x in roughness])
        vect_col_map = trimesh.visual.color.interpolate(q, color_map='gist_yarg')
        # vect_col_map = trimesh.visual.color.interpolate(q, color_map='jet')
        mesh.visual.vertex_colors = vect_col_map
        # base_mesh.show(viewer='gl')
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial())
        for primitive in py_mesh.primitives:
            # primitive.material = None
            primitive.color_0 = vect_col_map
        scene = pyrender.Scene(bg_color=[255, 255, 255], ambient_light=[2.5, 2.5, 2.5])

        # Intrinsic camera matrix
        with open('statue/intrinsics.txt', 'r') as f:
            K = np.array([[float(num) for num in line.split(' ')] for line in f])[:3, :3]

        # Extrinsic camera matrix
        with open('statue/6.txt', 'r') as f:
            cam_matrix = np.array([[float(num) for num in line.split(' ')] for line in f])[:4, :4]
        cam_matrix = np.linalg.inv(cam_matrix)

        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1],
                                           cx=K[0, 2], cy=K[1, 2],
                                           znear=0.001, zfar=10000)
        scene.add(camera, pose=cam_matrix)

        scene.add(py_mesh)
        pyrender.Viewer(scene)

    print("\nMean vertex defect = {}".format(np.mean(roughness)))
    print("Standard deviation = {}".format(np.std(roughness)))
    print("Largest deviation = {}".format(abs(max(roughness) - np.mean(roughness))))

    return np.mean(roughness)


def get_planarity(mesh, plane):
    [a, b, c, _] = plane

    angles = [min(get_angle([a, b, c], normal), get_angle([a, b, c], -normal)) for normal in mesh.face_normals]
    weighted_angles = np.sum([mesh.area_faces[idx] / mesh.area * angle for idx, angle in enumerate(angles)])

    print("\nComputing " + str(len(angles)) + " face angles...")
    print("Area weighted angular difference to reference =")
    print("{:.3f}\N{DEGREE SIGN}".format(weighted_angles))
    # print("Standard deviation= ")
    # print(np.std(angles))
    print("Largest angle = ")
    print("{:.3f}\N{DEGREE SIGN}".format(max(angles)))


def get_angle(v1, v2):
    a1, b1, c1 = v1
    a2, b2, c2 = v2

    d = (a1 * a2 + b1 * b2 + c1 * c2)
    e1 = np.math.sqrt(a1 * a1 + b1 * b1 + c1 * c1)
    e2 = np.math.sqrt(a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = np.math.degrees(np.math.acos(d))

    return A


def get_orthogonality(planes):
    plane_comb = combinations(planes, 2)

    for comb in plane_comb:
        A = get_angle(comb[0]['equation'][:3], comb[1]['equation'][:3])

        print("Angle is {:.3f}\N{DEGREE SIGN}".format(A))

        o3d.visualization.draw_geometries([comb[0]['inliers'], comb[1]['inliers']])
