import matplotlib.pyplot as plt
import numpy as np


def get_topological_metrics(mesh, mesh_name, output_path='.'):
    #######################################
    # Generic stats

    print("\nNumber of vertices = {}".format(len(mesh.vertices)))
    print("Number of faces = {}".format(len(mesh.faces)))
    print("Area of mesh = {:.5f}".format(mesh.area))
    print("Mean area of a face = {}".format(np.mean(mesh.area_faces)))
    print("Standard deviation = {}".format(np.std(mesh.area_faces)))
    print("Largest area of a face = {}".format(max(mesh.area_faces)))

    #######################################
    # Smoothness
    # For each face check the neighbours for size difference. Big ratio is bad, should be a smooth transition

    print("\nCalculating area ratio for each adjacent face...")
    neighbour_ratios = [mesh.area_faces[face_pair[0]] / mesh.area_faces[face_pair[1]]
                        if mesh.area_faces[face_pair[0]] > mesh.area_faces[face_pair[1]]
                        else mesh.area_faces[face_pair[1]] / mesh.area_faces[face_pair[0]]
                        for face_pair in mesh.face_adjacency]
    print("\nMean area ratio of adjacent faces = {:.5f}".format(np.mean(neighbour_ratios)))
    print("Standard deviation = {:.5f}".format(np.std(neighbour_ratios)))
    print("Largest area ratio of adjacent faces = {:.5f}".format(max(neighbour_ratios)))

    #######################################
    # Aspect Ratio

    print("\nCalculating aspect ratios for mesh faces...")
    # AR = abc/((b+c-a)(c+a-b)(a+b-c))
    el = mesh.edges_unique_length
    aspect_ratio = np.array([(el[fe[0]] * el[fe[1]] * el[fe[2]]) / (
            (el[fe[1]] + el[fe[2]] - el[fe[0]]) * (el[fe[2]] + el[fe[0]] - el[fe[1]]) * (
            el[fe[0]] + el[fe[1]] - el[fe[2]])) for fe in mesh.faces_unique_edges])
    print("\nMean aspect ratio = {:.5f}".format(np.mean(aspect_ratio)))
    print("Standard deviation = {:.5f}".format(np.std(aspect_ratio)))
    # Should be between 10-30
    print("Largest aspect ratio (ideal 10-30) = {}".format(max(aspect_ratio)))

    perc_perfect_faces = round(sum(aspect_ratio <= 1) / float(len(aspect_ratio)), 4)
    print("\nPercentage of faces with ideal aspect ratio = {}".format(perc_perfect_faces))

    perc_good_faces = round(sum(aspect_ratio <= 3) / float(len(aspect_ratio)), 4)
    print("Percentage of faces with aspect ratio <= 3 = {:.5f}".format(perc_good_faces))

    perc_bad_faces = round(sum(aspect_ratio >= 10) / float(len(aspect_ratio)), 4)
    print("Percentage of faces with aspect ratio >= 10 = {:.5f}".format(perc_bad_faces))

    #######################################
    # Skewness

    print("\nCalculating skewness for mesh faces...")
    # Vertex angle for equilateral triangle: 60 degrees or pi*2/6
    ideal_angle = np.radians(60)
    face_skewness = np.array(
        [max([(max(face_angles) - ideal_angle) / (np.pi - ideal_angle), (ideal_angle - min(face_angles)) / ideal_angle])
         for face_angles in mesh.face_angles])

    print("\nMean skewness of faces = {:.5f}".format(np.mean(face_skewness)))
    print("Standard deviation = {:.5f}".format(np.std(face_skewness)))
    print("Largest deviation = {:.5f}".format(max(face_skewness)))

    # Skewness up to 0.5 is considered acceptable
    perc_good_faces = round(sum(face_skewness <= 0.5) / float(len(face_skewness)), 4)
    print("\nPercentage of faces with skewness <= 0.5 = {:.5f}".format(perc_good_faces))

    # Setup histogram plot
    mu = np.mean(face_skewness)
    sigma = np.std(face_skewness)
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(face_skewness, bins='auto')
    good_faces_idx = np.argwhere(bins <= 0.5)
    for idx in good_faces_idx:
        patches[idx[0]].set_facecolor('#3cb043')

    # Mark good triangles percentile
    ax.axvline(0.5, alpha=1, ymax=1, linestyle=":", color="#dc1c13")
    ax.text(0.5, max(n), str(round(100 * perc_good_faces)) + '%', size=10, alpha=1)
    # ax.text(0.5, max(n) / 10, 0.5, size=12, alpha=1)

    # Mark other percentiles
    percent = [0.05, 0.25, 0.50, 0.75, 0.95]
    # Calculate percentiles
    quants = np.quantile(face_skewness, percent)
    # [quantile, opacity, length]
    quants = [[quant, 0.5, 1] for quant in quants]
    # Plot the lines
    for idx, quant in enumerate(quants):
        ax.axvline(quant[0], alpha=quant[1], ymax=quant[2], linestyle=":", color="#dc1c13")
        # ax.text(quant[0], max(n), str(round(percent[idx] * 100)) + "%", size=10, alpha=0.8)
        # ax.text(quant[0], 100, round(quant[0], 2), size=10, alpha=0.8)

    # 'best fit' line, only works with density = true
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    # ax.plot(bins, y, '--')

    ax.set_ylabel('Number of faces')
    ax.set_xlabel('Face Skewness')
    ax.set_title(r'Histogram of Face Skewness: $\mu=$' + str(round(mu, 3)) + ', $\sigma=$' + str(round(sigma, 3)))
    fig.tight_layout()
    plt.savefig(output_path + '/topological_output/' + mesh_name + '_skewness_hist.png', bbox_inches='tight',
                format='png')
    fig.clf()
    plt.close()

    exit()
    #######################################
    # Hole detection
    print("\n\nCalculating mesh outline perimeter...")
    outline = mesh.outline()

    # outline.colors = [[255, 0, 0]] * len(outline.entities)
    # scene = trimesh.Scene([mesh, outline])
    # scene.show()

    print("\nTotal length of mesh outline = {:.5f}".format(outline.length))
    print("\nNumber of closed paths = {}".format(len(outline.paths)))
    print("\nNumber of edges not in a closed path = {}".format(len(outline.dangling)))

    path_lengths = [outline.entities[path[0]].length(outline.vertices) for path in outline.paths]
    print("\nMean closed path length = {:.5f}".format(np.mean(path_lengths)))
    print("Standard deviation = {:.5f}".format(np.std(path_lengths)))
    print("Largest closed path = {:.5f}".format(max(path_lengths)))
