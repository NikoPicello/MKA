import os
import sys

import cv2
import imageio
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.getcwd())


def load_obj_mesh(mesh_file,
                  with_normal=False,
                  with_texture=False,
                  with_texture_image=False,
                  with_vertex_color=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []
    color_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, 'r')
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            if with_vertex_color:
                color_data.append(list(map(float, values[4:7])))
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(
                        map(lambda x: int(x.split('/')[1]),
                            [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(
                        map(lambda x: int(x.split('/')[2]),
                            [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
        elif 'mtllib' in line.split():
            mtlname = line.split()[-1]
            mtlfile = os.path.join(os.path.dirname(mesh_file), mtlname)
            with open(mtlfile, 'r') as fmtl:
                mtllines = fmtl.readlines()
                for mtlline in mtllines:
                    # if mtlline.startswith('map_Kd'):
                    if 'map_Kd' in mtlline.split():
                        texname = mtlline.split()[-1]
                        texfile = os.path.join(
                            os.path.dirname(mesh_file), texname)
                        texture_image = cv2.imread(texfile)
                        texture_image = cv2.cvtColor(texture_image,
                                                     cv2.COLOR_BGR2RGB)
                        break

    vertices = np.array(vertex_data)
    faces = np.array(face_data).astype(np.int32) - 1
    colors = np.array(color_data)

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data).astype(np.int32) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        if with_texture_image:
            return vertices, faces, norms, face_normals, \
                uvs, face_uvs, texture_image
        else:
            return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data).astype(np.int32) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        # norms = np.array(norm_data)
        # norms = normalize_v3(norms)
        # face_normals = np.array(face_norm_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    if with_vertex_color:
        if colors.max() > 1 and colors.max() < 255:
            colors = colors / 255.
        return vertices, faces, colors

    return vertices, faces


def normalize_v3(arr):
    """Normalize a numpy array of 3 component vectors shape=(n,3)"""
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and
    # shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array
    # using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles,
    # by taking the cross product of the vectors v1-v0,
    # and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle.
    # The length of each normal is dependent the vertices,
    # we need to normalize these, so that
    # our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals,
    # one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading),
    # we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would
    # then contribute to every vertex, so we need to
    # normalize again afterwards.
    # The cool part, we can actually add the normals through
    # an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def distortPoints(p, dist):

    dist = np.reshape(dist, -1) if dist is not None else []
    k1 = dist[0] if len(dist) > 0 else 0
    k2 = dist[1] if len(dist) > 1 else 0
    p1 = dist[2] if len(dist) > 2 else 0
    p2 = dist[3] if len(dist) > 3 else 0
    k3 = dist[4] if len(dist) > 4 else 0
    k4 = dist[5] if len(dist) > 5 else 0
    k5 = dist[6] if len(dist) > 6 else 0
    k6 = dist[7] if len(dist) > 7 else 0
    x, y = p[..., 0], p[..., 1]
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + x2
    c = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / \
        (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    x_ = c * x + p1 * 2 * xy + p2 * (r2 + 2 * x2)
    y_ = c * y + p2 * 2 * xy + p1 * (r2 + 2 * y2)
    p[..., 0] = x_
    p[..., 1] = y_
    return p


la = np.array([0.2, 0.2, 0.2])
ld = np.array([1.0, 1.0, 1.0])
ls = np.array([1.0, 1.0, 1.0])
ldir = np.array([0.0, 0.0, 1.0])
fma = np.array([0.1, 0.1, 0.1])
fmd = np.array([0.25, 0.25, 0.25])
fms = np.array([0.60, 0.60, 0.60])
# fma = np.array([1, 0.7, 0.0])
# fmd = np.array([0.4, 0.2, 0.0])
# fms = np.array([0.7, 0.35, 0.0])
fmss = 2.0
# bma = np.array([0.85, 0.85, 0.85])
# bmd = np.array([0.85, 0.85, 0.85])
# bms = np.array([0.60, 0.60, 0.60])
# bmss = 100.0


def PhongLightening(n, v):

    def reflect(ambiguous_name_I, N):
        return ambiguous_name_I - 2.0 * N.dot(ambiguous_name_I) * N

    def normalize(vec):
        return vec / (np.linalg.norm(vec) + 1e-8)

    ldir_ = normalize(-ldir)
    fn = normalize(n)
    vdir = -normalize(v)
    frdir = reflect(-ldir_, fn)

    ka = la * fma
    kd = ld * fms
    ks = ls * fms

    fca = ka
    fcd = kd * max(fn.dot(ldir_), 0.0)
    fcs = ks * pow(max(vdir.dot(frdir), 0.0), fmss)
    fc = np.clip(fca + fcd + fcs, 0.0, 1.0)
    return fc


def rasterize(
        v,
        tri,
        size,
        img=None,
        zbuf=None,
        c=None,
        uv_tri=None,
        uv=None,
        tex=None,
        norm=None,
        K=np.identity(3),
        w2c=None,
        dist=None,
        white_bkgd=False,
):
    """
    Rasterize mesh and output its rendered image and depth
    v: vertices of mesh
    tri: faces of mesh
    c: per-vertex color of mesh
    size: size of the render image
    K: intrinsic matrix of the camera
    dist: distortion coefficient of camera
    white_bkgd: background in white or not
    """

    h, w = size
    if zbuf is None:
        zbuf = np.ones([h, w], v.dtype) * float('inf')
    if img is None:
        img = np.ones([h, w, 3], np.float32) if white_bkgd else np.zeros(
            [h, w, 3], np.float32)
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
        # print('ets')

    # preprocess the texture map and uv,
    # in order to use torch.nn.functional.sample_grid
    if tex is not None:
        # tex = torch.from_numpy(tex).permute((2,0,1))[None,...].float()
        uv = uv * 2 - 1  # convert to [-1,1]
        uv[:, 1] = -uv[:, 1]

    # convert mesh to camera coordinate
    if w2c is not None:
        v = v.dot(w2c[:3, :3].T) + w2c[:3, 3:].T

    # distort input point
    if dist is not None:
        valid = np.where(v[:, 2] >= 1e-6)[0]
        v_proj = v[valid, :2] / v[valid, 2:]
        v_proj = distortPoints(v_proj, dist)
        v[valid, :2] = v_proj * v[valid, 2:]
    # projection to image plane
    v_proj = v.dot(K.T)[:, :2] / np.maximum(v[:, 2:], 1e-6)
    # triangluar coordinates of projected faces
    va = v_proj[tri[:, 0], :2]
    vb = v_proj[tri[:, 1], :2]
    vc = v_proj[tri[:, 2], :2]
    # face roatation
    front = np.cross(vc - va, vb - va)
    # triangluar local patch uv
    umin = np.maximum(
        np.ceil(np.vstack((va[:, 0], vb[:, 0], vc[:, 0])).min(0)),
        0).astype(np.int32)
    umax = np.minimum(
        np.floor(np.vstack((va[:, 0], vb[:, 0], vc[:, 0])).max(0)),
        w - 1).astype(np.int32)
    vmin = np.maximum(
        np.ceil(np.vstack((va[:, 1], vb[:, 1], vc[:, 1])).min(0)),
        0).astype(np.int32)
    vmax = np.minimum(
        np.floor(np.vstack((va[:, 1], vb[:, 1], vc[:, 1])).max(0)),
        h - 1).astype(np.int32)
    # valid triangulars
    front = np.where(
        np.logical_and(np.logical_and(umin <= umax, vmin <= vmax),
                       front > 0))[0]

    for t in front:
        #       [coeffb]*[b-a]
        # [u-a]=[coeffc]*[c-a] ==>
        # u = coeffb * b + coeffc * c + (1-coeffb-coeffc) * a
        A = np.concatenate(
            (vb[t:t + 1] - va[t:t + 1], vc[t:t + 1] - va[t:t + 1]), 0)
        x, y = np.meshgrid(
            range(umin[t], umax[t] + 1), range(vmin[t], vmax[t] + 1))
        u = np.vstack((x.reshape(-1), y.reshape(-1))).T
        coeff = (u.astype(v.dtype) - va[t:t + 1, :]).dot(np.linalg.pinv(A))
        coeff = np.concatenate((1 - coeff.sum(1).reshape(-1, 1), coeff), 1)
        # interploate color and z
        z = coeff.dot(v[tri[t], 2])
        if c is not None:
            c_ = coeff.dot(c[tri[t], :])
        # Texture coloring
        elif uv_tri is not None and uv is not None:  # and tex is not None:
            tex_uv_ = coeff.dot(uv[uv_tri[t, :]])
            c_ = np.concatenate(
                [tex_uv_, np.zeros_like(tex_uv_[..., :1])], axis=-1)
            # tex_uv_ = torch.from_numpy(tex_uv_).reshape(1,-1,1,2).float()
            # c_ = torch.nn.functional.grid_sample(
            #       tex, tex_uv_).numpy().reshape(3,-1).T
        # Face coloring (Phong model)
        elif norm is None:
            vs = coeff.dot(v[tri[t]])
            n = np.cross(v[tri[t, 1]] - v[tri[t, 0]],
                         v[tri[t, 2]] - v[tri[t, 0]])
            c_ = []
            for vv in vs:
                c_.append(PhongLightening(n, vv))
            c_ = np.array(c_)
        # Vert coloring (Phong model)
        else:
            ns, vs = norm[tri[t]], v[tri[t]]
            cs = []
            for v_, n_ in zip(ns, vs):
                cs.append(PhongLightening(v_, n_))
            cs = np.array(cs)
            c_ = coeff.dot(cs)

        # for every pixel in the local patch,
        # if it lies in the triangular and not occluded
        for i, (x, y) in enumerate(u):
            if np.prod(coeff[i] >= -1e-6) > 0 and zbuf[y, x] > z[i]:
                zbuf[y, x] = z[i]
                img[y, x, :] = c_[i, :]
    return img, zbuf


if __name__ == '__main__':

    smpldir = sys.argv[1]
    imgdir = sys.argv[2]
    outdir = sys.argv[3]
    annotdir = sys.argv[4]
    view = int(sys.argv[5])

    smplnames = sorted(os.listdir(smpldir))
    imgnames = sorted(os.listdir(imgdir))
    assert len(smplnames) == len(imgnames)
    annots = np.load(annotdir, allow_pickle=True).item()['cams']
    os.makedirs(outdir, exist_ok=True)

    ms = 0.25
    # imsize = (2448, 2048) if view < 48 else (4096, 3000)
    # imsize = (int(imsize[0]*ms), int(imsize[1]*ms))

    imgs = []
    for imgname, smplname in tqdm(zip(imgnames, smplnames)):
        verts, faces, normal, fnormals = load_obj_mesh(
            os.path.join(smpldir, smplname), with_normal=True)
        img = cv2.imread(os.path.join(imgdir, imgname))
        img = cv2.resize(img, (int(img.shape[1] * ms), int(img.shape[0] * ms)))
        if '00' in annots.keys():
            K = annots['%02d' % view]['K'].copy().reshape(3, 3)
            c2w = annots['%02d' % view]['RT'].copy().reshape(4, 4)
            dist = annots['%02d' % view]['D'].copy().reshape(-1)
        else:
            views = sorted(os.listdir(os.path.dirname(imgdir)))
            vid = views.index('%02d' % view)
            K = np.array(annots['K'][vid]).reshape(3, 3)
            c2w = np.array(annots['RT'][vid]).reshape(4, 4)
            dist = np.array(annots['D'][vid]).reshape(-1)
        K[:2] *= ms
        img, _ = rasterize(
            verts,
            faces, (img.shape[0], img.shape[1]),
            norm=normal,
            img=img,
            K=K,
            dist=dist,
            w2c=np.linalg.inv(c2w))
        img = (img.clip(0, 1) * 255).astype(np.uint8)
        # cv2.imwrite(os.path.join(outdir, imgname), img)
        imgs.append(img)
    imageio.mimwrite(
        os.path.join(outdir, 'smplvis.mp4'), imgs, fps=15, quality=8)
