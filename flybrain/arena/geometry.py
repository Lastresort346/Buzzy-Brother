# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Original procedural meshes; no downloaded art assets."""
import math
from panda3d.core import Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, NodePath


def mesh(name, vertices, normals, faces):
    data = GeomVertexData(name, GeomVertexFormat.getV3n3(), Geom.UHStatic)
    v, n = GeomVertexWriter(data, 'vertex'), GeomVertexWriter(data, 'normal')
    for position, normal in zip(vertices, normals):
        v.addData3(*position)
        n.addData3(*normal)
    triangles = GeomTriangles(Geom.UHStatic)
    for face in faces:
        triangles.addVertices(*face)
    geom = Geom(data)
    geom.addPrimitive(triangles)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def box_mesh():
    vertices, normals, triangles = [], [], []
    for normal, corners in [((0,0,1), [(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]),
                            ((0,0,-1), [(-1,1,-1),(1,1,-1),(1,-1,-1),(-1,-1,-1)]),
                            ((0,-1,0), [(-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1)]),
                            ((0,1,0), [(1,1,-1),(-1,1,-1),(-1,1,1),(1,1,1)]),
                            ((1,0,0), [(1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)]),
                            ((-1,0,0), [(-1,1,-1),(-1,-1,-1),(-1,-1,1),(-1,1,1)])]:
        start = len(vertices)
        vertices.extend([tuple(x / 2 for x in p) for p in corners])
        normals.extend([normal] * 4)
        triangles.extend([(start,start+1,start+2),(start,start+2,start+3)])
    return mesh('cube', vertices, normals, triangles)


def sphere_mesh(segments=12, rings=8):
    vertices, normals, triangles = [], [], []
    for j in range(rings+1):
        lat = math.pi * j / rings
        for i in range(segments+1):
            lon = 2 * math.pi * i / segments
            p = (math.sin(lat)*math.cos(lon), math.sin(lat)*math.sin(lon), math.cos(lat))
            vertices.append(p)
            normals.append(p)
    for j in range(rings):
        for i in range(segments):
            a = j*(segments+1)+i
            b = a+segments+1
            triangles.extend([(a,b,a+1),(a+1,b,b+1)])
    return mesh('sphere', vertices, normals, triangles)


def cylinder_mesh(segments=16):
    vertices, normals, triangles = [], [], []
    for i in range(segments):
        angle, next_angle = 2*math.pi*i/segments, 2*math.pi*(i+1)/segments
        x,y = math.cos(angle),math.sin(angle)
        xx,yy = math.cos(next_angle),math.sin(next_angle)
        start = len(vertices)
        vertices.extend([(x,y,-.5),(xx,yy,-.5),(xx,yy,.5),(x,y,.5),
                         (0,0,.5),(x,y,.5),(xx,yy,.5),(0,0,-.5),(xx,yy,-.5),(x,y,-.5)])
        normals.extend([(x,y,0),(xx,yy,0),(xx,yy,0),(x,y,0)]+[(0,0,1)]*3+[(0,0,-1)]*3)
        triangles.extend([(start,start+1,start+2),(start,start+2,start+3),
                          (start+4,start+5,start+6),(start+7,start+8,start+9)])
    return mesh('cylinder', vertices, normals, triangles)
