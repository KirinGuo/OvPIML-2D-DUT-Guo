"""Raw Q4 oversampling basis. MATLAB-compatible BL/BR/TR/TL and y-fast rows."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

CORE = 10
HALO = 6
SIDES = ("left", "right", "bottom", "top")
CORNER_BITS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


@lru_cache(None)
def q4_stiffness(dx, dy, nu=0.3):
    """Eight by eight K for a rectangle, unit E and thickness, plane stress."""
    signs = 2 * CORNER_BITS - 1
    D = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu**2)
    K = np.zeros((8, 8))
    for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3), 1/np.sqrt(3)):
            gx = signs[:, 0] * (1 + eta*signs[:, 1]) / (2*dx)
            gy = signs[:, 1] * (1 + xi*signs[:, 0]) / (2*dy)
            B = np.zeros((3, 8))
            B[0, ::2], B[1, 1::2] = gx, gy
            B[2, ::2], B[2, 1::2] = gy, gx
            K += B.T @ D @ B * (dx*dy/4)
    return K


def _dofs(nodes):
    return (2*np.asarray(nodes).reshape(-1, 1) + np.arange(2)).reshape(-1)


def _axis(low, high, core, halo):
    lo, hi = (0 if low else -halo), (core if high else core + halo)
    coord = np.arange(lo, hi+1)
    if not low:
        coord = coord[coord != 1]
    if not high:
        coord = coord[coord != core-1]
    return coord


@dataclass
class Geometry:
    type_id: int
    core: int
    halo: int
    x: np.ndarray
    y: np.ndarray
    xy: np.ndarray
    touches: tuple
    side_nodes: tuple
    corner_nodes: np.ndarray
    red_nodes: np.ndarray
    red_x: np.ndarray
    red_y: np.ndarray
    shape: tuple
    red_shape: tuple
    input_slice: tuple
    ij: tuple
    unit_k: np.ndarray


@lru_cache(None)
def geometry(type_id, core=CORE, halo=HALO, nu=0.3):
    if type_id not in range(1,10) or core < 4 or halo < 0:
        raise ValueError("Expected type 1..9, core >=4, and halo >=0")
    col, row = (type_id-1) % 3, (type_id-1) // 3
    touches = (col==0, col==2, row==2, row==0)
    x = _axis(*touches[:2], core, halo)
    y = _axis(*touches[2:], core, halo)
    nx, ny = len(x), len(y)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    xy = np.column_stack([XX.ravel(), YY.ravel()])
    side_nodes = (XX.ravel()==x[0], XX.ravel()==x[-1], YY.ravel()==y[0], YY.ravel()==y[-1])
    vertices = np.array([0, (nx-1)*ny, nx*ny-1, ny-1])
    red_nodes = np.flatnonzero(np.all((xy >= 0) & (xy <= core), axis=1))
    red_x, red_y = x[(x>=0)&(x<=core)], y[(y>=0)&(y<=core)]
    # Element index traverses y fastest as in the original MATLAB sparse assembly.
    cell_origin = (np.arange(nx-1)[:,None]*ny + np.arange(ny-1)).ravel()
    dofs = (2*(cell_origin[:,None,None] + np.array([0,ny,ny+1,1])[None,:,None])
            + np.arange(2)[None,None,:]).reshape(-1,8)
    ij = (np.repeat(dofs,8,axis=1).ravel(), np.tile(dofs,(1,8)).ravel())
    widths = np.repeat(np.diff(x),ny-1)
    heights = np.tile(np.diff(y),nx-1)
    ke = np.empty((len(cell_origin),64))
    for dx, dy in set(zip(widths,heights)):
        ke[(widths==dx)&(heights==dy)] = q4_stiffness(dx,dy,nu).ravel()
    return Geometry(type_id,core,halo,x,y,xy,touches,side_nodes,vertices,
                    red_nodes,red_x,red_y,(int(y[-1]-y[0]),int(x[-1]-x[0])),
                    (len(red_y),len(red_x)),
                    (slice(y[0]+halo,y[-1]+halo),slice(x[0]+halo,x[-1]+halo)),ij,ke)


def stiffness(material_yx, geom):
    material = np.asarray(material_yx,dtype=np.float64)
    if material.shape != geom.shape or not np.isfinite(material).all() or material.min() <= 0:
        raise ValueError(f"Expected positive finite E/E0 on clipped auxiliary grid {geom.shape}")
    # Area-average cells merged across each overlap strip before assembly.
    ix, iy = geom.x[:-1]-geom.x[0],geom.y[:-1]-geom.y[0]
    cell_e = np.add.reduceat(np.add.reduceat(material,iy,axis=0),ix,axis=1)
    cell_e /= np.diff(geom.y)[:,None]*np.diff(geom.x)[None,:]
    N = 2*len(geom.xy)
    return coo_matrix(((geom.unit_k*cell_e.T.reshape(-1,1)).ravel(),geom.ij),
                      shape=(N,N)).tocsc()


@lru_cache(None)
def _boundary(type_id, clamp_side, core, halo, nu):
    g = geometry(type_id,core,halo,nu)
    n = len(g.xy)
    actual = np.zeros(n,bool)
    artificial = np.zeros(n,bool)
    fixed = np.zeros(n,bool)
    if clamp_side != "none" and (clamp_side not in SIDES or not g.touches[SIDES.index(clamp_side)]):
        raise ValueError("Clamp must touch an actual edge of this block")
    for name,physical,side in zip(SIDES,g.touches,g.side_nodes):
        if name == clamp_side:
            fixed |= side
        elif physical:
            actual |= side
        else:
            artificial |= side
    vertices = np.zeros(n,bool)
    vertices[g.corner_nodes] = True
    controlled = (artificial & ~actual) | vertices | fixed
    xc = (g.xy[:,0]-g.x[0])/(g.x[-1]-g.x[0])
    yc = (g.xy[:,1]-g.y[0])/(g.y[-1]-g.y[0])
    N = np.column_stack([(1-xc)*(1-yc),xc*(1-yc),xc*yc,(1-xc)*yc])
    u = np.zeros((2*n,8))
    # Free physical edges have only auxiliary corner prescriptions.
    N[~controlled | fixed] = 0
    u[::2,::2] = N
    u[1::2,1::2] = N
    # Coarse corner coefficients on the clamped wall are eliminated globally.
    fixed_columns = _dofs(np.flatnonzero(fixed[g.corner_nodes]))
    u[:,fixed_columns] = 0
    return _dofs(np.flatnonzero(controlled)),_dofs(np.flatnonzero(~controlled)),u,fixed,actual


def raw_basis(material_yx,type_id,clamp_side="none",*,core=CORE,halo=HALO,nu=0.3,details=False):
    """Crop the FP64 auxiliary solution with eight columns; no corner transformation."""
    g = geometry(int(type_id),core,halo,nu)
    K = stiffness(material_yx,g)
    prescribed,free,u0,fixed,actual = _boundary(int(type_id),clamp_side,core,halo,nu)
    u = u0.copy()
    rhs = -(K[free][:,prescribed] @ u[prescribed])
    u[free] = splu(K[free][:,free]).solve(rhs)
    result = u[_dofs(g.red_nodes)]
    if details:
        residual = K[free] @ u
        scale = np.linalg.norm(rhs,axis=0)
        relres = np.linalg.norm(residual,axis=0)/np.where(scale>0,scale,1)
        return result, dict(full=u,fixed=fixed,free_physical=actual,prescribed=prescribed,
                            free=free,max_residual=float(relres.max()))
    return result


def encode_input(material_yx,type_id):
    """[physical E/E0, valid] on a nominal 22x22 array, y increasing from row 0."""
    g = geometry(int(type_id))
    E = np.asarray(material_yx,dtype=np.float32)
    if E.shape != g.shape:
        raise ValueError(f"Expected clipped window {g.shape}, got {E.shape}")
    data = np.zeros((2,CORE+2*HALO,CORE+2*HALO),np.float32)
    ys,xs = g.input_slice
    data[0,ys,xs],data[1,ys,xs] = E,1
    return data


def encode_target(basis,type_id):
    ny,nx = geometry(int(type_id)).red_shape
    source = np.asarray(basis).reshape(nx,ny,2,8)
    data = np.zeros((16,CORE,CORE),np.float32)
    data[:,:ny,:nx] = source.transpose(3,2,1,0).reshape(16,ny,nx)
    mask = np.zeros((1,CORE,CORE),np.uint8)
    mask[:,:ny,:nx] = 1
    return data,mask


def decode_target(target,type_id):
    ny,nx = geometry(int(type_id)).red_shape
    return np.asarray(target)[:,:ny,:nx].reshape(8,2,ny,nx).transpose(3,2,1,0).reshape(2*nx*ny,8)
