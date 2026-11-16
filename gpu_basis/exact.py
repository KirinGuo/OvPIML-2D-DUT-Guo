"""FP64 sparse auxiliary solves for only the owners routed to numerical bases."""
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

from basis.local import CORE, HALO, _boundary, geometry


@lru_cache(None)
def _plan(tid, side):
    g = geometry(tid)
    fixed, free, prescribed, _, _ = _boundary(tid, side, CORE, HALO, 0.3)
    n = len(free)
    rows, cols = g.ij
    element = np.repeat(np.arange(len(g.unit_k)),64)
    free_index = np.full(len(prescribed), -1, np.int32)
    free_index[free] = np.arange(n)
    rr, cc = free_index[rows], free_index[cols]
    both = (rr >= 0) & (cc >= 0)
    positions, slot = np.unique(cc[both]*n + rr[both],return_inverse=True)
    indices = np.asarray(positions % n, dtype=np.int32)
    indptr = np.r_[0,np.cumsum(np.bincount(positions//n,minlength=n))].astype(np.int32)
    coeff = g.unit_k.ravel()
    mixed = (rr>=0) & (cc<0)
    # Each column of RHS is a linear combination of the physical element moduli.
    vals = -coeff[mixed,None] * prescribed[cols[mixed],:]
    i, j = np.nonzero(vals)
    rhs = coo_matrix((vals[i,j],(8*rr[mixed][i]+j,element[mixed][i])),
                     shape=(8*n,len(g.unit_k))).tocsr()
    red = (2*g.red_nodes[:,None]+np.arange(2)).ravel()
    return (g,n,slot,element[both],coeff[both],indices,indptr,rhs,
            red,free_index[red],prescribed[red,:].copy())


def solve_pages(windows, type_id, clamp_side):
    """Actual E/E0 windows [y,x,owner], returned pages [200,8,owner]."""
    tid, side = int(type_id), str(clamp_side)
    a = np.asarray(windows, dtype=np.float64)
    if a.ndim == 2:
        a = a[:,:,None]
    plan = _plan(tid,side)
    g = plan[0]
    if a.shape[:2] != g.shape or not np.isfinite(a).all() or a.min() <= 0:
        raise ValueError(f"Invalid material in exact {g.shape} auxiliary window")
    if a.shape[2] >= 96:
        # Independent sparse systems share only immutable geometry; return in owner order.
        chunks = np.array_split(a, 4, axis=2)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda chunk: _solve_group(chunk, plan), chunks))
        return np.asfortranarray(np.concatenate(results, axis=2))
    return _solve_group(a,plan)


def _solve_group(a, plan):
    g,n,slot,elem,coeff,indices,indptr,rhs,red,ridx,base = plan
    blocks = a.shape[2]
    ix, iy = g.x[:-1]-g.x[0], g.y[:-1]-g.y[0]
    modulus = np.add.reduceat(np.add.reduceat(a,iy,axis=0),ix,axis=1)
    modulus /= (np.diff(g.y)[:,None]*np.diff(g.x)[None,:])[:,:,None]
    modulus = modulus.transpose(1,0,2).reshape(len(g.unit_k),blocks)
    rhs_all=(rhs@modulus).reshape(n,8,blocks)
    eligible=ridx>=0
    out=np.zeros((200,8,blocks),np.float32,order="F")
    for k in range(blocks):
        data = np.bincount(slot,weights=coeff*modulus[elem,k],minlength=len(indices))
        K = csc_matrix((data,indices,indptr),shape=(n,n),copy=False)
        solution = splu(K).solve(rhs_all[:,:,k])
        field = base.copy()
        field[eligible] = solution[ridx[eligible]]
        if not np.isfinite(field).all():
            raise FloatingPointError("Nonfinite numerical raw8 basis")
        out[:len(red),:,k] = field
    return out
