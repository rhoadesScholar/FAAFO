"""Summary figure: phantom, reconstructed plane with grid overlay, error maps, XZ sections."""
import numpy as np, pickle, os
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from .geom import ZT, ZS, ZV, Y, X
from .phantom import CRACK, crack_dist
_cands = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf']
FB = next((f for f in _cands if os.path.exists(f)), None)
def _font(sz): return ImageFont.truetype(FB, sz) if FB else ImageFont.load_default()
FL = _font(30); FS = _font(19)
def U8(a): return Image.fromarray(np.clip(a,0,255).astype(np.uint8))
def g3(a): return np.stack([a]*3,-1)
def title(im,t):
    c=Image.new('RGB',(im.width,im.height+36),'white'); c.paste(im,(0,36)); dr=ImageDraw.Draw(c)
    f=FS
    while FB and dr.textlength(t,font=f)>im.width-8 and f.size>12: f=_font(f.size-1)
    dr.text((6,8),t,fill=(30,30,30),font=f); return c
def lab(im,t,sub=None):
    h=48+(34 if sub else 0); c=Image.new('RGB',(im.width,im.height+h),'white'); c.paste(im,(0,h)); dr=ImageDraw.Draw(c)
    f=FL
    while FB and dr.textlength(t,font=f)>im.width-16 and f.size>16: f=_font(f.size-1)
    dr.text((10,8),t,fill='black',font=f)
    if sub:
        f=FS
        while FB and dr.textlength(sub,font=f)>im.width-16 and f.size>12: f=_font(f.size-1)
        dr.text((10,50),sub,fill=(90,90,90),font=f)
    return c
def hstack(ps,g=14):
    H_=max(p.height for p in ps); c=Image.new('RGB',(sum(p.width for p in ps)+g*(len(ps)-1),H_),'white'); x=0
    for p in ps: c.paste(p,(x,0)); x+=p.width+g
    return c
def vstack(ps,path,g=16):
    Wc=max(p.width for p in ps); c=Image.new('RGB',(Wc,sum(p.height for p in ps)+g*len(ps)),'white'); y=0
    for p in ps: c.paste(p,(0,y)); y+=p.height+g
    c.save(path); print(path,c.size)
def heat(e,valid,vmax=10):
    r=np.clip(np.nan_to_num(e.astype(float))/vmax,0,1); im=np.stack([255*r,255*(1-r)*0.7,60*(1-r)],-1); im[~valid]=[70,70,190]; return im.astype(np.uint8)
def grid_overlay(R):
    o=g3(R).astype(float); g=np.zeros(R.shape,bool); g[:, ::50]=True; g[::50,:]=True; g=ndi.binary_dilation(g)
    o[g]=0.45*o[g]+0.55*np.array([235,20,160]); return o
def make_figure(phantom_path, results_path, out):
    d=pickle.load(open(phantom_path,'rb')); res=pickle.load(open(results_path,'rb')); T=d['T']
    A=np.where(d['tA'],d['A'],0); B=np.where(d['tB'],d['B'],0)
    zA=20; yc=int(CRACK[90,1])
    names=[('CONNECTED','A. Connected mesh per slab (today)'),('JOINT_EXT','B. Joint: cut pieces + seam extension + 14 crack anchors + strip matcher'),('JOINT_DENSE','C. Joint + dense crack-face pairs')]
    names=[(n,t) for n,t in names if n in res]
    dcr=crack_dist(np.stack(np.mgrid[0:Y,0:X][::-1],-1).reshape(-1,2)).reshape(Y,X)
    zz=np.arange(ZT)[:,None,None]*np.ones((1,Y,X),np.int8)
    near_crack=(np.broadcast_to(dcr,(ZT,Y,X))<80)&(zz<ZS); near_seam=np.abs(zz-ZS)<6
    def st(n,m): e=res[n]['err'].astype(float); v=res[n]['valid']&np.isfinite(e); mm=m&v; return np.median(e[mm]),np.percentile(e[mm],90)
    P=[lab(hstack([title(U8(g3(T[zA])),'Truth, plane 20 (slab A)'),title(U8(g3(A[26])),'Slab A as imaged (cracked, tilted, bent)'),title(U8(g3(B[26])),'Slab B as imaged (rotated, shifted, bent)')]),
       '1. Two-slab phantom (80 × 400 × 600 vx): slab A cracked, slab B intact; each sectioned, deformed and imaged on its own',
       'solver inputs: the two slab volumes, detected surfaces, 8 seam anchors per crack side, 14 crack anchors; seam pairs then grown by warp-and-match extension on the faces')]
    P.append(lab(hstack([title(U8(grid_overlay(res[n]['out'][zA])),t) for n,t in names]),'2. Reconstructed slab A, plane 20 (magenta = reference grid)'))
    rows=[]
    for n,t in names:
        mc,pc=st(n,near_crack); ms,ps=st(n,near_seam)
        rows.append(title(U8(heat(res[n]['err'][zA],res[n]['valid'][zA])),f'near crack: median {mc:.1f} / 90th {pc:.1f} vx · seam band 90th {ps:.1f} vx'))
    P.append(lab(hstack(rows),'3. Displacement error vs truth, plane 20 (red ≥ 10 vx, blue = unfilled)'))
    def sec(v): return np.kron(v[:,yc,:],np.ones((3,2)))
    rows=[title(U8(g3(sec(T))),'Truth XZ section through the crack and the seam')]
    for n,t in names: rows.append(title(U8(g3(sec(res[n]['out']))),t))
    P.append(lab(hstack(rows[:2]),'4. XZ section: the seam at z = 40, the crack through slab A'))
    P.append(hstack(rows[2:]))
    vstack(P,out)
    for n in res: print(n,'near crack',np.round(st(n,near_crack),2),'seam',np.round(st(n,near_seam),2))
