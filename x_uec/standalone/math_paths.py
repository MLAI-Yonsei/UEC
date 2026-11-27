"""Exact frozen-path contractions; all computations retain shared weight noise."""
import torch
import torch.nn.functional as F

def rms(x,scale,eps):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*scale

def rms_vjp(x,scale,eps,coefficient):
    s=(x.square().mean(-1,keepdim=True)+eps).sqrt();a=coefficient*scale
    return a/s-x*(a*x).sum(-1,keepdim=True)/(x.shape[-1]*s**3)

def rms_jvp(x,scale,eps,tangent):
    s=(x.square().mean(-1,keepdim=True)+eps).sqrt()
    return scale[None,:,None]*(tangent/s[:,:,None]-x[:,:,None]*(x[:,:,None]*tangent).sum(1,keepdim=True)/(x.shape[-1]*s[:,:,None]**3))

def harrier_forward(weight,h,residual,constants):
    y=F.linear(h,weight)
    x=rms(y,constants['post_scale'],constants['post_eps'])+residual
    raw=rms(x,constants['final_scale'],constants['final_eps'])
    return raw,F.normalize(raw,p=2,dim=-1)

def harrier_vjp(weight,h,residual,c,coefficients):
    y=F.linear(h,weight);x=rms(y,c['post_scale'],c['post_eps'])+residual
    dx=rms_vjp(x,c['final_scale'],c['final_eps'],coefficients.detach())
    dy=rms_vjp(y,c['post_scale'],c['post_eps'],dx)
    return dy.T@h

def harrier_coordinate_jacobian(weight,h,residual,c,coordinates):
    d,width=weight.shape;rows=coordinates//width;cols=coordinates%width
    dy=F.one_hot(rows,num_classes=d).T.to(weight.dtype)[None]*h[:,cols][:,None,:]
    y=F.linear(h,weight);x=rms(y,c['post_scale'],c['post_eps'])+residual
    return rms_jvp(x,c['final_scale'],c['final_eps'],rms_jvp(y,c['post_scale'],c['post_eps'],dy))

def harrier_covariance(weight,h,residual,c,variance,full=False):
    y=weight@h;s=(y.square().mean()+c['post_eps']).sqrt()
    x=y/s*c['post_scale']+residual;scaled=x*c['final_scale'];norm=scaled.norm();u=scaled/norm
    a=c['final_scale']*c['post_scale']/(norm*s);v=y/(len(y)**.5*s)
    b=-(u*a)+torch.dot(u*a,v)*v;cc=-a*v;noise=variance@h.square()
    if full:
        j=torch.diag(a)+torch.outer(u,b)+torch.outer(cc,v)
        covariance=(j*noise[None])@j.T;covariance=(covariance+covariance.T)/2
        return covariance,covariance.diag().clone()
    diag=a.square()*noise+2*a*(u*b+cc*v)*noise
    diag+=u.square()*torch.sum(b.square()*noise)+cc.square()*torch.sum(v.square()*noise)+2*u*cc*torch.sum(b*v*noise)
    return None,diag

def gte_diagonal(weight,f,c,variance,chunk=32):
    """Exact pooled LN/L2 variance using a token Gram, no full output Jacobian.

    Group the rank-two LN and rank-one L2 terms before summing parameter
    variances. The token Gram sums over the SAME parameter at all tokens;
    cross-token terms are retained, rather than treating tokens independently.
    """
    active=f['mask'].bool();h=f['H'][active];residual=f['R'][active];n=len(h);d=len(weight)
    pre=F.linear(h,weight,c['bias'])+residual
    centered=pre-pre.mean(-1,keepdim=True);std=(centered.square().mean(-1,keepdim=True)+c['eps']).sqrt()
    x=centered/std;raw=(x*c['gamma']+c['beta']).mean(0);norm=raw.norm();u=raw/norm
    hs=h/(std*norm*n);hbar=hs.sum(0);gamma=c['gamma']
    q=(variance*hbar.square()).sum(1);k=u*gamma;aa=-u;bb=-(gamma-u*torch.sum(u*gamma))/d
    ww=-(x*gamma-u[None]*(x*k).sum(1,keepdim=True))/d
    # E[t,s] = sum_ab V[a,b] x[t,a]x[s,a] hs[t,b]hs[s,b].
    e=torch.zeros((n,n),device=weight.device,dtype=weight.dtype)
    for begin in range(0,d,chunk):
        stop=min(d,begin+chunk)
        factor=(x[:,begin:stop,None]*hs[:,None,:]*variance[begin:stop].sqrt()[None]).reshape(n,-1)
        e+=factor@factor.T
    fta=((variance*hbar[None])@hs.T).T
    cross=aa[None]*(fta*x*k).sum(1,keepdim=True)+bb[None]*(fta*x).sum(1,keepdim=True)+gamma[None]*x*fta
    base=gamma.square()*q+aa.square()*torch.sum(k.square()*q)+bb.square()*q.sum()+2*aa*bb*torch.sum(k*q)
    base+=2*gamma*(aa*k+bb)*q
    return base+2*(ww*cross).sum(0)+(ww*(e@ww)).sum(0)
