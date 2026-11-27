import torch
import torch.nn.functional as F

def last_indices(mask):
    if bool((mask.sum(-1)==0).any()):raise ValueError('Empty attention mask')
    return (torch.arange(mask.shape[-1],device=mask.device)[None]*mask.long()).max(-1).values

def forward(weight,h,residual,scale,eps):
    x=F.linear(h,weight)+residual
    raw=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*scale
    return raw,F.normalize(raw,p=2,dim=-1)

def raw_vjp(weight,h,residual,scale,eps,coefficients):
    x=F.linear(h,weight)+residual;rms=(x.square().mean(-1,keepdim=True)+eps).sqrt()
    a=coefficients.detach()*scale
    dx=a/rms-x*(a*x).sum(-1,keepdim=True)/(x.shape[-1]*rms**3)
    return dx.T@h

def coordinate_jacobian(weight,h,residual,scale,eps,coordinates):
    d,width=weight.shape;rows=coordinates//width;cols=coordinates%width
    x=F.linear(h,weight)+residual;rms=(x.square().mean(-1,keepdim=True)+eps).sqrt()
    selector=F.one_hot(rows,num_classes=d).T.to(weight.dtype)
    return scale[None,:,None]*(selector[None]/rms[:,:,None]
        -x[:,:,None]*x[:,rows][:,None,:]/(d*rms[:,:,None]**3))*h[:,cols][:,None,:]

def output_covariance(weight,h,residual,scale,eps,variance):
    # RMS scalar cancels under finalL2. Independentparameterrows yield diagonalpreactivationC.
    x=weight@h+residual;scaled=x*scale;norm=torch.linalg.norm(scaled);unit=scaled/norm
    j=(torch.eye(len(x),device=x.device,dtype=x.dtype)-torch.outer(unit,unit))/norm*scale[None,:]
    c=variance@h.square()
    result=(j*c[None,:])@j.T
    return (result+result.T)/2

def output_diagonal(weight,h,residual,scale,eps,variance):
    x=weight@h+residual;scaled=x*scale;norm=torch.linalg.norm(scaled);unit=scaled/norm
    c=(variance@h.square())*scale.square()/norm.square()
    # diag((I-uuT)diag(c)(I-uuT)), exact.
    return c*(1-2*unit.square())+unit.square()*(unit.square()*c).sum()
