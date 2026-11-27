"""Differentiable final-block split, per-pair EF, exact linearized covariance."""
import torch
import torch.nn.functional as F

def raw_vjp(weight,features,constants,coefficients):
    """Shared final-weight VJP through all tokens and pooled raw outputs."""
    h=torch.cat([f['H'][f['mask'].bool()] for f in features]);r=torch.cat([f['R'][f['mask'].bool()] for f in features])
    lengths=torch.tensor([int(f['mask'].sum()) for f in features],device=weight.device)
    a=torch.repeat_interleave(coefficients/lengths[:,None],lengths,dim=0)*constants['gamma']
    pre=F.linear(h,weight,constants['bias'])+r;centered=pre-pre.mean(-1,keepdim=True)
    std=(centered.square().mean(-1,keepdim=True)+constants['eps']).sqrt();x=centered/std
    delta=(a-a.mean(-1,keepdim=True)-x*(a*x).mean(-1,keepdim=True))/std
    return delta.T@h

def coordinate_jacobian(weight,features,constants,coordinates):
    """Raw pooled output derivatives for selected final-weight coordinates."""
    d,width=weight.shape;rows=coordinates//width;cols=coordinates%width
    selector=F.one_hot(rows,num_classes=d).T.to(weight.dtype);result=[]
    for f in features:
        active=f['mask'].bool();h=f['H'][active];r=f['R'][active]
        pre=F.linear(h,weight,constants['bias'])+r;std=(pre.var(-1,unbiased=False,keepdim=True)+constants['eps']).sqrt()
        x=(pre-pre.mean(-1,keepdim=True))/std;term=h[:,cols]/std/len(h);summed=term.sum(0)
        result.append(constants['gamma'][:,None]*(selector*summed[None]-summed[None]/d-(x.T@(x[:,rows]*term))/d))
    return torch.stack(result)


def split_forward(weight,features,constants):
    pre=F.linear(features['H'],weight,constants['bias'])+features['R']
    hidden=F.layer_norm(pre,(weight.shape[0],),constants['gamma'],constants['beta'],constants['eps'])
    mask=features['mask']
    raw=(hidden*mask[:,None]).sum(0)/mask.sum()
    return raw,raw/torch.linalg.vector_norm(raw)


def pair_score_gradients(weight,queries,passages,labels,constants):
    """Gradients of log likelihood, both endpoints combined before EF square."""
    def score(w,qh,qr,qmask,ph,pr,pmask):
        q,_=split_forward(w,dict(H=qh,R=qr,mask=qmask),constants)
        p,_=split_forward(w,dict(H=ph,R=pr,mask=pmask),constants)
        return torch.dot(q,p)
    derivative=torch.func.grad_and_value(score,argnums=0)
    gradients,scores=torch.vmap(derivative,in_dims=(None,0,0,0,0,0,0))(
        weight,queries['H'],queries['R'],queries['mask'],passages['H'],passages['R'],passages['mask'])
    # Avoid subtracting sigmoid(large positive) from1 and losing positive-label
    # scores prematurely. Negative-label likelihood score is -sigmoid(logit).
    residual=torch.where(labels>.5,torch.sigmoid(-scores),-torch.sigmoid(scores))
    return gradients*residual[:,None,None],scores


def normalization_jacobian(raw):
    norm=torch.linalg.vector_norm(raw);unit=raw/norm
    return (torch.eye(len(raw),dtype=raw.dtype,device=raw.device)-torch.outer(unit,unit))/norm


def predictive_covariance(weight,features,constants,parameter_variance,chunk_rows=16,
                          full=False,keep_factor=False):
    """Compute J diag(V) J^T through residual/LN/shared tokens/pool/L2.

    For full output, stream raw-mean Jacobian rows into ONE preallocated
    variance-weighted factor, then apply the full L2 Jacobian. For diagonal
    only, directly differentiate normalized outputs and reduce each row.
    """
    w=weight.detach().requires_grad_(True)
    raw,mean=split_forward(w,features,constants)
    output=raw if full else mean
    d=output.numel();sqrt_var=torch.sqrt(parameter_variance.reshape(-1))
    identity=torch.eye(d,dtype=w.dtype,device=w.device)
    factor=torch.empty((d,w.numel()),dtype=w.dtype,device=w.device) if full else None
    diagonal=torch.empty(d,dtype=w.dtype,device=w.device)
    for begin in range(0,d,chunk_rows):
        stop=min(d,begin+chunk_rows)
        grad=torch.autograd.grad(output,w,grad_outputs=identity[begin:stop],is_grads_batched=True,
                                 retain_graph=stop<d,create_graph=False)[0].detach().reshape(stop-begin,-1)
        grad.mul_(sqrt_var)
        if full:factor[begin:stop].copy_(grad)
        else:diagonal[begin:stop]=grad.square().sum(1)
        del grad
    pre_cov=cov=None
    if full:
        pre_cov=factor@factor.T
        jac=normalization_jacobian(raw.detach())
        cov=jac@pre_cov@jac.T;cov=(cov+cov.T)/2
        diagonal=cov.diag().clone()
    return dict(raw_mean=raw.detach(),mean=mean.detach(),variance_diag=diagonal,
                covariance=cov,pre_covariance=pre_cov,
                weighted_pre_jacobian=factor if keep_factor else None)
