import unittest
import torch
from standalone import math_qwen as q,math_paths as paths,math_gte as gte
from standalone.curvature import projected_noise,score_coordinate_jacobian

class StandaloneMathTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(21)
    def test_all_native_diagonals_match_direct_output_jacobians(self):
        d,k=4,6;w=torch.randn(d,k,dtype=torch.float64);v=torch.rand_like(w)*.01;h=torch.randn(k,dtype=w.dtype);r=torch.randn(d,dtype=w.dtype)
        cases=[(lambda x:q.forward(x,h,r,torch.ones(d),1e-6)[1],lambda:q.output_diagonal(w,h,r,torch.ones(d),1e-6,v))]
        c=dict(post_scale=torch.rand(d,dtype=w.dtype)+.5,final_scale=torch.rand(d,dtype=w.dtype)+.5,post_eps=1e-6,final_eps=1e-6)
        cases.append((lambda x:paths.harrier_forward(x,h,r,c)[1],lambda:paths.harrier_covariance(w,h,r,c,v,False)[1]))
        gc=dict(bias=torch.randn(d,dtype=w.dtype),gamma=torch.rand(d,dtype=w.dtype)+.5,beta=torch.randn(d,dtype=w.dtype),eps=1e-5)
        f=dict(H=torch.randn(3,k,dtype=w.dtype),R=torch.randn(3,d,dtype=w.dtype),mask=torch.ones(3,dtype=w.dtype))
        cases.append((lambda x:gte.split_forward(x,f,gc)[1],lambda:paths.gte_diagonal(w,f,gc,v)))
        for fn,fast in cases:
            jac=torch.autograd.functional.jacobian(fn,w);expected=(jac.square()*v).sum((1,2));self.assertTrue(torch.allclose(fast(),expected,atol=1e-11,rtol=1e-9))

    def test_streamed_gte_raw_vjp_retains_shared_weight_across_tokens(self):
        d,k=4,6;w=torch.randn(d,k,dtype=torch.float64);c=dict(bias=torch.randn(d,dtype=w.dtype),gamma=torch.rand(d,dtype=w.dtype)+.5,beta=torch.randn(d,dtype=w.dtype),eps=1e-5)
        features=[dict(H=torch.randn(n,k,dtype=w.dtype),R=torch.randn(n,d,dtype=w.dtype),mask=torch.ones(n,dtype=w.dtype)) for n in [2,4]];a=torch.randn(2,d,dtype=w.dtype)
        test=w.clone().requires_grad_();loss=sum((gte.split_forward(test,f,c)[0]*coef).sum() for f,coef in zip(features,a));expected=torch.autograd.grad(loss,test)[0]
        self.assertTrue(torch.allclose(gte.raw_vjp(w,features,c,a),expected,atol=1e-11,rtol=1e-9))

    def test_categorical_projection_and_shared_endpoint_score_jacobian(self):
        p=torch.tensor([[.3,.7],[.4,.6]],dtype=torch.float64);a=torch.randn_like(p);r=projected_noise(p,a)
        self.assertTrue(torch.allclose(r.sum(1),torch.zeros(2,dtype=p.dtype),atol=1e-15))
        w=torch.randn(3,4,dtype=p.dtype);qh=torch.randn(2,4,dtype=p.dtype);ph=torch.randn(2,4,dtype=p.dtype)
        def scores(t):return (qh@t.T)@(ph@t.T).T
        expected=torch.autograd.functional.jacobian(scores,w).reshape(2,2,-1)
        identity=torch.eye(w.numel(),dtype=w.dtype).reshape(-1,*w.shape)
        jq=torch.einsum('nk,adk->nda',qh,identity);jp=torch.einsum('nk,adk->nda',ph,identity)
        self.assertTrue(torch.allclose(score_coordinate_jacobian(qh@w.T,ph@w.T,jq,jp),expected,atol=1e-12))

if __name__=='__main__':unittest.main()
