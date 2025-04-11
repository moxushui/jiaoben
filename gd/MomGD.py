from math import gamma

import numpy as np
'''构造损失函数'''
def less(w0,w1):
    sample_fun1 = 63-(60*w1+w0)
    sample_fun2 = 65.2-(62*w1+w0)
    return 1/4*((sample_fun1**2+sample_fun2**2))


'''计算全量梯度'''
def grad(w0,w1):
    b_grad_w0 = -0.5*(128.2-122*w1-2*w0)
    b_grad_w1 = -0.5*(7822.4-7444*w1-122*w0)
    return b_grad_w0, b_grad_w1



def momgd(w0,w1,rph,gamma,alpha,beta):
    loss_history = []
    v_w0 ,v_w1 = 0,0
    for _ in range(1000):
        loss_value = less(w0,w1)
        loss_history.append(loss_value)
        d_w0,d_w1 = grad(w0,w1)
        if np.linalg.norm(grad(w0,w1)) <= alpha:
            print('梯度收敛')
            break
        v_w0 = gamma*v_w0+rph*d_w0
        v_w1 = gamma*v_w1+rph*d_w1
        w0 -= v_w0
        w1 -= v_w1
        if abs(less(w0,w1)-loss_value)<=beta:
            print('损失值收敛')
            break
    return w0,w1,loss_history



if __name__=='__main__':
    print(momgd(0,0,0.0001,0.01,alpha=10**-8,beta=10**-8))


