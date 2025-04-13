import numpy as np
'''构造损失函数'''
def less(w0,w1):
    sample_fun1 = 63-(60*w1+w0)
    sample_fun2 = 65.2-(62*w1+w0)
    return 1/4*((sample_fun1**2+sample_fun2**2))


'''计算梯度'''
def grad(w0,w1):
    b_grad_w0 = -0.5*(128.2-122*w1-2*w0)
    b_grad_w1 = -0.5*(7822.4-7444*w1-122*w0)
    return b_grad_w0, b_grad_w1



def adadelta(w0,w1,gama,alpha,beta,lota=10**-6):
    loss_history = []
    e_w0 ,e_w1 = 0,0
    ev_w0 ,ev_w1 = 0,0
    count = 0
    for _ in range(1000):
        count +=1
        loss_value = less(w0,w1)
        loss_history.append([count,w0,w1,loss_value])
        grad_w0,grad_w1 = grad(w0,w1)
        if np.linalg.norm(grad(w0,w1)) <= alpha:
            print('梯度收敛')
            break
        e_w0 = gama*e_w0 +(1-gama)*grad_w0**2
        e_w1 = gama*e_w1 +(1-gama)*grad_w1**2
        v_w0 = (ev_w0+lota)**0.5/(e_w0+lota)**0.5  * grad_w0
        v_w1 = (ev_w1 + lota) ** 0.5 / (e_w1 + lota) ** 0.5 * grad_w1
        w0 -= v_w0
        w1 -= v_w1
        if abs(less(w0,w1)-loss_value)<= beta:
            print('损失值收敛')
            break
        ev_w0 = gama*ev_w0+(1-gama)*v_w0**2
        ev_w1 = gama*ev_w1+(1-gama)*v_w1**2
    return loss_history

if __name__=='__main__':
    print(adadelta(0,0,0.9,10**-6,10**-8))