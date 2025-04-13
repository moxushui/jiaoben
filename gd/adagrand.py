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

def adagrand(w0,w1,rpo,alpha,beta,lota=10**-8):
    loss_history = []
    g_w0 ,g_w1 = 0,0
    count = 0
    for _ in range(1000):
        count +=1
        loss_value = less(w0,w1)
        loss_history.append([count,w0,w1,loss_value])
        grad_w0,grad_w1 = grad(w0,w1)
        if np.linalg.norm((grad_w0,grad_w1)) <= alpha:
            print('梯度收敛')
            break
        g_w0 = (g_w0+grad_w0**2)
        g_w1 = (g_w1+grad_w1**2)
        w0 -= rpo/((g_w0+lota)**0.5)*grad_w0
        w1 -= rpo/((g_w1+lota)**0.5)*grad_w1
        if abs(less(w0,w1)-loss_value)<= beta:
            print('损失值收敛')
            break
    return loss_history


if __name__ == '__main__':
    print(adagrand(0,0,0.1,10**-8,10**-6))


