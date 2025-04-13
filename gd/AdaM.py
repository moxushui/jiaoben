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


def adam(w0,w1,rpo,gama,lamb,alpha,beta,lota=10**-6):
    '''

    :param w0:
    :param w1:
    :param rpo:
    :param gama: 对历史梯度的权重
    :param lamb: 对历史平方梯度的权重
    :param alpha:
    :param beta:
    :return:
    '''
    loss_history = []
    v_w0,v_w1 = 0,0
    e_w0,e_w1 = 0,0
    for i in range(1,1001):
        loss_value = less(w0,w1)
        loss_history.append([i,w0,w1,loss_value])
        grad_w0,grad_w1 = grad(w0,w1)
        if np.linalg.norm(grad(w0,w1)) <= alpha:
            print('梯度收敛')
            break
        v_w0 = gama*v_w0 + (1-gama)*grad_w0
        v_w1 = gama*v_w1 + (1-gama)*grad_w1
        hat_v_w0 = v_w0/(1-gama**i)
        hat_v_w1 = v_w1/(1-gama**i)

        e_w0 = lamb*e_w0 +(1-lamb)*grad_w0**2
        e_w1 = lamb*e_w1 +(1-lamb)*grad_w1**2
        hat_e_w0 = e_w0/(1-lamb**i)
        hat_e_w1 = e_w1/(1-lamb**i)

        w0 -= rpo/(hat_e_w0+lota)**0.5  * hat_v_w0
        w1 -= rpo/(hat_e_w1+lota)**0.5  * hat_v_w1
        if abs(less(w0,w1)-loss_value) <= beta:
            print('损失值收敛')
            break
    return loss_history


if __name__ =='__main__':
    print(adam(0,0,0.01,0.9,0.999,10**-6,10**-8))
