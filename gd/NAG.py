from itertools import count

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

def nag(w0,w1,gama,rpo,alpha,beta):
    v_w0 = 0
    v_w1 = 0
    count = 0
    guess_wo,guess_w1 = w0 ,w1
    loss_history = []
    for _ in range(1000):
        count +=1
        loss_value = less(guess_wo,guess_w1)
        loss_history.append([count,guess_wo,guess_w1,loss_value])
        grad_w0, grad_w1 = grad(guess_wo, guess_w1)
        if np.linalg.norm((grad_w0,grad_w1)) <= alpha:
            print('梯度收敛')
            break
        v_w0 =gama*v_w0 + rpo*grad_w0
        v_w1 =gama*v_w1 + rpo*grad_w1

        w0 = w0 - v_w0
        w1 = w1 - v_w1
        if abs(less(w0,w1)-loss_value)< beta:
            print('损失值收敛')
            break
        guess_wo = w0 -gama*v_w0
        guess_w1 = w1 - gama*v_w1
    return loss_history


if __name__=='__main__':
    print(nag(0,0,0.01,0.00001,10**-6,10**-11))

'''moment 会比nag更早接近极值点.(但由于惯性，在极值点附件停留的时间比nag短，跑过头返回极值点也比nag更慢。)'''