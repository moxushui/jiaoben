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


def bgd(w0,w1,rph=0.001,alph=10**-6,beta=10**-6):
    '''

    :param w0:
    :param w1:
    :param rph: 学习率
    :param alph: 梯度收敛误差值
    :param beta: 损失收敛误差值
    :return: 权重和损失值变化结果
    '''
    '''优化代码'''
    '''计算当前损失值并保存'''
    loss_histroy = []
    for _ in range(100000):
        loss_value = less(w0, w1)
        loss_histroy.append(loss_value)
        d_w0, d_w1 = grad(w0, w1)
        if np.linalg.norm(grad(w0, w1)) <= alph:
            print('梯度收敛')
            break
        w0 -= rph*d_w0
        w1 -= rph*d_w1
        if abs(less(w0,w1) - loss_value) <= beta:
            print('损失值收敛')
            break
    return w0,w1,loss_histroy

    '''原代码'''
    # loss_lst2 =[]
    # for _ in range(1000):
    #     d_w0, d_w1 = grad(w0, w1)
    #     loss_value = less(w0,w1)
    #     if np.linalg.norm(grad(w0,w1)) <= alph:
    #         return w0,w1,loss_lst2
    #     w0 -= rph*d_w0
    #     w1 -= rph*d_w1
    #     loss_value1 = less(w0,w1)
    #     loss_lst2.append(loss_value1)
    #     if abs(loss_value1-loss_value)<= beta:
    #         return w0,w1,loss_lst2

if __name__=='__main__':
    print(bgd(0,0,rph=0.0001,alph=10**-8,beta=10**-6))

