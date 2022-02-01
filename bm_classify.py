import numpy as np

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    N, D = X.shape
    assert len(np.unique(y)) == 2
    y = np.where(y == 0, -1, y)

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":

        def get_deriv(x_n,y_n):
            val = y_n*(np.dot(w, x_n) + b)
            v = np.multiply(x_n, y_n) 
            if val > 0:         
                return np.zeros(D),0
            if val <= 0:  
                return v, y_n

        v_get_deriv = np.vectorize(get_deriv)
        
        min_error = float('inf')
        for i in range(max_iterations): 
            d = np.dot(X,w)
            a = np.add(d,b)
            b_error = np.sum(e)
            l_sum=np.add.reduce(l)
            delta = np.multiply(l_sum,step_size)
            dd = np.divide(delta, N)
            w=np.add(w, l_sum)      
            b = b + step_size * float(b_error/N)

    elif loss == "logistic":

        def get_delta_e(x_n, y_n):
            if y_n == 0:
                y_n = -1
            z = y_n*(np.dot(w,x_n) + b)
            v = -1*np.exp(-z)*sigmoid(z)
            return v

        def get_delta_b(x_n, y_n):
            if y_n == 0:
                y_n = -1
            del_e = get_delta_e(x_n, y_n)
            bv = np.multiply(del_e,y_n)
            return bv

        def get_delta_w(x_n, y_n):
            if y_n == 0:
                y_n = -1
            del_e = get_delta_e(x_n, y_n)
            ynxn = np.multiply(y_n, x_n)
            val = np.multiply(del_e, ynxn)
            return val

        def calc_loss(x_n, y_n):
            if y_n == 0:
                y_n = -1
            t = np.dot(w,x_n) + b
            z = y_n*t
            l = np.log(1+np.exp(-1* z))
            return l

        min_error=float('inf')

        for i in range(max_iterations):
            l = np.zeros(N)
            d = np.zeros([N,D])
            e = np.zeros(N)
            for n in range(N):
                l[n] = calc_loss(X[n], y[n])
                d[n] = get_delta_w(X[n], y[n])
                e[n] = get_delta_b(X[n], y[n])

            error = np.sum(l)
            b_error = np.sum(e)
            if error < min_error:
                min_error=error
                d_sum=np.add.reduce(d)  
                m = np.multiply(d_sum, step_size)
                div = np.divide(m, N)
                w = np.subtract(w,div)
                b = b - step_size * float(b_error/N)
    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    e = np.exp(np.multiply(-1,z))
    a = np.add(1,e)
    return np.power(a,-1)


def binary_predict(X, w, b):
    N, D = X.shape
    preds=np.zeros(N)
    for x_n in range(X.shape[0]):
        y = np.dot(w,X[x_n]) + b
        if y <= 0:
            preds[x_n] = 0
        else:
            preds[x_n] = 1

    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    def calc_dot(k, w_yn, feature):
        diff = np.subtract(k, w_yn)
        dot = np.dot(diff, feature)
        return dot

    def calc_z(w_k_dot, b_k, b_yn):
        sum = np.add(w_k_dot, b_k)
        z = np.subtract(sum, b_yn)
        return z

    def calc_norm(w):
        max = np.max(w)
        norm = np.subtract(w, max)
        return norm

    v_calc_z = np.vectorize(calc_z)
    v_calc_cot = np.vectorize(calc_dot)

    def gd_multi(row):
        nonlocal w
        nonlocal b
        nonlocal w_sum
        nonlocal b_sum
        feature = row[0:2]
        actual_label = int(row[2])
        diff = np.subtract(w, w[actual_label])
        dot = np.dot(diff,feature)
        sum = np.add(dot, b)
        w_z = np.subtract(sum, b[actual_label])
        max = np.max(w_z)
        norm = np.subtract(w_z, max)
        w_sf = np.exp(norm)
        denom = np.sum(w_sf) 
        denom_k = np.subtract(denom, w_sf[actual_label])
        numerator = 0.00
        w_sf[actual_label] = np.multiply(-1.0, denom_k)
        delta = np.divide(w_sf, np.add(1.0, denom_k))
        w_dm = np.multiply(delta[:, None], feature)
        w_sum = np.add(w_sum, w_dm)
        b_sum += delta

    v_gd = np.vectorize(gd_multi, otypes=[object])

    np.random.seed(42) 
    if gd_type == "sgd":
        for it in range(max_iterations):
            n = np.random.choice(N)
            feature = X[n]
            actual_label = y[n]
            diff = np.subtract(w, w[actual_label])

            dot = np.dot(diff,feature)
            sum = np.add(dot, b)
            w_z = np.subtract(sum, b[actual_label])
            w_sf = np.exp(w_z)
            denom = np.sum(w_sf) 
            denom_k = np.subtract(denom, w_sf[actual_label])
            numerator = 0.00

            w_sf[actual_label] = np.multiply(-1.0, denom_k)
            delta = np.divide(w_sf, np.add(1.0, denom_k))
            w_dm = np.multiply(delta[:, None], feature)
            w_delta_f = np.multiply(w_dm, step_size)
            b_delta_f = np.multiply(delta, step_size)
            w = np.subtract(w, w_delta_f)
            b = np.subtract(b, b_delta_f)

    elif gd_type == "gd":

        total = np.concatenate((X,y[:,np.newaxis]), axis=1)      
        for it in range(max_iterations):    
            w_sum = np.zeros((C,D))
            b_sum = 0
            np.apply_along_axis(gd_multi, 1, total)
            w_delta_f = np.multiply(w_sum, step_size)
            b_delta_f = np.multiply(b_sum, step_size)
            w_f = np.divide(w_delta_f,N)
            b_f = np.divide(b_delta_f,N)
            w = np.subtract(w, w_f)
            b = np.subtract(b, b_f)
            
                
    else:
        raise "Undefined algorithm."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b

def multiclass_predict(X, w, b):
    N, D = X.shape

    def get_scores(wc, bc, x_n):
        return np.dot(wc, x_n) + bc

    v_get_scores=np.vectorize(get_scores)

    def get_max_scores(x_n):
        scores = v_get_scores(w, b, x_n)
        return np.argmax(scores)

    v_get_max_scores = np.vectorize(get_max_scores)
    preds = np.zeros(N)
    for n in range(X.shape[0]):
        score = float('-inf')
        pred = 0
        for c in range(w.shape[0]):
            temp = np.dot(w[c], X[n]) + b[c]
            if temp > score:
                score = temp
                pred = c
        preds[n] = pred
    assert preds.shape == (N,)
    return preds