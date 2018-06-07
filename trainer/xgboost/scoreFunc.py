import numpy as np 

def get_score_wieght(label):
    
    weight = []
    
    for i in range(len(label)):
        
        if i < len(label)*0.2:
            weight.append(4)
        elif i >= len(label)*0.2 and i < len(label)*0.4:
            weight.append(2.5)
        elif i >= len(label)*0.4 and i < len(label)*0.6:
            weight.append(2)
        elif i >= len(label)*0.6 and i < len(label)*0.8:
            weight.append(1.5)
        else:
            weight.append(1)
            
    return np.flip(weight, axis=0)
    

def time_discriminator_score(estimator, x, y):
    
    y_pred = estimator.predict(x)    
    weight = get_score_wieght(y)
    
    return np.mean(np.equal(y_pred,y).astype(np.float32)*weight)