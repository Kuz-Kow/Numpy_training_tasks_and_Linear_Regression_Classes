import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
        
    
    def fit(self, X, y):
        
            X = np.insert(X,0, 1, axis= 1)
            
            XT_X_inv = np.linalg.inv(X.T @ X)
            
            weights = np.linalg.multi_dot([XT_X_inv, X.T, y])
            
            self.b , self.w = weights[0], weights[1:]

    
    def predict(self, X_test):
        return X_test @ self.w + self.b
    
    
    
class LinearRegressionGradient:
    def __init__(self, training_rate = 0.01, tolerance = 1e-8) -> None:
         self.training_rate = training_rate
         self.tolerance = tolerance

         
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.bias, self.weights = 0, np.zeros(n_features)
        previous_db, previous_dw = 0, np.zeros(n_features)
        
        while True:
            y_pred = X @ self.weights + self.bias
            db = 1 / n_samples * np.sum(y_pred- y)
            dw = 1 / n_samples * X.T @ (y_pred - y)
            
            self.bias -= self.training_rate * db
            self.weights -= self.training_rate * dw
            
            abs_db_reduction = np.abs(db - previous_db)
            abs_dw_reduction = np.abs(dw - previous_dw)
            
            if abs_db_reduction < self.tolerance:
                if abs_dw_reduction.all() < self.tolerance:
                    break
        
                    
            previous_dw = dw
            previous_db = db
        
    def predict(self, X_test):
        return X_test @ self.weights + self.bias
    

def mse(y_pred, y_true):
    return np.mean((y_true - y_pred) ** 2)

    
    
def main():
    
    rng = np.random.default_rng(42)

    X = rng.uniform(0, 10, size=(100, 1))

    noise = rng.normal(0, 2, size=100)

    y = 3 * X[:, 0] + 5 + noise
    
    model_normal = LinearRegression()
    model_gradient = LinearRegressionGradient()

    model_normal.fit(X, y)
    model_gradient.fit(X, y)

    pred_normal = model_normal.predict(X)
    pred_gradient = model_gradient.predict(X)

    print("Normal Equation:", mse(y, pred_normal))
    print("Gradient Descent:", mse(y, pred_gradient))
    
    X_test = np.arange(10).reshape(-1,1)
    regression = model_gradient.predict(X_test)
    
    fig, ax = plt.subplots()
    
    ax.scatter(X[:, 0], y, label = "Training data")
    ax.plot(X_test[:, 0], regression.ravel(), label = "Prediction")
    ax.legend()

    plt.show()
    
if __name__ == "__main__":
    main()   