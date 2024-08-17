import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

data = pd.read_csv('data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

reg = LinearRegression()
history = reg.fit(X, y, tol=.5e-2, learning_rate=1e-5)

print(f'R = {reg.score(X, y)}')
print(f'c = {reg.intercept_}')
for index, coef in enumerate(reg.coef_):
    print(f'm_{index + 1} = {coef}')

# Plotando o gráfico MSE vs Iteração
plt.plot(history)
plt.xlabel('Número da Iteração')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.title('Convergência do Erro (MSE) Durante o Treinamento')
plt.show()

y_pred = reg.predict(X)

for index, coef in enumerate(reg.coef_):
    plt.scatter(X[:, index], y, color='blue')
    plt.plot(X[:, index], y_pred, color='red')
    plt.xlabel(f'X_{index+1}')
    plt.ylabel('y')
    plt.title(f'Relação entre x_{index+1} e y')
    plt.show()