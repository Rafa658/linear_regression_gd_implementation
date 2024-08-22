### Multiple Linear Regression using Gradient Descent

É definida uma variável dependente $y$ que se relacione com múltiplas variáveis independentes $x_1$, $x_2$, ..., $x_n$ de acordo com:

$$ y = m_1 x_1 + m_2 x_2 + ... + m_n x_n + c $$

Onde $m_1$, $m_2$, ..., $m_n$ são os pesos de cada variável e $c$ um viés. O modelo será treinado com base em $m$ amostras, de forma que:

$$ y_1 = m_1 x^1_1 + m_2 x^1_2 + ... + m_n x^1_n + c $$

$$ y_2 = m_1 x^2_1 + m_2 x^2_2 + ... + m_n x^2_n + c $$

$$ y_m = m_1 x^m_1 + m_2 x^m_2 + ... + m_n x^m_n + c $$

Ainda, pode ser escrito em termos matriciais: $\textbf{y} = \textbf{X} \textbf{w}$

Onde:
- $\textbf{y}$ é um vetor $m \times 1$ que representa as amostras;
- $\textbf{X}$ é um vetor $m \times (n+1)$, na qual cada linha é uma amostra e foi inserida uma coluna extra para o viés;
- $\textbf{w}$ é um vetor $(n+1) \times 1$, que representa os pesos de cada variável.

Define-se uma função custo baseada em _mean square error_ (MSE):

$$ E(\textbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^i - \hat{y}^i)^2$$

De forma a minimizar a função, utilizou-se o gradiente descendente. Esse algoritmo baseia-se em atualizar os coeficientes iterativamente, movendo-se contra o gradiente da função de custo, que representa tendência de aumento (isto é, movendo-se contra o aumento de forma a diminuir o custo):

$$ \textbf{w} := \textbf{w} - \alpha \cdot \nabla_{\textbf{w}} E(\textbf{w}) $$

Onde $\alpha$ é a taxa de aprendizagem. O gradiente com relação aos coeficientes $\textbf{w}$ é:

$$ \nabla_{\textbf{w}} E(\textbf{w}) = \frac{2}{m} \textbf{X}^T (\textbf{X} \textbf{w} - \textbf{y}) $$

Finalmente, o resultado é fornecido após um determinado número de iterações, ou até a diminuição do erro atingir uma tolerância pré-determinada.
