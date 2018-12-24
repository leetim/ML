# Логическая регрессия

### Бинарнай классификация

- Сигнум $sing{\,x}$
- Сигмоийда $\sigma(x) = \frac{1}{1 + e^{-x}}$

$P(y | x) = \sigma(y\,W^Tx)$
### Функция правдоподобия

- $L(D | g) = \prod\limits_{n = 1}^N P(y| x_n)$
- $L(D | g) = \ln\prod\limits_{n = 1}^N P(y| x_n) = \sum\limits_{n = 1}^N \ln P(y| x_n)$