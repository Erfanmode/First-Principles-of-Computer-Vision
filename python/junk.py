from sympy import Matrix, symbols, factor, simplify, expand

a, b, c = symbols('a b c')
x, y, z = symbols('x y z')
# Create a symbolic matrix
M = Matrix([[1, 0, 0],
            [0, 1, 0],
            [a, b, c]])
Minv = M.inv()

print(M)
print(Minv)
print(Minv.subs({a: x, }))
eq = (x**3 - a**2*x)
print(eq,'\n', simplify(eq), '\n', factor(eq))