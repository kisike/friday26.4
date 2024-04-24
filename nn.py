
# Filter changed files
#  18 changes: 18 additions & 0 deletions18  
# B03898_01_codes/B03898_01_01.py
# @@ -0,0 +1,18 @@
# """"
# README
# ======
# This file contains Python codes.
# ======
# """

# # class Greeting(object):
# def __init__(self, my_greeting):
#          self.my_greeting = my_greeting

# def say_hello(self, name):
#          print(self.my_greeting, name)

# greeting =("Hello")
# greeting.say_hello("World")
# greeting.say_hello("Dog")
# greeting.say_hello("Cat")
#  1 change: 1 addition & 0 deletions1  
# B03898_01_codes/B03898_01_02.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""from functools import partialdef greeting(my_greeting, name):    print "%s %s" % (my_greeting, name)say_hello_to = partial(greeting, "Hello")say_hello_to("World")say_hello_to("Dog")say_hello_to("Cat")
#  1 change: 1 addition & 0 deletions1  
# B03898_01_codes/B03898_01_03.txt
# @@ -0,0 +1 @@
# README======This file contains Markdown code.To be run in IPython Notebook.======Text Examples===This is an example of an *italic* text.This is an example of a **bold*** text.This is an example of a list item:- Item #1- Item #2- Item #3---#heading 1##heading 2###heading 3####heading 4#####heading 5######heading 6
#  9 changes: 9 additions & 0 deletions9  
# B03898_01_codes/B03898_01_04.txt
# @@ -0,0 +1,9 @@
# """"
# README
# ======
# This file contains Python codes.
# ======
# """

# answer = 3 + 5
# print answer
#  15 changes: 15 additions & 0 deletions15  
# B03898_01_codes/B03898_01_05.txt
# @@ -0,0 +1,15 @@
# README
# ======
# This file contains Python codes.
# To be run in IPython Notebook.
# ======

# import matplotlib.pyplot as plt
# import numpy as np
# import math

# x = np.linspace(0, 2*math.pi)
# plt.plot(x, np.sin(x), label=r'$\sin(x)$')
# plt.plot(x, np.cos(x), 'ro', label=r'$\cos(x)$')
# plt.title(r'Two plots in a graph')
# plt.legend()
#  7 changes: 7 additions & 0 deletions7  
# B03898_01_codes/B03898_01_06.txt
# @@ -0,0 +1,7 @@
# README
# ======
# This file contains Markdown codes.
# To be run in IPython Notebook.
# ======

# $$N(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x} e^{-\frac{z^2}{2}}\, dz$$
#  9 changes: 9 additions & 0 deletions9  
# B03898_01_codes/B03898_01_07.txt
# @@ -0,0 +1,9 @@
# README
# ======
# This file contains Python codes.
# To be run in IPython Notebook.
# ======

# from IPython.display import Math
# Math(r'N(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x} e^{-\frac{z^2}{2}}\, dz')

#  9 changes: 9 additions & 0 deletions9  
# B03898_01_codes/B03898_01_08.txt
# @@ -0,0 +1,9 @@
# README
# ======
# This file contains Python codes.
# To be run in IPython Notebook.
# ======

# from IPython.display import Image
# Image(url='http://python.org/images/python-logo.gif')

#  12 changes: 12 additions & 0 deletions12  
# B03898_01_codes/B03898_01_09.txt
# @@ -0,0 +1,12 @@
# README
# ======
# This file contains Python codes.
# To be run in IPython Notebook.
# ======

# from IPython.lib.display import YouTubeVideo

# # An introduction to Python by Google.
# YouTubeVideo('tKTZoB2Vjuk')


#  1 change: 1 addition & 0 deletions1  
# B03898_01_codes/B03898_01_10.txt
# @@ -0,0 +1 @@
# README======This file contains Python codes.To be run in IPython Notebook.======from IPython.display import HTMLtable = """<table><tr><th>Header 1</th><th>Header 2</th></tr><tr><td>row 1, cell 1</td><td>row 1, cell 2</td></tr><tr><td>row 2, cell 1</td><td>row 2, cell 2</td></tr></table>"""HTML(table)
#  1 change: 1 addition & 0 deletions1  
# B03898_01_codes/B03898_01_11.txt
# @@ -0,0 +1 @@
# README======This file contains Python codes.To be run in IPython Notebook.======import pandas.io.data as webimport datetimestart = datetime.datetime(2014, 1, 1)end = datetime.datetime(2014, 12, 31)df = web.DataReader("AAPL", 'yahoo', start, end)df.head()
#  25 changes: 25 additions & 0 deletions25  
# B03898_02_Codes/B03898_02_01.py
# @@ -0,0 +1,25 @@
# """"
# README
# ======
# This is a Python code.
# ======
# """

# """ Linear regression with SciPy """
# from scipy import stats

# stock_returns = [0.065, 0.0265, -0.0593, -0.001, 0.0346]
# mkt_returns = [0.055, -0.09, -0.041, 0.045, 0.022]

# beta, alpha, r_value, p_value, std_err = \
#     stats.linregress(stock_returns, mkt_returns)
# print beta, alpha

# """ Calculating the SML """
# rf = 0.05
# mrisk_prem = 0.085
# risk_prem = mrisk_prem * beta
# print "Risk premium:", risk_prem

# expected_stock_return = rf + risk_prem
# print "Expected stock return:", expected_stock_return
#  25 changes: 25 additions & 0 deletions25  
# B03898_02_Codes/B03898_02_02.py
# @@ -0,0 +1,25 @@
# """
# README
# ======
# This is a Python code.
# ======
# """

# """ Least squares regression with statsmodels """
# import numpy as np
# import statsmodels.api as sm

# # Generate some sample data
# num_periods = 9
# all_values = np.array([np.random.random(8)
#                        for i in range(num_periods)])

# # Filter the data
# y_values = all_values[:, 0]  # First column values as Y
# x_values = all_values[:, 1:]  # All other values as X

# x_values = sm.add_constant(x_values)  # Include the intercept
# results = sm.OLS(y_values, x_values).fit()  # Regress and fit the model

# print results.summary()
# print results.params
#  1 change: 1 addition & 0 deletions1  
# B03898_02_Codes/B03898_02_03.py
# @@ -0,0 +1 @@
# """README======This is a Python code.======"""""" A simple linear optimization problem with 2 variables """import pulpx = pulp.LpVariable("x", lowBound=0)y = pulp.LpVariable("y", lowBound=0)problem = pulp.LpProblem("A simple maximization objective",                         pulp.LpMaximize)problem += 3*x + 2*y, "The objective function"problem += 2*x + y <= 100, "1st constraint"problem += x + y <= 80, "2nd constraint"problem += x <= 40, "3rd constraint"problem.solve()print "Maximization Results:"for variable in problem.variables():    print variable.name, "=", variable.varValue
#  44 changes: 44 additions & 0 deletions44  
# B03898_02_Codes/B03898_02_04.py
# @@ -0,0 +1,44 @@
# """"
# README
# ======
# This file contains Python codes.
# ======
# """

# """ An example of implementing an IP model with binary conditions  """
# import pulp

# dealers = ["X", "Y", "Z"]
# variable_costs = {"X": 500,
#                   "Y": 350,
#                   "Z": 450}
# fixed_costs = {"X": 4000,
#                "Y": 2000,
#                "Z": 6000}

# # Define PuLP variables to solve
# quantities = pulp.LpVariable.dicts("quantity",
#                                    dealers,
#                                    lowBound=0,
#                                    cat=pulp.LpInteger)
# is_orders = pulp.LpVariable.dicts("orders", dealers,
#                                   cat=pulp.LpBinary)

# """
# This is an example of implementing an IP model with binary
# variables the wrong way.
# """
# # Initialize the model with constraints
# model = pulp.LpProblem("A cost minimization problem", pulp.LpMinimize)
# model += sum([(variable_costs[i] * quantities[i] +
#                fixed_costs[i]) * is_orders[i] for i in dealers]), \
#          "Minimize portfolio cost"
# model += sum([quantities[i] for i in dealers]) == 150, \
#          "Total contracts required"
# model += 30 <= quantities["X"] <= 100, \
#          "Boundary of total volume of X"
# model += 30 <= quantities["Y"] <= 90, \
#          "Boundary of total volume of Y"
# model += 30 <= quantities["Z"] <= 70, \
#          "Boundary of total volume of Z"
# model.solve() # Will encounter an error
#  50 changes: 50 additions & 0 deletions50  
# B03898_02_Codes/B03898_02_05.py
# @@ -0,0 +1,50 @@
# """"
# README
# ======
# This file contains Python codes.
# ======
# """

# import pulp

# dealers = ["X", "Y", "Z"]
# variable_costs = {"X": 500,
#                   "Y": 350,
#                   "Z": 450}
# fixed_costs = {"X": 4000,
#                "Y": 2000,
#                "Z": 6000}

# # Define PuLP variables to solve
# quantities = pulp.LpVariable.dicts("quantity",
#                                    dealers,
#                                    lowBound=0,
#                                    cat=pulp.LpInteger)
# is_orders = pulp.LpVariable.dicts("orders", dealers,
#                                   cat=pulp.LpBinary)

# """
# This is an example of implementing an IP model with binary
# variables the correct way.
# """
# # Initialize the model with constraints
# model = pulp.LpProblem("A cost minimization problem",
#                        pulp.LpMinimize)
# model += sum([variable_costs[i]*quantities[i] +
#               fixed_costs[i]*is_orders[i] for i in dealers]), \
#          "Minimize portfolio cost"
# model += sum([quantities[i] for i in dealers]) == 150, \
#          "Total contracts required"
# model += is_orders["X"]*30 <= quantities["X"] <= \
#          is_orders["X"]*100, "Boundary of total volume of X"
# model += is_orders["Y"]*30 <= quantities["Y"] <= \
#          is_orders["Y"]*90, "Boundary of total volume of Y"
# model += is_orders["Z"]*30 <= quantities["Z"] <= \
#          is_orders["Z"]*70, "Boundary of total volume of Z"
# model.solve()

# print "Minimization Results:"
# for variable in model.variables():
#     print variable, "=", variable.varValue

# print "Total cost: %s" % pulp.value(model.objective)
#  1 change: 1 addition & 0 deletions1  
# B03898_02_Codes/B03898_02_06.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" Linear algebra with NumPy matrices """import numpy as npA = np.array([[2, 1, 1],              [1, 3, 2],              [1, 0, 0]])B = np.array([4, 5, 6])print np.linalg.solve(A, B) 
#  1 change: 1 addition & 0 deletions1  
# B03898_02_Codes/B03898_02_07.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" LU decomposition with SciPy """import scipy.linalg as linalgimport numpy as npA = np.array([[2., 1., 1.],              [1., 3., 2.],              [1., 0., 0.]])B = np.array([4., 5., 6.])LU = linalg.lu_factor(A)x = linalg.lu_solve(LU, B)print xP, L, U = linalg.lu(A)print Pprint Lprint U
#  1 change: 1 addition & 0 deletions1  
# B03898_02_Codes/B03898_02_08.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" Cholesky decomposition with NumPy """import numpy as npA = np.array([[10., -1., 2., 0.],              [-1., 11., -1., 3.],              [2., -1., 10., -1.],              [0.0, 3., -1., 8.]])B = np.array([6., 25., -11., 15.])L = np.linalg.cholesky(A)print Lprint np.dot(L, L.T.conj())  # A=L.L*y = np.linalg.solve(L, B)  # L.L*.x=B. When L*.x=y, then L.y=B.print yx = np.linalg.solve(L.T.conj(), y)  # x=L*'.yprint xprint np.mat(A) * np.mat(x).T  # B=Ax
#  1 change: 1 addition & 0 deletions1  
# B03898_02_Codes/B03898_02_09.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" QR decomposition with scipy """import scipyimport numpy as npA = np.array([    [2., 1., 1.],    [1., 3., 2.],    [1., 0., 0]])B = np.array([4., 5., 6.])Q, R = scipy.linalg.qr(A)  # QR decompositiony = np.dot(Q.T, B)  # Let y=Q`.Bx = scipy.linalg.solve(R, y)  # Solve Rx=yprint x
#  1 change: 1 addition & 0 deletions1  
# B03898_02_Codes/B03898_02_10.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" Solve Ax=B with the Jacobi method """import numpy as npdef jacobi(A, B, n, tol=1e-10):    # Initializes x with zeroes with same shape and type as B    x = np.zeros_like(B)    for it_count in range(n):        x_new = np.zeros_like(x)                for i in range(A.shape[0]):            s1 = np.dot(A[i, :i], x[:i])            s2 = np.dot(A[i, i + 1:], x[i + 1:])            x_new[i] = (B[i] - s1 - s2) / A[i, i]        if np.allclose(x, x_new, tol):            break        x = x_new    return xA = np.array([[10., -1., 2., 0.],              [-1., 11., -1., 3.],              [2., -1., 10., -1.],              [0.0, 3., -1., 8.]])B = np.array([6., 25., -11., 15.])n = 25x = jacobi(A,B,n)print "x =", x
#  1 change: 1 addition & 0 deletions1  
# B03898_02_Codes/B03898_02_11.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" Solve Ax=B with the Gauss-Seidel method """import numpy as npdef gauss(A, B, n, tol=1e-10):    L = np.tril(A)  # Returns the lower triangular matrix of A    U = A - L  # Decompose A = L + U    L_inv = np.linalg.inv(L)    x = np.zeros_like(B)        for i in range(n):                Ux = np.dot(U, x)        x_new = np.dot(L_inv, B - Ux)                if np.allclose(x, x_new, tol):            break                    x = x_new            return xA = np.array([[10., -1., 2., 0.],              [-1., 11., -1., 3.],              [2., -1., 10., -1.],              [0.0, 3., -1., 8.]])B = np.array([6., 25., -11., 15.])n = 100x = gauss(A, B, n)print "x =", x
#  1 change: 1 addition & 0 deletions1  
# B03898_03_codes/bisection.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" The bisection method """def bisection(f, a, b, tol=0.1, maxiter=10):    """    :param f: The function to solve    :param a: The x-axis value where f(a)<0    :param b: The x-axis value where f(b)>0    :param tol: The precision of the solution    :param maxiter: Maximum number of iterations    :return: The x-axis value of the root,                number of iterations used    """    c = (a+b)*0.5  # Declare c as the midpoint ab    n = 1  # Start with 1 iteration    while n <= maxiter:        c = (a+b)*0.5        if f(c) == 0 or abs(a-b)*0.5 < tol:            # Root is found or is very close            return c, n        n += 1        if f(c) < 0:            a = c        else:            b = c                    return c, nif __name__ == "__main__":    y = lambda x: x**3 + 2*x**2 - 5    root, iterations = bisection(y, -5, 5, 0.00001, 100)    print "Root is:", root    print "Iterations:", iterations
#  1 change: 1 addition & 0 deletions1  
# B03898_03_codes/general_nonlinear_solvers_no_solution.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======""""""General nonlinear solvers- with no solution"""import scipy.optimize as optimizey = lambda x: x**3 + 2.*x**2 - 5.dy = lambda x: 3.*x**2 + 4.*xprint optimize.fsolve(y, -5., fprime=dy)print optimize.root(y, -5.)
#  1 change: 1 addition & 0 deletions1  
# B03898_03_codes/general_nonlinear_solvers_with_solution.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======""""""General nonlinear solvers- with a solution"""import scipy.optimize as optimizey = lambda x: x**3 + 2.*x**2 - 5.dy = lambda x: 3.*x**2 + 4.*xprint optimize.fsolve(y, 5., fprime=dy)print optimize.root(y, 5.)
#  1 change: 1 addition & 0 deletions1  
# B03898_03_codes/incremental_search.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" An incremental search algorithm """import numpy as npdef incremental_search(f, a, b, dx):    """    :param f: The function to solve    :param a: The left boundary x-axis value    :param b: The right boundary x-axis value    :param dx: The incremental value in searching    :return: The x-axis value of the root,                number of iterations used    """    fa = f(a)        c = a + dx     fc = f(c)        n = 1    while np.sign(fa) == np.sign(fc):        if a >= b:            return a - dx, n                a = c        fa = fc        c = a + dx        fc = f(c)        n += 1    if fa == 0:        return a, n    elif fc == 0:        return c, n    else:        return (a + c)/2., nif __name__ == "__main__":    """    The keyword 'lambda' creates an anonymous function    with input argument x    """    y = lambda x: x**3 + 2.0*x**2 - 5.    root, iterations = incremental_search(y, -5., 5., 0.001)    print "Root is:", root    print "Iterations:", iterations
#  1 change: 1 addition & 0 deletions1  
# B03898_03_codes/newton.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" The Newton-Raphson method """def newton(f, df, x, tol=0.001, maxiter=100):    """    :param f: The function to solve    :param df: The derivative function of f    :param x: Initial guess value of x    :param tol: The precision of the solution    :param maxiter: Maximum number of iterations    :return: The x-axis value of the root,                number of iterations used    """    n = 1    while n <= maxiter:        x1 = x - f(x)/df(x)        if abs(x1 - x) < tol:  # Root is very close            return x1, n        else:            x = x1            n += 1    return None, nif __name__ == "__main__":    y = lambda x: x**3 + 2*x**2 - 5    dy = lambda x: 3*x**2 + 4*x    root, iterations = newton(y, dy, 5.0, 0.00001, 100)    print "Root is:", root    print "Iterations:", iterations
#  1 change: 1 addition & 0 deletions1  
# B03898_03_codes/scipy_optimize.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======""""""Documentation athttp://docs.scipy.org/doc/scipy/reference/optimize.html"""import scipy.optimize as optimizey = lambda x: x**3 + 2.*x**2 - 5.dy = lambda x: 3.*x**2 + 4.*x# Call method: bisect(f, a, b[, args, xtol, rtol, maxiter, ...])print "Bisection method: %s" \      % optimize.bisect(y, -5., 5., xtol=0.00001)# Call method: newton(func, x0[, fprime, args, tol, ...])print "Newton's method: %s" \      % optimize.newton(y, 5., fprime=dy)# When fprime=None, then the secant method is used.print "Secant method: %s" \      % optimize.newton(y, 5.)# Call method: brentq(f, a, b[, args, xtol, rtol, maxiter, ...])print "Brent's method: %s" \      % optimize.brentq(y, -5., 5.)
#  1 change: 1 addition & 0 deletions1  
# B03898_03_codes/secant.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.======"""""" The secant root-finding method """def secant(f, a, b, tol=0.001, maxiter=100):    """    :param f: The function to solve    :param a: Initial x-axis guess value    :param b: Initial x-axis guess value, where b>a    :param tol: The precision of the solution    :param maxiter: Maximum number of iterations    :return: The x-axis value of the root,                number of iterations used    """    n = 1    while n <= maxiter:        c = b - f(b)*((b-a)/(f(b)-f(a)))        if abs(c-b) < tol:            return c, n        a = b        b = c        n += 1    return None, nif __name__ == "__main__":    y = lambda x: x**3 + 2*x**2 - 5    root, iterations = secant(y, -5.0, 5.0, 0.00001, 100)    print "Root is:", root    print "Iterations:", iterations
#  52 changes: 52 additions & 0 deletions52  
# B03898_04_codes/BinomialCRRLattice.py
# @@ -0,0 +1,52 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Price an option by the binomial CRR lattice """
# from BinomialCRROption import BinomialCRROption
# import numpy as np


# class BinomialCRRLattice(BinomialCRROption):

#     def _setup_parameters_(self):
#         super(BinomialCRRLattice, self)._setup_parameters_()
#         self.M = 2*self.N + 1

#     def _initialize_stock_price_tree_(self):
#         self.STs = np.zeros(self.M)
#         self.STs[0] = self.S0 * self.u**self.N

#         for i in range(self.M)[1:]:
#             self.STs[i] = self.STs[i-1]*self.d

#     def _initialize_payoffs_tree_(self):
#         odd_nodes = self.STs[::2]
#         return np.maximum(
#             0, (odd_nodes - self.K) if self.is_call
#             else(self.K - odd_nodes))

#     def __check_early_exercise__(self, payoffs, node):
#         self.STs = self.STs[1:-1]  # Shorten the ends of the list
#         odd_STs = self.STs[::2]
#         early_ex_payoffs = \
#             (odd_STs-self.K) if self.is_call \
#             else (self.K-odd_STs)
#         payoffs = np.maximum(payoffs, early_ex_payoffs)

#         return payoffs

# if __name__ == "__main__":
#     from BinomialCRRLattice import BinomialCRRLattice
#     eu_option = BinomialCRRLattice(
#         50, 50, 0.05, 0.5, 2,
#         {"sigma": 0.3, "is_call": False})
#     print "European put: %s" % eu_option.price()

#     am_option = BinomialCRRLattice(
#         50, 50, 0.05, 0.5, 2,
#         {"sigma": 0.3, "is_call": False, "is_eu": False})
#     print "American put: %s" % am_option.price()
#  32 changes: 32 additions & 0 deletions32  
# B03898_04_codes/BinomialCRROption.py
# @@ -0,0 +1,32 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Price an option by the binomial CRR model """
# from BinomialTreeOption import BinomialTreeOption
# import math


# class BinomialCRROption(BinomialTreeOption):

#     def _setup_parameters_(self):
#         self.u = math.exp(self.sigma * math.sqrt(self.dt))
#         self.d = 1./self.u
#         self.qu = (math.exp((self.r-self.div)*self.dt) -
#                    self.d)/(self.u-self.d)
#         self.qd = 1-self.qu

# if __name__ == "__main__":
#     from BinomialCRROption import BinomialCRROption
#     eu_option = BinomialCRROption(
#         50, 50, 0.05, 0.5, 2,
#         {"sigma": 0.3, "is_call": False})
#     print "European put: %s" % eu_option.price()

#     am_option = BinomialCRROption(
#         50, 50, 0.05, 0.5, 2,
#         {"sigma": 0.3, "is_call": False, "is_eu": False})
#     print "American put: %s" % am_option.price()
#  67 changes: 67 additions & 0 deletions67  
# B03898_04_codes/BinomialEuropeanOption.py
# @@ -0,0 +1,67 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Price a European option by the binomial tree model """
# from StockOption import StockOption
# import math
# import numpy as np


# class BinomialEuropeanOption(StockOption):

#     def _setup_parameters_(self):
#         """ Required calculations for the model """
#         self.M = self.N + 1  # Number of terminal nodes of tree
#         self.u = 1 + self.pu  # Expected value in the up state
#         self.d = 1 - self.pd  # Expected value in the down state
#         self.qu = (math.exp((self.r-self.div)*self.dt) -
#                    self.d) / (self.u-self.d)
#         self.qd = 1-self.qu

#     def _initialize_stock_price_tree_(self):
#         # Initialize terminal price nodes to zeros
#         self.STs = np.zeros(self.M)

#         # Calculate expected stock prices for each node
#         for i in range(self.M):
#             self.STs[i] = self.S0*(self.u**(self.N-i))*(self.d**i)

#     def _initialize_payoffs_tree_(self):
#         # Get payoffs when the option expires at terminal nodes
#         payoffs = np.maximum(
#             0, (self.STs-self.K) if self.is_call
#             else (self.K-self.STs))

#         return payoffs

#     def _traverse_tree_(self, payoffs):
#         # Starting from the time the option expires, traverse
#         # backwards and calculate discounted payoffs at each node
#         for i in range(self.N):
#             payoffs = (payoffs[:-1] * self.qu +
#                        payoffs[1:] * self.qd) * self.df

#         return payoffs

#     def __begin_tree_traversal__(self):
#         payoffs = self._initialize_payoffs_tree_()
#         return self._traverse_tree_(payoffs)

#     def price(self):
#         """ The pricing implementation """
#         self._setup_parameters_()
#         self._initialize_stock_price_tree_()
#         payoffs = self.__begin_tree_traversal__()

#         return payoffs[0] # Option value converges to first node

# if __name__ == "__main__":
#     from BinomialEuropeanOption import BinomialEuropeanOption
#     eu_option = BinomialEuropeanOption(
#         50, 50, 0.05, 0.5, 2,
#         {"pu": 0.2, "pd": 0.2, "is_call": False})
#     print eu_option.price()
#  48 changes: 48 additions & 0 deletions48  
# B03898_04_codes/BinomialLROption.py
# @@ -0,0 +1,48 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Price an option by the Leisen-Reimer tree """
# from BinomialTreeOption import BinomialTreeOption
# import math


# class BinomialLROption(BinomialTreeOption):

#     def _setup_parameters_(self):
#         odd_N = self.N if (self.N%2 == 1) else (self.N+1)
#         d1 = (math.log(self.S0/self.K) +
#               ((self.r-self.div) +
#                (self.sigma**2)/2.) *
#               self.T) / (self.sigma * math.sqrt(self.T))
#         d2 = (math.log(self.S0/self.K) +
#               ((self.r-self.div) -
#                (self.sigma**2)/2.) *
#               self.T) / (self.sigma * math.sqrt(self.T))
#         pp_2_inversion = \
#             lambda z, n: \
#             .5 + math.copysign(1, z) * \
#             math.sqrt(.25 - .25 * math.exp(
#                 -((z/(n+1./3.+.1/(n+1)))**2.)*(n+1./6.)))
#         pbar = pp_2_inversion(d1, odd_N)

#         self.p = pp_2_inversion(d2, odd_N)
#         self.u = 1/self.df * pbar/self.p
#         self.d = (1/self.df - self.p*self.u)/(1-self.p)
#         self.qu = self.p
#         self.qd = 1-self.p

# if __name__ == "__main__":
#     from BinomialLROption import BinomialLROption
#     eu_option = BinomialLROption(
#         50, 50, 0.05, 0.5, 3,
#         {"sigma": 0.3, "is_call": False})
#     print "European put: %s" % eu_option.price()

#     am_option = BinomialLROption(
#         50, 50, 0.05, 0.5, 3,
#         {"sigma": 0.3, "is_call": False, "is_eu": False})
#     print "American put: %s" % am_option.price()
#  69 changes: 69 additions & 0 deletions69  
# B03898_04_codes/BinomialLRWithGreeks.py
# @@ -0,0 +1,69 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Compute option price, delta and gamma by the LR tree """
# from BinomialLROption import BinomialLROption
# import numpy as np


# class BinomialLRWithGreeks(BinomialLROption):

#     def __new_stock_price_tree__(self):
#         """
#         Create additional layer of nodes to our
#         original stock price tree
#         """
#         self.STs = [np.array([self.S0*self.u/self.d,
#                               self.S0,
#                               self.S0*self.d/self.u])]

#         for i in range(self.N):
#             prev_branches = self.STs[-1]
#             st = np.concatenate((prev_branches * self.u,
#                                  [prev_branches[-1] * self.d]))
#             self.STs.append(st)

#     def price(self):
#         self._setup_parameters_()
#         self.__new_stock_price_tree__()
#         payoffs = self.__begin_tree_traversal__()

#         """ Option value is now in the middle node at t=0"""
#         option_value = payoffs[len(payoffs)/2]

#         payoff_up = payoffs[0]
#         payoff_down = payoffs[-1]
#         S_up = self.STs[0][0]
#         S_down = self.STs[0][-1]
#         dS_up = S_up - self.S0
#         dS_down = self.S0 - S_down

#         """ Get delta value """
#         dS = S_up - S_down
#         dV = payoff_up - payoff_down
#         delta = dV/dS

#         """ Get gamma value """
#         gamma = ((payoff_up-option_value)/dS_up -
#                  (option_value-payoff_down)/dS_down) / \
#                 ((self.S0+S_up)/2. - (self.S0+S_down)/2.)

#         return option_value, delta, gamma

# if __name__ == "__main__":
#     from BinomialLRWithGreeks import BinomialLRWithGreeks
#     eu_call = BinomialLRWithGreeks(
#         50, 50, 0.05, 0.5, 300, {"sigma": 0.3, "is_call": True})
#     results = eu_call.price()
#     print "European call values"
#     print "Price: %s\nDelta: %s\nGamma: %s" % results

#     eu_put = BinomialLRWithGreeks(
#         50, 50, 0.05, 0.5, 300, {"sigma":0.3, "is_call": False})
#     results = eu_put.price()
#     print "European put values"
#     print "Price: %s\nDelta: %s\nGamma: %s" % results
#  76 changes: 76 additions & 0 deletions76  
# B03898_04_codes/BinomialTreeOption.py
# @@ -0,0 +1,76 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Price a European or American option by the binomial tree """
# from StockOption import StockOption
# import math
# import numpy as np


# class BinomialTreeOption(StockOption):

#     def _setup_parameters_(self):
#         self.u = 1 + self.pu  # Expected value in the up state
#         self.d = 1 - self.pd  # Expected value in the down state
#         self.qu = (math.exp((self.r-self.div)*self.dt) -
#                    self.d)/(self.u-self.d)
#         self.qd = 1-self.qu

#     def _initialize_stock_price_tree_(self):
#         # Initialize a 2D tree at T=0
#         self.STs = [np.array([self.S0])]

#         # Simulate the possible stock prices path
#         for i in range(self.N):
#             prev_branches = self.STs[-1]
#             st = np.concatenate((prev_branches*self.u,
#                                  [prev_branches[-1]*self.d]))
#             self.STs.append(st)  # Add nodes at each time step

#     def _initialize_payoffs_tree_(self):
#         # The payoffs when option expires
#         return np.maximum(
#             0, (self.STs[self.N]-self.K) if self.is_call
#             else (self.K-self.STs[self.N]))

#     def __check_early_exercise__(self, payoffs, node):
#         early_ex_payoff = \
#             (self.STs[node] - self.K) if self.is_call \
#             else (self.K - self.STs[node])

#         return np.maximum(payoffs, early_ex_payoff)

#     def _traverse_tree_(self, payoffs):
#         for i in reversed(range(self.N)):
#             # The payoffs from NOT exercising the option
#             payoffs = (payoffs[:-1] * self.qu +
#                        payoffs[1:] * self.qd) * self.df

#             # Payoffs from exercising, for American options
#             if not self.is_european:
#                 payoffs = self.__check_early_exercise__(payoffs,
#                                                         i)

#         return payoffs

#     def __begin_tree_traversal__(self):
#         payoffs = self._initialize_payoffs_tree_()
#         return self._traverse_tree_(payoffs)

#     def price(self):
#         self._setup_parameters_()
#         self._initialize_stock_price_tree_()
#         payoffs = self.__begin_tree_traversal__()

#         return payoffs[0]

# if __name__ == "__main__":
#     from BinomialTreeOption import BinomialTreeOption
#     am_option = BinomialTreeOption(
#         50, 50, 0.05, 0.5, 2,
#         {"pu": 0.2, "pd": 0.2, "is_call": False, "is_eu": False})
#     print am_option.price()
#  102 changes: 102 additions & 0 deletions102  
# B03898_04_codes/FDCnAm.py
# @@ -0,0 +1,102 @@
# """
# README
# ======
# This file contains Python codes.
# ====
# """

# """ Price an American option by the Crank-Nicolson method """
# import numpy as np
# import sys

# from FDCnEu import FDCnEu


# class FDCnAm(FDCnEu):

#     def __init__(self, S0, K, r, T, sigma, Smax, M, N, omega, tol,
#                  is_call=True):
#         super(FDCnAm, self).__init__(
#             S0, K, r, T, sigma, Smax, M, N, is_call)
#         self.omega = omega
#         self.tol = tol
#         self.i_values = np.arange(self.M+1)
#         self.j_values = np.arange(self.N+1)

#     def _setup_boundary_conditions_(self):
#         if self.is_call:
#             self.payoffs = np.maximum(
#                 self.boundary_conds[1:self.M]-self.K, 0)
#         else:
#             self.payoffs = np.maximum(
#                 self.K-self.boundary_conds[1:self.M], 0)

#         self.past_values = self.payoffs
#         self.boundary_values = self.K * \
#                                np.exp(-self.r *
#                                       self.dt *
#                                       (self.N-self.j_values))

#     def _traverse_grid_(self):
#         """ Solve using linear systems of equations """
#         aux = np.zeros(self.M-1)
#         new_values = np.zeros(self.M-1)

#         for j in reversed(range(self.N)):
#             aux[0] = self.alpha[1]*(self.boundary_values[j] +
#                                     self.boundary_values[j+1])
#             rhs = np.dot(self.M2, self.past_values) + aux
#             old_values = np.copy(self.past_values)
#             error = sys.float_info.max

#             while self.tol < error:
#                 new_values[0] = \
#                     max(self.payoffs[0],
#                         old_values[0] +
#                         self.omega/(1-self.beta[1]) *
#                         (rhs[0] -
#                          (1-self.beta[1])*old_values[0] +
#                          (self.gamma[1]*old_values[1])))

#                 for k in range(self.M-2)[1:]:
#                     new_values[k] = \
#                         max(self.payoffs[k],
#                             old_values[k] +
#                             self.omega/(1-self.beta[k+1]) *
#                             (rhs[k] +
#                              self.alpha[k+1]*new_values[k-1] -
#                              (1-self.beta[k+1])*old_values[k] +
#                              self.gamma[k+1]*old_values[k+1]))

#                 new_values[-1] = \
#                     max(self.payoffs[-1],
#                         old_values[-1] +
#                         self.omega/(1-self.beta[-2]) *
#                         (rhs[-1] +
#                          self.alpha[-2]*new_values[-2] -
#                          (1-self.beta[-2])*old_values[-1]))

#                 error = np.linalg.norm(new_values - old_values)
#                 old_values = np.copy(new_values)

#             self.past_values = np.copy(new_values)

#         self.values = np.concatenate(([self.boundary_values[0]],
#                                       new_values,
#                                       [0]))

#     def _interpolate_(self):
#         # Use linear interpolation on final values as 1D array
#         return np.interp(self.S0,
#                          self.boundary_conds,
#                          self.values)

# if __name__ == "__main__":
#     from FDCnDo import FDCnDo
#     option = FDCnAm(50, 50, 0.1, 5./12., 0.4, 100, 100,
#                     42, 1.2, 0.001)
#     print option.price()

#     option = FDCnAm(50, 50, 0.1, 5./12., 0.4, 100, 100,
#                     42, 1.2, 0.001, False)
#     print option.price()
#  37 changes: 37 additions & 0 deletions37  
# B03898_04_codes/FDCnDo.py
# @@ -0,0 +1,37 @@
# """
# README
# ======
# This file contains Python codes.
# ====
# """

# """
# Price a down-and-out option by the Crank-Nicolson
# method of finite differences.
# """
# import numpy as np

# from FDCnEu import FDCnEu


# class FDCnDo(FDCnEu):

#     def __init__(self, S0, K, r, T, sigma, Sbarrier, Smax, M, N,
#                  is_call=True):
#         super(FDCnDo, self).__init__(
#             S0, K, r, T, sigma, Smax, M, N, is_call)
#         self.dS = (Smax-Sbarrier)/float(self.M)
#         self.boundary_conds = np.linspace(Sbarrier,
#                                           Smax,
#                                           self.M+1)
#         self.i_values = self.boundary_conds/self.dS

# if __name__ == "__main__":
#     from FDCnDo import FDCnDo
#     option = FDCnDo(50, 50, 0.1, 5./12., 0.4, 40, 100, 120,
#                     500)
#     print option.price()

#     option = FDCnDo(50, 50, 0.1, 5./12., 0.4, 40, 100, 120,
#                     500, False)
#     print option.price()
#  54 changes: 54 additions & 0 deletions54  
# B03898_04_codes/FDCnEu.py
# @@ -0,0 +1,54 @@
# """
# README
# ======
# This file contains Python codes.
# ====
# """

# """ Crank-Nicolson method of Finite Differences """
# import numpy as np
# import scipy.linalg as linalg

# from FDExplicitEu import FDExplicitEu


# class FDCnEu(FDExplicitEu):

#     def _setup_coefficients_(self):
#         self.alpha = 0.25*self.dt*(
#             (self.sigma**2)*(self.i_values**2) -
#             self.r*self.i_values)
#         self.beta = -self.dt*0.5*(
#             (self.sigma**2)*(self.i_values**2) +
#             self.r)
#         self.gamma = 0.25*self.dt*(
#             (self.sigma**2)*(self.i_values**2) +
#             self.r*self.i_values)
#         self.M1 = -np.diag(self.alpha[2:self.M], -1) + \
#                   np.diag(1-self.beta[1:self.M]) - \
#                   np.diag(self.gamma[1:self.M-1], 1)
#         self.M2 = np.diag(self.alpha[2:self.M], -1) + \
#                   np.diag(1+self.beta[1:self.M]) + \
#                   np.diag(self.gamma[1:self.M-1], 1)

#     def _traverse_grid_(self):
#         """ Solve using linear systems of equations """
#         P, L, U = linalg.lu(self.M1)

#         for j in reversed(range(self.N)):
#             x1 = linalg.solve(L,
#                               np.dot(self.M2,
#                                      self.grid[1:self.M, j+1]))
#             x2 = linalg.solve(U, x1)
#             self.grid[1:self.M, j] = x2

# if __name__ == "__main__":
#     from FDCnEu import FDCnEu
#     option = FDCnEu(50, 50, 0.1, 5./12., 0.4, 100, 100,
#                     100, False)
#     print option.price()

#     option = FDCnEu(50, 50, 0.1, 5./12., 0.4, 100, 100,
#                     1000, False)
#     print option.price()

#  59 changes: 59 additions & 0 deletions59  
# B03898_04_codes/FDExplicitEu.py
# @@ -0,0 +1,59 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Explicit method of Finite Differences """
# import numpy as np

# from FiniteDifferences import FiniteDifferences


# class FDExplicitEu(FiniteDifferences):

#     def _setup_boundary_conditions_(self):
#         if self.is_call:
#             self.grid[:, -1] = np.maximum(
#                 self.boundary_conds - self.K, 0)
#             self.grid[-1, :-1] = (self.Smax - self.K) * \
#                                  np.exp(-self.r *
#                                         self.dt *
#                                         (self.N-self.j_values))
#         else:
#             self.grid[:, -1] = \
#                 np.maximum(self.K-self.boundary_conds, 0)
#             self.grid[0, :-1] = (self.K - self.Smax) * \
#                                np.exp(-self.r *
#                                       self.dt *
#                                       (self.N-self.j_values))

#     def _setup_coefficients_(self):
#         self.a = 0.5*self.dt*((self.sigma**2) *
#                               (self.i_values**2) -
#                               self.r*self.i_values)
#         self.b = 1 - self.dt*((self.sigma**2) *
#                               (self.i_values**2) +
#                               self.r)
#         self.c = 0.5*self.dt*((self.sigma**2) *
#                               (self.i_values**2) +
#                               self.r*self.i_values)

#     def _traverse_grid_(self):
#         for j in reversed(self.j_values):
#             for i in range(self.M)[2:]:
#                 self.grid[i,j] = self.a[i]*self.grid[i-1,j+1] +\
#                                  self.b[i]*self.grid[i,j+1] + \
#                                  self.c[i]*self.grid[i+1,j+1]


# if __name__ == "__main__":
#     from FDExplicitEu import FDExplicitEu
#     option = FDExplicitEu(50, 50, 0.1, 5./12., 0.4, 100, 100,
#                                   1000, False)
#     print option.price()

#     option = FDExplicitEu(50, 50, 0.1, 5./12., 0.4, 100, 100,
#                           100, False)
#     print option.price()
#  51 changes: 51 additions & 0 deletions51  
# B03898_04_codes/FDImplicitEu.py
# @@ -0,0 +1,51 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """
# Price a European option by the implicit method
# of finite differences.
# """
# import numpy as np
# import scipy.linalg as linalg

# from FDExplicitEu import FDExplicitEu


# class FDImplicitEu(FDExplicitEu):

#     def _setup_coefficients_(self):
#         self.a = 0.5*(self.r*self.dt*self.i_values -
#                       (self.sigma**2)*self.dt*(self.i_values**2))
#         self.b = 1 + \
#                  (self.sigma**2)*self.dt*(self.i_values**2) + \
#                  self.r*self.dt
#         self.c = -0.5*(self.r * self.dt*self.i_values +
#                        (self.sigma**2)*self.dt*(self.i_values**2))
#         self.coeffs = np.diag(self.a[2:self.M], -1) + \
#                       np.diag(self.b[1:self.M]) + \
#                       np.diag(self.c[1:self.M-1], 1)

#     def _traverse_grid_(self):
#         """ Solve using linear systems of equations """
#         P, L, U = linalg.lu(self.coeffs)
#         aux = np.zeros(self.M-1)

#         for j in reversed(range(self.N)):
#             aux[0] = np.dot(-self.a[1], self.grid[0, j])
#             x1 = linalg.solve(L, self.grid[1:self.M, j+1]+aux)
#             x2 = linalg.solve(U, x1)
#             self.grid[1:self.M, j] = x2

# if __name__ == "__main__":
#     from FDImplicitEu import FDImplicitEu
#     option = FDImplicitEu(50, 50, 0.1, 5./12., 0.4, 100, 100,
#                           100, False)
#     print option.price()

#     option = FDImplicitEu(50, 50, 0.1, 5./12., 0.4, 100, 100,
#                           10000, False)
#     print option.price()
#  55 changes: 55 additions & 0 deletions55  
# B03898_04_codes/FiniteDifferences.py
# @@ -0,0 +1,55 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Shared attributes and functions of FD """
# import numpy as np


# class FiniteDifferences(object):

#     def __init__(self, S0, K, r, T, sigma, Smax, M, N,
#                  is_call=True):
#         self.S0 = S0
#         self.K = K
#         self.r = r
#         self.T = T
#         self.sigma = sigma
#         self.Smax = Smax
#         self.M, self.N = int(M), int(N)  # Ensure M&N are integers
#         self.is_call = is_call

#         self.dS = Smax / float(self.M)
#         self.dt = T / float(self.N)
#         self.i_values = np.arange(self.M)
#         self.j_values = np.arange(self.N)
#         self.grid = np.zeros(shape=(self.M+1, self.N+1))
#         self.boundary_conds = np.linspace(0, Smax, self.M+1)

#     def _setup_boundary_conditions_(self):
#         pass

#     def _setup_coefficients_(self):
#         pass

#     def _traverse_grid_(self):
#         """  Iterate the grid backwards in time """
#         pass

#     def _interpolate_(self):
#         """
#         Use piecewise linear interpolation on the initial
#         grid column to get the closest price at S0.
#         """
#         return np.interp(self.S0,
#                          self.boundary_conds,
#                          self.grid[:, 0])

#     def price(self):
#         self._setup_boundary_conditions_()
#         self._setup_coefficients_()
#         self._traverse_grid_()
#         return self._interpolate_()
#  64 changes: 64 additions & 0 deletions64  
# B03898_04_codes/ImpliedVolatilityModel.py
# @@ -0,0 +1,64 @@
# """
# README
# ======
# This file contains Python codes.
# ====
# """

# """
# Get implied volatilities from a Leisen-Reimer binomial
# tree using the bisection method as the numerical procedure.
# """
# from bisection import bisection
# from BinomialLROption import BinomialLROption

# class ImpliedVolatilityModel(object):

#     def __init__(self, S0, r, T, div, N,
#                  is_call=False):
#         self.S0 = S0
#         self.r = r
#         self.T = T
#         self.div = div
#         self.N = N
#         self.is_call = is_call

#     def _option_valuation_(self, K, sigma):
#         # Use the binomial Leisen-Reimer tree
#         lr_option = BinomialLROption(
#             self.S0, K, self.r,  self.T, self.N,
#             {"sigma": sigma,
#              "is_call": self.is_call,
#              "div": self.div})
#         return lr_option.price()

#     def get_implied_volatilities(self, Ks, opt_prices):
#         impvols = []
#         for i in range(len(Ks)):
#             # Bind f(sigma) for use by the bisection method
#             f = lambda sigma: \
#                 self._option_valuation_(
#                     Ks[i], sigma) - opt_prices[i]
#             impv = bisection(f, 0.01, 0.99, 0.0001, 100)[0]
#             impvols.append(impv)
#         return impvols

# if __name__ == "__main__":
#     # The data
#     strikes = [75, 80, 85, 90, 92.5, 95, 97.5,
#                100, 105, 110, 115, 120, 125]
#     put_prices = [0.16, 0.32, 0.6, 1.22, 1.77, 2.54, 3.55,
#                   4.8, 7.75, 11.8, 15.96, 20.75, 25.81]

#     model = ImpliedVolatilityModel(99.62, 0.0248, 78/365.,
#                                    0.0182, 77, is_call=False)
#     impvols_put = model.get_implied_volatilities(strikes,
#                                                  put_prices)

#     # Begin plotting the results
#     import matplotlib.pyplot as plt
#     plt.plot(strikes, impvols_put)
#     plt.xlabel('Strike Prices')
#     plt.ylabel('Implied Volatilities')
#     plt.title('AAPL Put Implied Volatilities expiring in 78 days')
#     plt.show()
#  33 changes: 33 additions & 0 deletions33  
# B03898_04_codes/StockOption.py
# @@ -0,0 +1,33 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Store common attributes of a stock option """
# import math


# class StockOption(object):

#     def __init__(self, S0, K, r, T, N, params):
#         self.S0 = S0
#         self.K = K
#         self.r = r
#         self.T = T
#         self.N = max(1, N) # Ensure N have at least 1 time step
#         self.STs = None  # Declare the stock prices tree

#         """ Optional parameterss used by derived classes """
#         self.pu = params.get("pu", 0)  # Probability of up state
#         self.pd = params.get("pd", 0)  # Probability of down state
#         self.div = params.get("div", 0)  # Divident yield
#         self.sigma = params.get("sigma", 0)  # Volatility
#         self.is_call = params.get("is_call", True)  # Call or put
#         self.is_european = params.get("is_eu", True)  # Eu or Am

#         """ Computed values """
#         self.dt = T/float(N)  # Single time step, in years
#         self.df = math.exp(
#             -(r-self.div) * self.dt)  # Discount factor
#  50 changes: 50 additions & 0 deletions50  
# B03898_04_codes/TrinomialLattice.py
# @@ -0,0 +1,50 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Price an option by the trinomial lattice """
# from TrinomialTreeOption import TrinomialTreeOption
# import numpy as np


# class TrinomialLattice(TrinomialTreeOption):

#     def _setup_parameters_(self):
#         super(TrinomialLattice, self)._setup_parameters_()
#         self.M = 2*self.N+1

#     def _initialize_stock_price_tree_(self):
#         self.STs = np.zeros(self.M)
#         self.STs[0] = self.S0 * self.u**self.N

#         for i in range(self.M)[1:]:
#             self.STs[i] = self.STs[i-1]*self.d

#     def _initialize_payoffs_tree_(self):
#         return np.maximum(
#             0, (self.STs-self.K) if self.is_call
#             else(self.K-self.STs))

#     def __check_early_exercise__(self, payoffs, node):
#         self.STs = self.STs[1:-1]  # Shorten the ends of the list
#         early_ex_payoffs = \
#             (self.STs-self.K) if self.is_call \
#             else(self.K-self.STs)
#         payoffs = np.maximum(payoffs, early_ex_payoffs)

#         return payoffs

# if __name__ == "__main__":
#     from TrinomialLattice import TrinomialLattice
#     eu_option = TrinomialLattice(
#         50, 50, 0.05, 0.5, 2,
#         {"sigma": 0.3, "is_call":False})
#     print "European put: %s" % eu_option.price()

#     am_option = TrinomialLattice(
#         50, 50, 0.05, 0.5, 2,
#         {"sigma": 0.3, "is_call": False, "is_eu": False})
#     print "American put: %s" % am_option.price()
#  69 changes: 69 additions & 0 deletions69  
# B03898_04_codes/TrinomialTreeOption.py
# @@ -0,0 +1,69 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Price an option by the Boyle trinomial tree """
# from BinomialTreeOption import BinomialTreeOption
# import math
# import numpy as np


# class TrinomialTreeOption(BinomialTreeOption):

#     def _setup_parameters_(self):
#         """ Required calculations for the model """
#         self.u = math.exp(self.sigma*math.sqrt(2.*self.dt))
#         self.d = 1/self.u
#         self.m = 1
#         self.qu = ((math.exp((self.r-self.div) *
#                              self.dt/2.) -
#                     math.exp(-self.sigma *
#                              math.sqrt(self.dt/2.))) /
#                    (math.exp(self.sigma *
#                              math.sqrt(self.dt/2.)) -
#                     math.exp(-self.sigma *
#                              math.sqrt(self.dt/2.))))**2
#         self.qd = ((math.exp(self.sigma *
#                              math.sqrt(self.dt/2.)) -
#                     math.exp((self.r-self.div) *
#                              self.dt/2.)) /
#                    (math.exp(self.sigma *
#                              math.sqrt(self.dt/2.)) -
#                     math.exp(-self.sigma *
#                              math.sqrt(self.dt/2.))))**2.

#         self.qm = 1 - self.qu - self.qd

#     def _initialize_stock_price_tree_(self):
#         """ Initialize a 2D tree at t=0 """
#         self.STs = [np.array([self.S0])]

#         for i in range(self.N):
#             prev_nodes = self.STs[-1]
#             self.ST = np.concatenate(
#                 (prev_nodes*self.u, [prev_nodes[-1]*self.m,
#                                      prev_nodes[-1]*self.d]))
#             self.STs.append(self.ST)

#     def _traverse_tree_(self, payoffs):
#         """ Traverse the tree backwards """
#         for i in reversed(range(self.N)):
#             payoffs = (payoffs[:-2] * self.qu +
#                        payoffs[1:-1] * self.qm +
#                        payoffs[2:] * self.qd) * self.df

#             if not self.is_european:
#                 payoffs = self.__check_early_exercise__(payoffs,
#                                                         i)

#         return payoffs

# if __name__ == "__main__":
#     from TrinomialTreeOption import TrinomialTreeOption
#     print "European put:", TrinomialTreeOption(
#         50, 50, 0.05, 0.5, 2, {"sigma": 0.3, "is_call": False}).price()
#     print "American put:", TrinomialTreeOption(
#         50, 50, 0.05, 0.5, 2,  {"sigma": 0.3, "is_call": False, "is_eu": False}).price()
#  1 change: 1 addition & 0 deletions1  
# B03898_04_codes/bisection.py
# @@ -0,0 +1 @@
# """README======This file contains Python codes.======"""""" The bisection method """def bisection(f, a, b, tol=0.1, maxiter=10):    """    :param f: The function to solve    :param a: The x-axis value where f(a)<0    :param b: The x-axis value where f(b)>0    :param tol: The precision of the solution    :param maxiter: Maximum number of iterations    :return: The x-axis value of the root,                number of iterations used    """    c = (a+b)*0.5  # Declare c as the midpoint ab    n = 1  # Start with 1 iteration    while n <= maxiter:        c = (a+b)*0.5        if f(c) == 0 or abs(a-b)*0.5 < tol:            # Root is found or is very close            return c, n        n += 1        if f(c) < 0:            a = c        else:            b = c                    return c, nif __name__ == "__main__":    y = lambda x: x**3 + 2*x**2 - 5    root, iterations = bisection(y, -5, 5, 0.00001, 100)    print "Root is:", root    print "Iterations:", iterations
#  96 changes: 96 additions & 0 deletions96  
# B03898_05_Codes/BootstrapYieldCurve.py
# @@ -0,0 +1,96 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Bootstrapping the yield curve """
# import math

# class BootstrapYieldCurve(object):

#     def __init__(self):
#         self.zero_rates = dict()  # Map each T to a zero rate
#         self.instruments = dict()  # Map each T to an instrument

#     def add_instrument(self, par, T, coup, price,
#                        compounding_freq=2):
#         """  Save instrument info by maturity """
#         self.instruments[T] = (par, coup, price, compounding_freq)

#     def get_zero_rates(self):
#         """  Calculate a list of available zero rates """
#         self.__bootstrap_zero_coupons__()
#         self.__get_bond_spot_rates__()
#         return [self.zero_rates[T] for T in self.get_maturities()]

#     def get_maturities(self):
#         """ Return sorted maturities from added instruments. """
#         return sorted(self.instruments.keys())

#     def __bootstrap_zero_coupons__(self):
#         """ Get zero rates from zero coupon bonds """
#         for T in self.instruments.iterkeys():
#             (par, coup, price, freq) = self.instruments[T]
#             if coup == 0:
#                 self.zero_rates[T] = \
#                     self.zero_coupon_spot_rate(par, price, T)

#     def __get_bond_spot_rates__(self):
#         """ Get spot rates for every marurity available """
#         for T in self.get_maturities():
#             instrument = self.instruments[T]
#             (par, coup, price, freq) = instrument

#             if coup != 0:
#                 self.zero_rates[T] = \
#                     self.__calculate_bond_spot_rate__(
#                         T, instrument)

#     def __calculate_bond_spot_rate__(self, T, instrument):
#         """ Get spot rate of a bond by bootstrapping """
#         try:
#             (par, coup, price, freq) = instrument
#             periods = T * freq  # Number of coupon payments
#             value = price
#             per_coupon = coup / freq  # Coupon per period

#             for i in range(int(periods)-1):
#                 t = (i+1)/float(freq)
#                 spot_rate = self.zero_rates[t]
#                 discounted_coupon = per_coupon * \
#                                     math.exp(-spot_rate*t)
#                 value -= discounted_coupon

#             # Derive spot rate for a particular maturity
#             last_period = int(periods)/float(freq)        
#             spot_rate = -math.log(value /
#                                   (par+per_coupon))/last_period
#             return spot_rate

#         except:
#             print "Error: spot rate not found for T=%s" % t

#     def zero_coupon_spot_rate(self, par, price, T):
#         """ Get zero rate of a zero coupon bond """
#         spot_rate = math.log(par/price)/T
#         return spot_rate


# if __name__ == "__main__":
#     yield_curve = BootstrapYieldCurve()
#     yield_curve.add_instrument(100, 0.25, 0., 97.5)
#     yield_curve.add_instrument(100, 0.5, 0., 94.9)
#     yield_curve.add_instrument(100, 1.0, 0., 90.)
#     yield_curve.add_instrument(100, 1.5, 8, 96., 2)
#     yield_curve.add_instrument(100, 2., 12, 101.6, 2)
#     y = yield_curve.get_zero_rates()
#     x = yield_curve.get_maturities()

#     import matplotlib.pyplot as plt
#     plt.plot(x, y)
#     plt.title("Zero Curve")
#     plt.ylabel("Zero Rate (%)")
#     plt.xlabel("Maturity in Years")
#     plt.show()
#  221 changes: 221 additions & 0 deletions221  
# B03898_05_Codes/VasicekCZCB.py
# @@ -0,0 +1,221 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Price a callable zero coupon bond by the Vasicek model """
# import math
# import numpy as np
# import scipy.stats as st


# class VasicekCZCB:

#     def __init__(self):
#         self.norminv = st.distributions.norm.ppf
#         self.norm = st.distributions.norm.cdf        

#     def vasicek_czcb_values(self, r0, R, ratio, T, sigma, kappa,
#                             theta, M, prob=1e-6,
#                             max_policy_iter=10,
#                             grid_struct_const=0.25, rs=None):
#         r_min, dr, N, dtau = \
#             self.vasicek_params(r0, M, sigma, kappa, theta,
#                                 T, prob, grid_struct_const, rs)
#         r = np.r_[0:N]*dr + r_min
#         v_mplus1 = np.ones(N)

#         for i in range(1, M+1):
#             K = self.exercise_call_price(R, ratio, i*dtau)
#             eex = np.ones(N)*K
#             subdiagonal, diagonal, superdiagonal = \
#                 self.vasicek_diagonals(sigma, kappa, theta,
#                                        r_min, dr, N, dtau)
#             v_mplus1, iterations = \
#                 self.iterate(subdiagonal, diagonal, superdiagonal,
#                              v_mplus1, eex, max_policy_iter)
#         return r, v_mplus1

#     def vasicek_params(self, r0, M, sigma, kappa, theta, T,
#                        prob, grid_struct_const=0.25, rs=None):
#         (r_min, r_max) = (rs[0], rs[-1]) if not rs is None \
#             else self.vasicek_limits(r0, sigma, kappa,
#                                      theta, T, prob)
#         dt = T/float(M)
#         N = self.calculate_N(grid_struct_const, dt,
#                              sigma, r_max, r_min)
#         dr = (r_max-r_min)/(N-1)
#         return r_min, dr, N, dt

#     def calculate_N(self, max_structure_const, dt,
#                     sigma, r_max, r_min):
#         N = 0
#         while True:
#             N += 1
#             grid_structure_interval = dt*(sigma**2)/(
#                 ((r_max-r_min)/float(N))**2)
#             if grid_structure_interval > max_structure_const:
#                 break

#         return N

#     def vasicek_limits(self, r0, sigma, kappa,
#                        theta, T, prob=1e-6):
#         er = theta+(r0-theta)*math.exp(-kappa*T)
#         variance = (sigma**2)*T if kappa==0 else \
#                     (sigma**2)/(2*kappa)*(1-math.exp(-2*kappa*T))
#         stdev = math.sqrt(variance)
#         r_min = self.norminv(prob, er, stdev)
#         r_max = self.norminv(1-prob, er, stdev)
#         return r_min, r_max

#     def vasicek_diagonals(self, sigma, kappa, theta,
#                           r_min, dr, N, dtau):
#         rn = np.r_[0:N]*dr + r_min
#         subdiagonals = kappa*(theta-rn)*dtau/(2*dr) - \
#                        0.5*(sigma**2)*dtau/(dr**2)
#         diagonals = 1 + rn*dtau + sigma**2*dtau/(dr**2)
#         superdiagonals = -kappa*(theta-rn)*dtau/(2*dr) - \
#                          0.5*(sigma**2)*dtau/(dr**2)

#         # Implement boundary conditions.
#         if N > 0:
#             v_subd0 = subdiagonals[0]
#             superdiagonals[0] = superdiagonals[0] - \
#                                 subdiagonals[0]
#             diagonals[0] += 2*v_subd0
#             subdiagonals[0] = 0

#         if N > 1:
#             v_superd_last = superdiagonals[-1]
#             superdiagonals[-1] = superdiagonals[-1] - \
#                                  subdiagonals[-1]
#             diagonals[-1] += 2*v_superd_last
#             superdiagonals[-1] = 0

#         return subdiagonals, diagonals, superdiagonals

#     def check_exercise(self, V, eex):
#         return V > eex

#     def exercise_call_price(self, R, ratio, tau):
#         K = ratio*np.exp(-R*tau)
#         return K

#     def vasicek_policy_diagonals(self, subdiagonal, diagonal,
#                                  superdiagonal, v_old, v_new,
#                                  eex):
#         has_early_exercise = self.check_exercise(v_new, eex)
#         subdiagonal[has_early_exercise] = 0
#         superdiagonal[has_early_exercise] = 0
#         policy = v_old/eex
#         policy_values = policy[has_early_exercise]
#         diagonal[has_early_exercise] = policy_values
#         return subdiagonal, diagonal, superdiagonal

#     def iterate(self, subdiagonal, diagonal, superdiagonal,
#                 v_old, eex, max_policy_iter=10):
#         v_mplus1 = v_old
#         v_m = v_old
#         change = np.zeros(len(v_old))
#         prev_changes = np.zeros(len(v_old))

#         iterations = 0
#         while iterations <= max_policy_iter:
#             iterations += 1

#             v_mplus1 = self.tridiagonal_solve(subdiagonal,
#                                               diagonal,
#                                               superdiagonal,
#                                               v_old)
#             subdiagonal, diagonal, superdiagonal = \
#                 self.vasicek_policy_diagonals(subdiagonal,
#                                               diagonal,
#                                               superdiagonal,
#                                               v_old,
#                                               v_mplus1,
#                                               eex)

#             is_eex = self.check_exercise(v_mplus1, eex)
#             change[is_eex] = 1

#             if iterations > 1:
#                 change[v_mplus1 != v_m] = 1

#             is_no_more_eex = False if True in is_eex else True
#             if is_no_more_eex:
#                 break

#             v_mplus1[is_eex] = eex[is_eex]
#             changes = (change == prev_changes)

#             is_no_further_changes = all((x == 1) for x in changes)
#             if is_no_further_changes:
#                 break

#             prev_changes = change
#             v_m = v_mplus1

#         return v_mplus1, (iterations-1)

#     def tridiagonal_solve(self, a, b, c, d):
#         nf = len(a)  # Number of equations
#         ac, bc, cc, dc = \
#             map(np.array, (a, b, c, d))  # Copy the array
#         for it in xrange(1, nf):
#             mc = ac[it]/bc[it-1]
#             bc[it] = bc[it] - mc*cc[it-1] 
#             dc[it] = dc[it] - mc*dc[it-1]

#         xc = ac
#         xc[-1] = dc[-1]/bc[-1]

#         for il in xrange(nf-2, -1, -1):
#             xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

#         del bc, cc, dc  # Delete variables from memory

#         return xc

# if __name__ == "__main__":
#     r0 = 0.05
#     R = 0.05
#     ratio = 0.95
#     sigma = 0.03
#     kappa = 0.15
#     theta = 0.05
#     prob = 1e-6
#     M = 250
#     max_policy_iter=10
#     grid_struct_interval = 0.25
#     rs = np.r_[0.0:2.0:0.1]

#     Vasicek = VasicekCZCB()
#     r, vals = Vasicek.vasicek_czcb_values(r0, R, ratio, 1.,
#                                           sigma, kappa, theta,
#                                           M, prob,
#                                           max_policy_iter,
#                                           grid_struct_interval,
#                                           rs)
#     import matplotlib.pyplot as plt
#     plt.title("Callable Zero Coupon Bond Values by r")
#     plt.plot(r, vals, label='1 yr')

#     for T in [5., 7., 10., 20.]:
#         r, vals = \
#             Vasicek.vasicek_czcb_values(r0, R, ratio, T,
#                                         sigma, kappa,
#                                         theta, M, prob,
#                                         max_policy_iter,
#                                         grid_struct_interval,
#                                         rs)
#         plt.plot(r, vals, label=str(T)+' yr',
#                  linestyle="--", marker=".")

#     plt.ylabel("Value ($)")
#     plt.xlabel("r")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#  26 changes: 26 additions & 0 deletions26  
# B03898_05_Codes/bond_convexity.py
# @@ -0,0 +1,26 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Calculate convexity of a bond """
# from bond_ytm import bond_ytm
# from bond_price import bond_price


# def bond_convexity(price, par, T, coup, freq, dy=0.01):
#     ytm = bond_ytm(price, par, T, coup, freq)

#     ytm_minus = ytm - dy    
#     price_minus = bond_price(par, T, ytm_minus, coup, freq)

#     ytm_plus = ytm + dy
#     price_plus = bond_price(par, T, ytm_plus, coup, freq)

#     convexity = (price_minus+price_plus-2*price)/(price*dy**2)
#     return convexity

# if __name__ == "__main__":
#     print bond_convexity(95.0428, 100, 1.5, 5.75, 2)
#  27 changes: 27 additions & 0 deletions27  
# B03898_05_Codes/bond_mod_duration.py
# @@ -0,0 +1,27 @@
# """"
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Calculate modified duration of a bond """
# from bond_ytm import bond_ytm
# from bond_price import bond_price


# def bond_mod_duration(price, par, T, coup, freq, dy=0.01):
#     ytm = bond_ytm(price, par, T, coup, freq)

#     ytm_minus = ytm - dy    
#     price_minus = bond_price(par, T, ytm_minus, coup, freq)

#     ytm_plus = ytm + dy
#     price_plus = bond_price(par, T, ytm_plus, coup, freq)

#     mduration = (price_minus-price_plus)/(2.*price*dy)
#     return mduration

# if __name__ == "__main__":
#     from bond_mod_duration import bond_mod_duration
#     print bond_mod_duration(95.04, 100, 1.5, 5.75, 2, 0.01)
#  22 changes: 22 additions & 0 deletions22  
# B03898_05_Codes/bond_price.py
# @@ -0,0 +1,22 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Get bond price from YTM """
# def bond_price(par, T, ytm, coup, freq=2):
#     freq = float(freq)
#     periods = T*freq
#     coupon = coup/100.*par/freq
#     dt = [(i+1)/freq for i in range(int(periods))]
#     price = sum([coupon/(1+ytm/freq)**(freq*t) for t in dt]) + \
#             par/(1+ytm/freq)**(freq*T)
#     return price


# if __name__ == "__main__":
#     from bond_ytm import bond_ytm
#     ytm = bond_ytm(95.0428, 100, 1.5, 5.75, 2)
#     print bond_price(100, 1.5, ytm, 5.75, 2)
#  27 changes: 27 additions & 0 deletions27  
# B03898_05_Codes/bond_ytm.py
# @@ -0,0 +1,27 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Get yield-to-maturity of a bond """
# import scipy.optimize as optimize


# def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
#     freq = float(freq)
#     periods = T*freq
#     coupon = coup/100.*par/freq
#     dt = [(i+1)/freq for i in range(int(periods))]
#     ytm_func = lambda(y): \
#         sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + \
#         par/(1+y/freq)**(freq*t) - price

#     return optimize.newton(ytm_func, guess)

# if __name__ == "__main__":
#     ytm = bond_ytm(95.0428, 100, 1.5, 5.75, 2)
#     print ytm


#  27 changes: 27 additions & 0 deletions27  
# B03898_05_Codes/brennan_schwartz.py
# @@ -0,0 +1,27 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Simulate interest rate path by the Brennan Schwartz model """
# import numpy as np

# def brennan_schwartz(r0, K, theta, sigma, T=1., N=10, seed=777):    
#     np.random.seed(seed)
#     dt = T/float(N)    
#     rates = [r0]
#     for i in range(N):
#         dr = K*(theta-rates[-1])*dt + \
#             sigma*rates[-1]*np.random.normal()
#         rates.append(rates[-1] + dr)
#     return range(N+1), rates

# if __name__ == "__main__":
#     x, y = brennan_schwartz(0.01875, 0.20, 0.01, 0.012, 10.,
#                             10000)

#     import matplotlib.pyplot as plt
#     plt.plot(x,y)
#     plt.show()
#  27 changes: 27 additions & 0 deletions27  
# B03898_05_Codes/cir.py
# @@ -0,0 +1,27 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Simulate interest rate path by the CIR model """
# import math
# import numpy as np

# def cir(r0, K, theta, sigma, T=1., N=10,seed=777):
#     np.random.seed(seed)
#     dt = T/float(N)    
#     rates = [r0]
#     for i in range(N):
#         dr = K*(theta-rates[-1])*dt + \
#             sigma*math.sqrt(rates[-1])*np.random.normal()
#         rates.append(rates[-1] + dr)
#     return range(N+1), rates

# if __name__ == "__main__":
#     x, y = cir(0.01875, 0.20, 0.01, 0.012, 10., 200)

#     import matplotlib.pyplot as plt
#     plt.plot(x,y)
#     plt.show()
#  45 changes: 45 additions & 0 deletions45  
# B03898_05_Codes/exact_zcb.py
# @@ -0,0 +1,45 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# import numpy as np

# """ Get zero coupon bond price by Vasicek model """
# def exact_zcb(theta, kappa, sigma, tau, r0=0.):
#     B = (1 - np.exp(-kappa*tau)) / kappa
#     A = np.exp((theta-(sigma**2)/(2*(kappa**2))) *
#                (B-tau) - (sigma**2)/(4*kappa)*(B**2))
#     return A * np.exp(-r0*B)

# import math
# def exercise_value(K, R, t):
#     return K*math.exp(-R*t)


# if __name__ == "__main__":
#     Ts = np.r_[0.0:25.5:0.5]
#     zcbs = [exact_zcb(0.5, 0.02, 0.03, t, 0.015) for t in Ts]

#     import matplotlib.pyplot as plt
#     plt.title("Zero Coupon Bond (ZCB) Values by Time")
#     plt.plot(Ts, zcbs, label='ZCB')
#     plt.ylabel("Value ($)")
#     plt.xlabel("Time in years")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     Ks = [exercise_value(0.95, 0.015, t) for t in Ts]
#     plt.title("Zero Coupon Bond (ZCB) "
#               "and Strike (K) Values by Time")
#     plt.plot(Ts, zcbs, label='ZCB')
#     plt.plot(Ts, Ks, label='K', linestyle="--", marker=".")
#     plt.ylabel("Value ($)")
#     plt.xlabel("Time in years")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#  46 changes: 46 additions & 0 deletions46  
# B03898_05_Codes/forward_rates.py
# @@ -0,0 +1,46 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """


# """
# Get a list of forward rates
# starting from the second time period
# """


# class ForwardRates(object):

#     def __init__(self):
#         self.forward_rates = []
#         self.spot_rates = dict()

#     def add_spot_rate(self, T, spot_rate):
#         self.spot_rates[T] = spot_rate

#     def __calculate_forward_rate___(self, T1, T2):
#         R1 = self.spot_rates[T1]
#         R2 = self.spot_rates[T2]
#         forward_rate = (R2*T2 - R1*T1)/(T2 - T1)
#         return forward_rate

#     def get_forward_rates(self):
#         periods = sorted(self.spot_rates.keys())
#         for T2, T1 in zip(periods, periods[1:]):
#             forward_rate = \
#                 self.__calculate_forward_rate___(T1, T2)
#             self.forward_rates.append(forward_rate)

#         return self.forward_rates

# if __name__ == "__main__":
#     fr = ForwardRates()
#     fr.add_spot_rate(0.25, 10.127)
#     fr.add_spot_rate(0.50, 10.469)
#     fr.add_spot_rate(1.00, 10.536)
#     fr.add_spot_rate(1.50, 10.681)
#     fr.add_spot_rate(2.00, 10.808)
#     print fr.get_forward_rates()
#  26 changes: 26 additions & 0 deletions26  
# B03898_05_Codes/rendleman_bartter.py
# @@ -0,0 +1,26 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Simulate interest rate path by the Rendleman-Barter model """
# import numpy as np

# def rendleman_bartter(r0, theta, sigma, T=1.,N=10,seed=777):        
#     np.random.seed(seed)
#     dt = T/float(N)    
#     rates = [r0]
#     for i in range(N):
#         dr = theta*rates[-1]*dt + \
#              sigma*rates[-1]*np.random.normal()
#         rates.append(rates[-1] + dr)
#     return range(N+1), rates

# if __name__ == "__main__":
#     x, y = rendleman_bartter(0.01875, 0.01, 0.012, 10., 200)

#     import matplotlib.pyplot as plt
#     plt.plot(x,y)
#     plt.show()
#  25 changes: 25 additions & 0 deletions25  
# B03898_05_Codes/vasicek.py
# @@ -0,0 +1,25 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Simulate interest rate path by the Vasicek model """
# import numpy as np

# def vasicek(r0, K, theta, sigma, T=1., N=10, seed=777):    
#     np.random.seed(seed)
#     dt = T/float(N)    
#     rates = [r0]
#     for i in range(N):
#         dr = K*(theta-rates[-1])*dt + sigma*np.random.normal()
#         rates.append(rates[-1] + dr)
#     return range(N+1), rates

# if __name__ == "__main__":
#     x, y = vasicek(0.01875, 0.20, 0.01, 0.012, 10., 200)

#     import matplotlib.pyplot as plt
#     plt.plot(x,y)
#     plt.show()
#  18 changes: 18 additions & 0 deletions18  
# B03898_05_Codes/zero_coupon_bond.py
# @@ -0,0 +1,18 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# def zero_coupon_bond(par, y, t):
#     """
#     Price a zero coupon bond.
    
#     Par - face value of the bond.
#     y - annual yield or rate of the bond.
#     t - time to maturity in years.
#     """
#     return par/(1+y)**t

# print zero_coupon_bond(100, 0.05, 5)
#  1,046 changes: 1,046 additions & 0 deletions1,046  
# B03898_06_codes/Chapter 6 Notebook.ipynb
# Large diffs are not rendered by default.

#  4 changes: 4 additions & 0 deletions4  
# B03898_06_codes/README
# @@ -0,0 +1,4 @@
# README
# ======
# Run "Chapter 6 Notebook.ipynb" on IPython Notebook.
# ======
#  9 changes: 9 additions & 0 deletions9  
# B03898_07_Codes/B03898_07_01.txt
# @@ -0,0 +1,9 @@
# README
# ======
# This file contains Terminal commands.
# ======

# hadoop fs -copyFromLocal /home/cloudera/Downloads/pg4300.txt pg4300.txt 

# hadoop fs -ls  

#  14 changes: 14 additions & 0 deletions14  
# B03898_07_Codes/B03898_07_02.py
# @@ -0,0 +1,14 @@
# """
# README
# ======
# This file contains Python codes.
# Save this file as mapper.py.
# ======
# """

# #!/usr/bin/python
# import sys

# for line in sys.stdin:
#     for word in line.strip().split():
#         print "%s\t%d" % (word, 1)
#  6 changes: 6 additions & 0 deletions6  
# B03898_07_Codes/B03898_07_03.txt
# @@ -0,0 +1,6 @@
# README
# ======
# This file contains Terminal commands.
# ======

# chmod +x /home/cloudera/word_count/mapper.py
#  1 change: 1 addition & 0 deletions1  
# B03898_07_Codes/B03898_07_04.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.Save this file as reduce.py.======"""#!/usr/bin/pythonimport sys current_word = Nonecurrent_count = 1 for line in sys.stdin:    word, count = line.strip().split('\t')    if current_word:        if word == current_word:            current_count += int(count)        else:            print "%s\t%d" % (current_word, current_count)            current_count = 1     current_word = word if current_count > 1:    print "%s\t%d" % (current_word, current_count)
#  6 changes: 6 additions & 0 deletions6  
# B03898_07_Codes/B03898_07_05.txt
# @@ -0,0 +1,6 @@
# README
# ======
# This file contains Terminal commands.
# ======

# chmod +x /home/cloudera/word_count/reduce.py
#  23 changes: 23 additions & 0 deletions23  
# B03898_07_Codes/B03898_07_06.txt
# @@ -0,0 +1,23 @@
# README
# ======
# This file contains Terminal commands.
# ======

# echo "foo foo quux labs foo bar quux" | /home/cloudera/word_count/mapper.py 

# echo "foo foo quux labs foo bar quux" | /home/cloudera/word_count/mapper.py | sort -k1,1 | /home/cloudera/word_count/reduce.py 

# hadoop jar \
# /usr/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-2.5.0-mr1-cdh5.3.0.jar \
# -file /home/cloudera/word_count/mapper.py \
# -mapper /home/cloudera/word_count/mapper.py \
# -file /home/cloudera/word_count/reduce.py \
# -reducer /home/cloudera/word_count/reduce.py \
# -input pg4300.txt \
# -output pg4300-output

# hadoop fs -ls 

# hadoop fs -ls pg4300-output 

# hadoop fs -cat pg4300-output/part-00000 
#  8 changes: 8 additions & 0 deletions8  
# B03898_07_Codes/B03898_07_07.txt
# @@ -0,0 +1,8 @@
# README
# ======
# This file contains Terminal commands.
# ======

# hadoop fs -copyFromLocal /home/cloudera/Downloads/ibm.csv ibm.csv 

# hadoop fs -ls 
#  1 change: 1 addition & 0 deletions1  
# B03898_07_Codes/B03898_07_08.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.Save this file as mapper.py.in the directory:/home/cloudera/stock/======"""#!/usr/bin/python import sys   is_first_line = True for line in sys.stdin: 	   if is_first_line: 		      is_first_line = False 		      continue  	   row =  line.split(',') 	   open_price = float(row[1]) 	   close_price = float(row[-3]) 	   change = (open_price-close_price)/open_price * 100 	   change_text = str(round(change,1)) + "%" 	   print "%s\t%d" % (change_text, 1)
#  1 change: 1 addition & 0 deletions1  
# B03898_07_Codes/B03898_07_09.txt
# @@ -0,0 +1 @@
# README======This file contains Terminal commands.======chmod +x /home/cloudera/stock/mapper.pychmod +x /home/cloudera/stock/reduce.pyhadoop jar \/usr/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-2.5.0-mr1-cdh5.3.0.jar \-file /home/cloudera/stock/mapper.py \-mapper /home/cloudera/stock/mapper.py \-file /home/cloudera/stock/reduce.py \-reducer /home/cloudera/stock/reduce.py \-input ibm.csv \-output stock-outputhadoop fs -copyToLocal  stock-output/part-00000 /home/cloudera/stock/ 
#  6 changes: 6 additions & 0 deletions6  
# B03898_07_Codes/B03898_07_10.txt
# @@ -0,0 +1,6 @@
# README
# ======
# This file contains Terminal commands.
# ======

# sudo yum install python-matplotlib
#  1 change: 1 addition & 0 deletions1  
# B03898_07_Codes/B03898_07_11.py
# @@ -0,0 +1 @@
# """"README======This file contains Python codes.Save this file as analysis.py.in the directory:/home/cloudera/stock/======"""import matplotlib.pyplot as plt  with open('/home/cloudera/stock/part-00000', 'rb') as f:    x, y = [], []    for line in f.readlines():      data = line.split() 		      x.append(float(data[0].strip('%'))) 		      y.append(float(data[1]))    print "Max:", max(x)    print "Min:", min(x)    plt.bar(x, y, width=0.1)    plt.show()
#  6 changes: 6 additions & 0 deletions6  
# B03898_07_Codes/B03898_07_12.txt
# @@ -0,0 +1,6 @@
# README
# ======
# This file contains Terminal commands.
# ======

# python /home/cloudera/stock/analysis.py
#  8 changes: 8 additions & 0 deletions8  
# B03898_07_Codes/B03898_07_13.txt
# @@ -0,0 +1,8 @@
# README
# ======
# This file contains Terminal commands.
# ======

# mkdir -p data/db

# mongod –dbpath data/db
#  1 change: 1 addition & 0 deletions1  
# B03898_07_Codes/B03898_07_14.py
# @@ -0,0 +1 @@
# """README======This file contains Python codes.======"""import pymongotry:    client = pymongo.MongoClient("localhost", 27017)    print "Connected successfully!!!"except pymongo.errors.ConnectionFailure, e:   print "Could not connect to MongoDB: %s" % e
#  68 changes: 68 additions & 0 deletions68  
# B03898_07_Codes/B03898_07_15.py
# @@ -0,0 +1,68 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# import datetime as dt
# import pymongo

# client = pymongo.MongoClient("localhost", 27017)
# ticks_db = client.ticks_db
# aapl_collection = ticks_db.aapl

# tick = {"ticker": "aapl",
#         "time": dt.datetime(2014, 11, 17, 10, 0, 0),
#         "open": 115.58,
#         "high": 116.08,
#         "low": 114.49,
#         "last": 114.96,
#         "vol": 1900000}

# tick_id = aapl_collection.insert(tick)
# print tick_id
# print ticks_db.collection_names()

# print aapl_collection.find_one()
# print aapl_collection.find_one({"time": dt.datetime(2014, 11, 17, 10, 0, 0)})

# from bson.objectid import ObjectId
# print aapl_collection.find_one({"_id": \
# 	ObjectId("548490486d3ba7178b6c36ba")})

# aapl_collection.remove()

# aapl_collection.insert([tick,
#                        {"ticker": "aapl",
#                         "time": dt.datetime(2014, 11, 17, 10, 1, 0),
#                         "open": 115.58,
#                         "high": 116.08,
#                         "low": 114.49,
#                         "last": 115.00,
#                         "vol": 2000000},
#                        {"ticker": "aapl",
#                         "time": dt.datetime(2014, 11, 17, 10, 2, 0),
#                         "open": 115.58,
#                         "high": 116.08,
#                         "low": 113.49,
#                         "last": 115.00,
#                         "vol": 2100000}])

# print aapl_collection.count()
# print aapl_collection.find({"open": 115.58}).count()

# for aapl_tick in aapl_collection.find():
#    print aapl_tick

# cutoff_time = dt.datetime(2014, 11, 17, 10, 2, 0)
# for tick in aapl_collection.find(
#        {"time": {"$lt": cutoff_time}}).sort("time"):
#    print tick 

# sorted_ticks = aapl_collection.find().sort(
#     [("time", pymongo.DESCENDING)])
# for tick in sorted_ticks:
#     print tick 


#  203 changes: 203 additions & 0 deletions203  
# B03898_08_codes/AlgoSystem.py
# @@ -0,0 +1,203 @@
# """
# README
# ======
# This file contains Python codes.
# ======
# """

# """ Implementing the Mean-Reverting Algorithm """
# from ib.ext.Contract import Contract
# from ib.ext.Order import Order
# from ib.opt import Connection, message
# import time
# import pandas as pd
# import datetime as dt


# class AlgoSystem:
#     def __init__(self, symbol, qty, resample_interval,
#                  averaging_period=5, port=7496):
#         self.client_id = 1
#         self.order_id = 1
#         self.qty = qty
#         self.symbol_id, self.symbol = 0, symbol
#         self.resample_interval = resample_interval
#         self.averaging_period = averaging_period
#         self.port = port
#         self.tws_conn = None
#         self.bid_price, self.ask_price = 0, 0
#         self.last_prices = pd.DataFrame(columns=[self.symbol_id])
#         self.average_price = 0
#         self.is_position_opened = False
#         self.account_code = None
#         self.unrealized_pnl, self.realized_pnl = 0, 0
#         self.position = 0

#     def error_handler(self, msg):
#         if msg.typeName == "error" and msg.id != -1:
#             print "Server Error:", msg

#     def server_handler(self, msg):
#         if msg.typeName == "nextValidId":
#             self.order_id = msg.orderId
#         elif msg.typeName == "managedAccounts":
#             self.account_code = msg.accountsList
#         elif msg.typeName == "updatePortfolio" \
#                 and msg.contract.m_symbol == self.symbol:
#             self.unrealized_pnl = msg.unrealizedPNL
#             self.realized_pnl = msg.realizedPNL
#             self.position = msg.position
#         elif msg.typeName == "error" and msg.id != -1:
#             return

#     def tick_event(self, msg):
#         if msg.field == 1:
#             self.bid_price = msg.price
#         elif msg.field == 2:
#             self.ask_price = msg.price
#         elif msg.field == 4:
#             self.last_prices.loc[dt.datetime.now()] = msg.price
#             resampled_prices = \
#                 self.last_prices.resample(self.resample_interval,
#                                           how='last',
#                                           fill_method="ffill")
#             self.average_price = resampled_prices.tail(
#                 self.averaging_period).mean()[0]