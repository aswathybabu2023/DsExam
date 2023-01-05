import numpy as np
x=np.array([[1,3],[2,1]])
y=np.array([[2,3],[1,0]])
z=np.array([[1,4],[3,1]])
print("matrix1:\n",x)
print("matrix2:\n",y)
print("matrix3:\n",z)
a=np.square(x)

b=np.multiply(2,y)

c=np.multiply(z,z,z)

op1=np.add(a,b)

op2=np.subtract(op1,c)
print("Result after the operation 'x^2+2y-z^3' :\n",op2)
