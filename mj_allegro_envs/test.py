import numpy as np
def parse_eq(eq):
  lhs,rhs = eq.split('=')
  coeffs = {}
  coef = 0
  var = ''
  is_negative = False
  for x in lhs:
    if(x == ' '):
      continue
    elif(x == '-'):
      if(var !=''):
        if(coef == 0): coef = 1
        coeffs[var] = coef
        coef = 0
      is_negative = True
    elif(x.isnumeric()):
      coef = (10*coef) + int(x)
    elif(x.isalpha()):
      var = x
    elif(x == '+'):
      if(is_negative):
        coef = coef * -1
        is_negative = False
      if(coef == 0): coef = 1
      coeffs[var] = coef
      coef = 0
  if(is_negative):
      coef = coef * -1
      is_negative = False
  if(coef == 0): coef = 1
  coeffs[var] = coef

  y = int(rhs)
  return coeffs, y
  

def solve_system(equations):
  """"
  Takes in a list of strings for each equation. 
  Returns a numpy array with a row for each equation value
  """
  # return np.array([[0],[0],[0],[0]])
  next_is_neg = False
  mat_A = None
  mat_y = None
  for e in equations:
    coeffs,y = parse_eq(e)
    row = np.zeros(4)
    i = 0
    for x in ['a','b','c','d']:
      if(x in coeffs):
        row[i] = coeffs[x]
      i+=1
    if(mat_A is None):
    	mat_A = row
    	mat_y = np.array([y])
    else:
    	mat_A = np.concatenate((mat_A,row),0)
    	mat_y = np.concatenate((mat_y,np.array([y])))
  mat_A = mat_A.reshape((len(equations),4))
  mat_y = mat_y.reshape((len(equations),1))
  inv_A = np.linalg.inv(mat_A)
  return np.matmul(inv_A, mat_y)

def test_eq():
  sys_eq = ['2 a + b - 3 c + d = 9',
            '-5 a + 1 b - 4 c + d = -14',
            'a + 2 b - 10 c = -7',
            'a + 2 b = 13']
  results = solve_system(sys_eq)
  expected = np.array([[3],[5],[2],[4]])

  assert(np.all(abs(expected-results) < 1e-10 ))

test_eq()