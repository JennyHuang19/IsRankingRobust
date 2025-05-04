from RankAMIP.logistic import make_BT_design_matrix
import numpy as np
import pandas as pd

df = pd.DataFrame({'Player1': ['Alice', 'Bob', 'Alice'], 
                   'Player2': ['Bob', 'Charlie', 'David'],
                   'a_winning':[0,1,0]
                   })

X_expected = np.array([[-1,0,0], [1,-1,0], [0,0,-1]])
y_expected = np.array([0,1,0])

X, y, encoding = make_BT_design_matrix(df)
assert np.all(X == X_expected) and np.all(y==y_expected)
breakpoint()
