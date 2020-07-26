import numpy as np
import pandas as pd
import unittest
import sys


class IrisQuality(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
              header=None, names = ['sepal length cm', 'sepal width cm', 
                                    'petal length cm','petal width cm', 'class'], 
                  index_col=False
                  )

    # Defining the columns as target and features
    cls.target='class'
    cls.features=['sepal length cm', 'sepal width cm', 'petal length cm','petal width cm']

  def test_data_completeness(self):
    # Check on the numbers of records and attributes
      self.assertEqual(self.df.shape[0], 150)
      self.assertEqual(self.df.shape[1], 5)

  def test_missing_data(self):
    # Check for missing/empty records from the data source
      self.assertEqual(self.df.isna().any(axis=None), False)

  def test_duplication(self):
    # Check for number of duplicates within in the data source
    # on prior investigation, it is found that there are 5 duplicate records in the data
      self.assertEqual(len(self.df.drop_duplicates(keep=False))+5, len(self.df))

  def test_positive(self):
    # Check that all recorded attributes (width and length) is positive.
      for i in self.features:
        self.assertEqual(all(i >= 0 for i in self.df[i]), True)



def main(out = sys.stderr, verbosity = 2): 
    loader = unittest.TestLoader() 
  
    suite = loader.loadTestsFromModule(sys.modules[__name__]) 
    unittest.TextTestRunner(out, verbosity = verbosity).run(suite) 
      
if __name__ == '__main__': 
    with open('data_quality.output', 'w') as f: 
        main(f) 

# if __name__ == '__main__':
#     unittest.main(verbosity=2)