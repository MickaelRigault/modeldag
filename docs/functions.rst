Functions
======


In modeldag_, there are 4 different funtional forms that are accepted for ``func``, and different function locations for:

.. code-block:: python
		
    model = {'a': {'func': func, 'kwargs': dict, 'as': None_str_list'},
             'b': {'func': func, 'kwargs': dict, 'as': None_str_list'}
             }

For each of them, ``ModelDag.draw()`` will know how to fill the dataframe columns.

Functional form
--------------

.. grid:: 4
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Random variate
      :columns: 12 6 6 3
      :class-card: sd-border-0
      :shadow: None

      these ``func`` accept ``size`` as input and return a 1d-array of size ``size``.
      The returned array is the random variable you drew. 

   .. grid-item-card:: Transformation
      :columns: 12 6 6 3
      :class-card: sd-border-0
      :shadow: None

      these ``func`` takes a variable as input and returns a same-size variable. modeldag_ has no drawing to do, just accept the new function as input. 

      
   .. grid-item-card:: 1d-pdf
      :columns: 12 6 6 3
      :class-card: sd-border-0
      :shadow: None

      these ``func`` accept ``xx`` but not ``size`` and tt returns two variable: ``xx`` *shape (M)* and ``pdf`` *shape (M,)*.
      modeldag_ will draw``N=size`` variables from ``xx`` using ``pdf`` as weight.
      

   .. grid-item-card:: 2d-pdf
      :columns: 12 6 6 3
      :class-card: sd-border-0
      :shadow: None

      these ``func`` accept ``xx`` but not ``size`` and tt returns two variable: ``xx`` *shape (M)* and ``pdf`` *shape (size, M)*.
      modeldag_ will draw ``N=1`` variable from ``xx`` for each of ``pdf`` entries, making this ``N=size`` variable.


**Here are some examples**
      
.. tab-set::

    .. tab-item:: Random variate

       .. code-block:: python

            import modeldag
            import numpy as np	       

            # your own function (here a random draw)	    
            def func_rvs(size, coef=1, low=0, high=np.pi):
                """ the quadratic sum between value*scale and floor """
                values = np.random.uniform(size=size)
                return np.cos(values)

            # model construction | a and b are independent
            model = {"a": {"func": np.random.uniform,
                          "kwargs": {"low":0, "high":1}
                          },
                    "b": {"func": func_rvs,
                          "kwargs": {}}
                    }

            # create the DAG
            dag = modeldag.ModelDAG(model)
            data = dag.draw(1_000)
            _ = data.plot.scatter("a","b", s=1)

       .. image:: ./gallery/func_random.png
		  
    .. tab-item:: Transformation

       .. code-block:: python

            import modeldag
            import numpy as np	       

            # your own function (here a deterministic transformation)
            def func_transformation(value, scale=0.05, floor=0.2):
                """ the quadratic sum between value*scale and floor """
                return np.sqrt( (value*scale)**2 + floor**2)

            # model construction | b predictibly depends on a
            model = {"a": {"func": np.random.uniform,
                          "kwargs": {"low":0, "high":1}
                          },
                    "b": {"func": func_transformation,
                          "kwargs": {"value":"@a"}}
                    }

            # create the DAG		    
            dag = modeldag.ModelDAG(model)
            data = dag.draw(1_000)
            _ = data.plot.scatter("a","b", s=1)

       .. image:: ./gallery/func_transformation.png
		  
    .. tab-item:: 1d-pdf

       .. code-block:: python

            import modeldag
            import numpy as np
            from scipy import stats     

            # your own function (here a 1d PDF)	    
            def func_1dpdf(mean=2, scale=2, xx="-5:10:0.05"):
                """ a PDF with parameters that depends on an input variable-array 

                model: $pdf_2d = N(mean*power, scale)$

                Returns
                -------
                list
                    - xx: shape M
                    - pdf: shape (size==len(mean), M)
                """
                xx = eval(f"np.r_[{xx}]")
                pdf_ = stats.norm.pdf(xx, loc=mean, scale=scale)
                return xx, pdf_ # shapes: M, (M,)

            # model construction | a and b are independent
            model = {"a": {"func": np.random.uniform,
                          "kwargs": {"low":0, "high":1}
                          },
                    "b": {"func": func_1dpdf,
                          "kwargs": {"mean":2}
                         }
                    }

            # create the DAG		    
            dag = modeldag.ModelDAG(model)
            data = dag.draw(1_000)
            _ = data.plot.scatter("a","b", s=1)
	    
       .. image:: ./gallery/func_1dpdf.png
		  
    .. tab-item:: 2d-pdf

       .. code-block:: python

            import modeldag
            import numpy as np
            from scipy import stats     

            # your own function (here a 2d PDF)	     
            def func_2dpdf(mean, power=1, scale=2, xx="-5:10:0.05"):
                """ a PDF with parameters that depends on an input variable-array 

                model: $pdf_2d = N(mean*power, scale)$

                Returns
                -------
                list
                    - xx: shape M
                    - pdf: shape (size==len(mean), M)
                """
                xx = eval(f"np.r_[{xx}]")
                mean = np.atleast_2d(mean).T
                pdf_ = stats.norm.pdf(xx, loc=mean*power, scale=scale)
                return xx, pdf_ # shapes: M, (len(mean), M)

            # model construction | b depends on a		
            model = {"a": {"func": np.random.uniform,
                          "kwargs": {"low":0, "high":1}
                          },
                    "b": {"func": func_2dpdf,
                          "kwargs": {"mean":"@a", "power":10}
			  }
                    }

            # create the DAG		    
            dag = modeldag.ModelDAG(model)
            data = dag.draw(1_000)
            _ = data.plot.scatter("a","b", s=1)
	    
       .. image:: ./gallery/func_2dpdf.png

Function locations
---------------

``ModelDag`` accepts `obj=` as input. This way you can define use any method from your input object as a model ``func``. If so, pass it a a `string` in the model definition. The `string` format can also be used for any function from the `namespace`. Otherwise, simply directly provide the function itself. All with work the same. Remark that the namespace is checked prior the object.

.. tab-set::

    .. tab-item:: function

       .. code-block:: python

            import modeldag
            import numpy as np

            # your function
            def foo(value, scale=0.03, floor=0.2):
                return np.sqrt( (value*scale)**2 + floor**2)

            # model construction | foo already defined
            model = {"a": {"func": np.random.uniform,
                          "kwargs": {"low":0, "high":1}
                          },
                    "b": {"func": foo, 
                          "kwargs": {"value":"@a"}
                          }
                    }

            # create the DAG
            dag = modeldag.ModelDAG(model)
            data = dag.draw(1_000)
            _ = data.plot.scatter("a","b", s=1)   
   
    .. tab-item:: string

       .. code-block:: python

            import modeldag
            import numpy as np

            # model construction | string for any func in your namespace
            model = {"a": {"func": np.random.uniform,
                          "kwargs": {"low":0, "high":1}
                          },
                    "b": {"func": "np.random.uniform", # string
                          "kwargs": {"low":0, "high":1}
                          }
                    }

            # create the DAG
            dag = modeldag.ModelDAG(model)
            data = dag.draw(1_000)
            _ = data.plot.scatter("a","b", s=1)

    .. tab-item:: class method

       .. code-block:: python

            import modeldag
            import numpy as np

            # Define a class
            class MyClass():

                def __init__(self, floor=0.2):
                    """ """
                    self.floor = floor

                def foo_from_class(self, value, scale=0.03):
                    """ """
                    return np.sqrt( (value*scale)**2 + self.floor**2)

            # model construction | string for any method in the input object
            model = {"a": {"func": np.random.uniform,
                          "kwargs": {"low":0, "high":1}
                          },
                    "b": {"func": "foo_from_class", # as string
                          "kwargs": {"value":"@a"}
                          }
                    }

            # create the DAG, while inputing a class or class instance.
            dag = modeldag.ModelDAG(model, obj=MyClass(floor=0.2) )
            data = dag.draw(1_000)
            _ = data.plot.scatter("a","b", s=1)

	    

.. _modeldag: https://github.com/MickaelRigault/modeldag
