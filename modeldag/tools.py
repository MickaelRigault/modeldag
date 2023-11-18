import numpy as np

def apply_gaussian_noise(errmodel, data):
    """ apply gaussian noise to current entries.

    Parameters
    ----------
    errmodel: dict
        dict that will feed a ModelDAG. 
        format: {x: {func:, kwargs:{}}}. 
        this will draw x_err following the this formula and will update 
        x assuming x_true for the original x and x_err for the given x drawn here.
        you can refeer to the original x using '@x_true' in the func kwargs.

    data: None
        original dataframe to be noisified. If None self.data is used.

    Returns
    -------
    self or dataframe
        - self if data is None
        - dataframe otherwise.
    """
    from . import ModelDAG
    
    data_ = data.copy()
    size = len(data)
    
    key_to_change = list( errmodel.keys() )
    # store the input one as "{}_true"
    data_ = data_.rename({k:k+"_true" for k in key_to_change}, axis=1)

    # Build the actual modeldag we want: {k}_err and resulting {k} from {k}_true and {k}_err
    errmodel = {k+"_err":v for k,v in errmodel.items()}
    offsetmodel = {k: {"func": np.random.normal,
                       "kwargs":{"loc":f"@{k}_true", "scale":f"@{k}_err"}} 
                   for k in key_to_change}
    
    errormodel = ModelDAG({**errmodel, **offsetmodel})
    new_data = errormodel.draw(size=size, data=data_)
    return new_data
