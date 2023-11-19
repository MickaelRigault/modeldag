import pandas
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

    # Build the actual modeldag: {k}_err and resulting {k} from {k}_true and {k}_err
    errmodel = {k+"_err":v for k,v in errmodel.items()}
    offsetmodel = {k: {"func": np.random.normal,
                       "kwargs":{"loc":f"@{k}_true", "scale":f"@{k}_err"}} 
                   for k in key_to_change}
    
    errormodel = ModelDAG({**errmodel, **offsetmodel})
    new_data = errormodel.draw(size=size, data=data_)
    return new_data

# =================== #
#  model manipulation #
# =================== #
def get_modelcopy(model):
    """ get a 'deep' copy of the model (dict) without using copy.deepmodel """
    import copy
    return {k:copy.copy(v) if type(v) != dict else \
                  {k_:copy.copy(v_) for k_,v_ in v.copy().items()}
           for k,v in model.copy().items()}

def get_modeldf(model, explode=True):
    """ convert the model (dict) into a pandas.DataFrame.
    This dataframe contains as, entry, input and ndeps.

    Parameters
    ----------
    model: dict
        dictionary that builds the model.
        - {entry: {func:{}, kwargs:{}, as:{}}, }
    
    explode: bool
        Should this explode the dataframe for the inputs (list)
    
    Returns
    -------
    pandas.DataFrame
    """
    modeldf = pandas.DataFrame( list(model.values()), 
                                index=model.keys()
                              ).reset_index(names=["model_name"])
    # naming convention    
    if "as" not in modeldf:
        modeldf["as"] = modeldf["model_name"]
        modeldf["entry"] = modeldf["as"]
    else:
        f_ = modeldf["as"].fillna( dict(modeldf["model_name"]) )
        f_.name = "entry"
        # merge and explode the names and inputs
        modeldf = modeldf.join(f_) 

    # now the entry connections
    modeldf["input"] = modeldf["kwargs"].apply(lambda x: [] if type(x) is not dict else \
                                                   [l.split("@")[-1].split(" ")[0]
                                                    for l in x.values() if type(l) is str \
                                                    and "@" in l]
                                              )
    # number of internal dependencies
    modeldf["ndeps"] = modeldf["input"].apply(len)
    
    if not explode:
        return modeldf.explode("entry").set_index("entry")
    
    return modeldf.explode("entry").explode("input").set_index("entry")

def make_model_direct(model, missing_entries="raise"):
    """ re-organise the input dictionary input a direct graph model 

    Parameters
    ----------
    model: dict
        dictionary that builds the model.
        - {entry: {func:{}, kwargs:{}, as:{}}, }
        
    missing_entries: str
        How should this behave if there are left-entries not buildable from DAG.
        - 'raise': ValueError
        - 'warn': warnings
        - othersiwe: ValueError

    Returns
    -------
    dict
        new ordered model (same format but re-ordered)
    """
    
    df = get_modeldf(model, explode=False) # get the dataframe for manipulation
    entries = list(df.index)
    
    roots = list(df[df["ndeps"]==0].index.astype(str)) # those depending on nothing
    used_entries = roots.copy()
    next_entries = ["bla"] # will go away
    # built the entries from their dependencies
    while len(next_entries) > 0:
        left_entries = [k for k in entries if k not in used_entries]
        next_entries = df.loc[left_entries]["input"].apply(lambda x: np.in1d(x, used_entries).all())
        next_entries = list(next_entries[next_entries].index.astype(str))
        used_entries = used_entries + next_entries

    # Test if the graph ok.
    not_used_entries = [k for k in entries if k not in used_entries]
    if len(not_used_entries) >0:
        message = f"input model do not form a DAG. These entries {missing_entries} cannot make a 'direct' graph"
        if missing_entries == "raise":
            raise ValueError(message)
        if missing_entries == "warn":
            import warnings
            warnings.warn(missing_entries)
        else:
            warnings.warn(f"missing_entries input cannot be parsed (missing_entries)")
            raise ValueError(message)
        
    # get the sorted model
    sorted_inname = df.loc[used_entries]["model_name"].unique()
    direct_model = {k:model[k] for k in sorted_inname}
    return direct_model
