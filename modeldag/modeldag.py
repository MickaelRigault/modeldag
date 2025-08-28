import numpy as np
import pandas
import inspect
import warnings

#__all__ = ["ModelDAG"]


def draw_ndpdf(xx, ndpdf, size=1):
    """ draw n-parameters for joint multivariate pdf
    
    Parameters
    ----------
    xx: array
         array of definition for the pdf. format: (nx, ny, ndim)
        
    ndpdf: array
        pdf of the multivariates. format: (ntargets, nx, ny)
                 
    Returns
    -------
    array 
        format: (ndim, ntargets)
    """
    if len(xx.shape) != 3:
        raise NotImplementedError(f"Only xx.shape (n, m, k) implemented. len(shape) input: {len(xx.shape)} ")

    xxshape = np.prod(xx.shape[:-1])
    xx_index = np.arange(xxshape)
    xx_flatten = xx.reshape(xxshape, xx.shape[-1])
    # shape of ndpdf: (ntarget, *xx.shape[:-1])
    ndpdf_flatten = ndpdf.reshape(ndpdf.shape[0], np.prod(xx.shape[:-1]))
    drawn_indexes = [np.random.choice(xx_index, size=size, p=pdf_/pdf_.sum())
                     for pdf_ in ndpdf_flatten]
    return xx_flatten[np.hstack(drawn_indexes)].T


class ModelDAG( object ):
    """
    Models are dict of arguments that may have 3 entries:
    model = {key1 : {'func': func, 'kwargs': dict, 'as': None_str_list'},
             key2 : {'func': func, 'kwargs': dict, 'as': None_str_list'},
             ...
             }
    
    """
    def __init__(self, model={}, obj=None, as_conflict="raise"):
        """ 
        
        Parameters
        ----------
        model: dict
            dictionary descripting the model DAG

        obj: object
            instance the model is attached too. It may contain 
            method called by the DAG.

        as_conflict: string
            how the code should behave in case of naming conflict:
            - raise: raises a ValueError
            - warn: do a warning (warnings.warn) and skip.
            - skip: do nothing initial dict, so original key part will be overwritten.

        Returns
        -------
        instance
        """
        self.set_model( model, as_conflict=as_conflict)
        self.obj = obj

    def __str__(self):
        """ """
        import pprint
        return pprint.pformat(self.model, sort_dicts=False)

    def __repr__(self):
        """ """
        return self.__str__()
    
    def to_graph(self, engine="networkx", incl_param=False):
        """ converts the model into another graph library object 

        Parameters
        ----------
        engine: string
            Implemented:
            - NetworkX (networkx.org/documentation/stable/tutorial.html)
            - Graphviz (https://pygraphviz.github.io/documentation/stable/index.html)

        Return
        ------
        Graph instance
            a new instance object.
        """
        if engine == "networkx":
            import networkx as nx
            graph = nx.Graph()
        elif engine in ["graphviz", "pygraphviz"]:
            import pygraphviz as pgv
            graph = pgv.AGraph(directed=True, strict=True)
        else:
            raise NotImplementedError(f"engine {engine} is not implemented. networkx and graphviz are")

        # Nodes and Edges
        for name in self.entries:
            graph.add_node(name)
    
        for name, to_name in self.entry_inputof.items():
            graph.add_edge(name, to_name)

        if incl_param:
            model_df = self.get_modeldf()["kwargs"]
            kwargs_df = model_df.groupby(level=0).first() # remove exploded duplicates
            default_kwargs = self.get_func_parameters(default=True)

            # loop over nodes (model parameters)
            for name, input_kwargs in kwargs_df.items():
                if type(input_kwargs) != dict:
                    input_kwargs = {} # fix Nans, Nones etc.
                    
                kwargs = default_kwargs.get(name, {}) | input_kwargs

                param_text = ""
                for k, v in kwargs.items():
                    # ignore if internal connection.
                    if "@" in str(v): 
                        continue
                    
                    # If not, create an node and edge
                    param_text += f"{k} = {v}"+"\n"

                if param_text == "":
                    continue # no entry
                    
                new_node = f"{name} input"
                graph.add_edge(new_node, name)
                new_edge = graph.get_edge(new_node, name)
                # and make it look special.
                # node
                node = graph.get_node(new_node)
                node.attr["fontsize"] = 10
                node.attr["label"] = param_text
                node.attr["fontcolor"] = "#ADADAD"
                node.attr["shape"] = "plaintext"
                # edge
                new_edge.attr["color"] = "#ADADAD"
                new_edge.attr["arrowhead"] = "none"
                #arrowsize

        return graph
    
    def to_networkx(self, incl_param=False):
        """ shortcut to to_graph('networkx') """
        return self.to_graph(engine="networkx", incl_param=incl_param)

    def to_graphviz(self, incl_param=False):
        """ shortcut to to_graph('graphviz') """
        return self.to_graph(engine="graphviz", incl_param=incl_param)

    
    # ============ #
    #   Method     #
    # ============ #
    def set_model(self, model, as_conflict="raise"):
        """ sets the model to the instance (inplace) applying basic validation. """
        from .tools import _get_valid_model_
        self._model = _get_valid_model_(model, as_conflict=as_conflict)
    

    def visualize(self, fileout="tmp_modelvisualize.svg", incl_param=False):
        """ """
        from IPython.display import SVG
        
        ag = self.to_graphviz(incl_param=incl_param)
        ag.graph_attr["epsilon"] = "0.001"
        
        ag.layout("dot")  # layout with dot
        ag.draw(fileout)
        return SVG(fileout)

    def get_model(self, prior_inputs=None, missing_entries="raise",
                      seed="allowed",
                      **kwargs):
        """ get a copy of the model 
        
        Parameters
        ----------
        **kwargs can change the model entry parameters
            for instance, t0: {"low":0, "high":10}
            will update model["t0"]["kwargs"] = ...

        Returns
        -------
        dict
           a copy of the model (with param potentially updated)
        """
        from .tools import make_model_direct, get_modelcopy
        
        model = get_modelcopy(self.model)
        for k,v in kwargs.items():
            model[k]["kwargs"] = model[k].get("kwargs",{}) | v

        direct_model = make_model_direct( model,
                                          missing_entries=missing_entries,
                                          prior_inputs=prior_inputs)
                
        return direct_model
    
    def change_model(self, **kwargs):
        """ change the model attached to this instance
        
        **kwargs will update the entry  parameters ("param", e.g. t0["kwargs"])

        See also
        --------
        get_model: get a copy of the model
        """
        self.model = self.get_model(**kwargs)

    def get_func_parameters(self, default=False, incl_args=True, fillargs=np.nan):
        """ get a dictionary with the parameters name of all model functions
        
        Parameters
        ----------
        default: bool
            get the default model parameter without the model updates.

        Returns
        -------
        dict
        """
        import inspect
        inspected = {}
        for k, m in self.model.items():
            func = self._parse_input_func(name=k, func=m.get("func", None))
            try:
                # ::-1 since kwargs are after args
                params = inspect.getfullargspec( func ).args[::-1]
                values = inspect.getfullargspec( func ).defaults[::-1]
                if incl_args:
                    values_full = [fillargs]*len(params)
                    values_full[:len(values)] = values
                    kwargs_ = dict( zip(params, values_full) )
                else:
                    kwargs_ = dict( zip(params, values) )
            except:
                kwargs_ = {}
                
            inspected[k] = kwargs_

        # should we update the default func values ?
        if not default:
            current_model = self.get_model()
            manual_kwargs = {k: v.get("kwargs",{}) for k,v in current_model.items()}
            parameters = {k: inspected[k] | manual_kwargs[k] for k in current_model.keys() }
        else:
            parameters = inspected
        
        return parameters

    def get_func_with_args(self, argname):
        """ return list of parameter names that have {argname} in their options (args or kwargs) """
        dict_func = self.get_func_parameters(incl_args=True)
        return [k for k, v in dict_func.items() if argname in v]
    
    def get_backward_entries(self, name, incl_input=True):
        """ get the list of entries that affects the input on.
        Changing any of the return entry name impact the given name.

        Parameters
        ----------
        name: str
            name of the entry

        incl_input: bool
            should the returned list include or not 
            the given name ? 

        Return
        ------
        list
            list of backward entry names 
        """
        depends_on = self.entry_dependencies.dropna()

        names = np.atleast_1d(name)
        if incl_input:
            backward_entries = list(names)
        else:
            backward_entries = []

        leads_to_changing = depends_on.loc[depends_on.index.isin(names)]
        while len(leads_to_changing)>0:
            _ = [backward_entries.append(name_) for name_ in list(leads_to_changing.values)]
            leads_to_changing = depends_on.loc[depends_on.index.isin(leads_to_changing)]

        return backward_entries

    def get_forward_entries(self, name, incl_input=True):
        """ get the list of forward entries. 
        These would be affected if the given entry name is changed.

        Parameters
        ----------
        name: str
            name of the entry

        incl_input: bool
            should the returned list include or not 
            the given name ? 

        Return
        ------
        list
            list of forward entry names 
        """
        inputs_of = self.entry_inputof.explode()

        names = np.atleast_1d(name)
        if incl_input:
            forward_entries = list(names)
        else:
            forward_entries = []

        leads_to_changing = inputs_of.loc[inputs_of.index.isin(names)]
        while len(leads_to_changing)>0:
            # all names individually
            _ = [forward_entries.append(name_) for name_ in list(leads_to_changing.values)]
            leads_to_changing = inputs_of.loc[inputs_of.index.isin(leads_to_changing)]

        return forward_entries

    def get_modeldf(self, explode=True, model=None):
        """ get a pandas.DataFrame version of the model dict

        Parameters
        ----------
        explode: bool
            should the input entry be exploded ?
            
        model: dict 
            if given, the modeldf will be that of this model,
            otherwise it uses self.model.
        Returns
        -------
        pandas.DataFrame
        """
        from .tools import get_modeldf
        if model is None:
            model = self.model

        modeldf = get_modeldf(explode=explode, model=model)
        return modeldf
        
    # ============ #
    #  Drawers     #
    # ============ #
    def redraw_from(self, name, data, incl_name=True, size=None, **kwargs):
        """ re-draw the data starting from the given entry name.
        
        All forward values will be updated while the independent 
        entries are left unchanged.

        Parameters
        ----------
        name: str, list
            entry name or names. See self.entries

        data: pandas.DataFrame
            data to be updated 
            Must at least include entry needed by name if any. 
            See self.entry_dependencies

        incl_name: bool
            should the given name be updated or not ?

        size: None
            number of entries to draw. Ignored if not needed.

        **kwargs goes to self.draw() -> get_model

        Returns
        -------
        pandas.DataFrame
            the updated version of the input data.

        """
        if len(np.atleast_1d(name)) > 1: # several entries
            # do not include the input entry at fist
            name = list(np.atleast_1d(name)) # as list 
            limit_to_entries = [self.get_forward_entries(name_, incl_input=False) for name_ in name]
            limit_to_entries = list(np.unique(np.concatenate(limit_to_entries))) # unique
            if np.any([name_ in limit_to_entries for name_ in name]): # means some entry want to change another given
                raise ValueError("At least on of the input name have at least one other given as forward entry. This is not supported by this method.")
            
            if incl_name:
                limit_to_entries += name
        else:
            limit_to_entries = self.get_forward_entries(name, incl_input=incl_name)
            
        return self.draw(size, limit_to_entries=limit_to_entries, data=data)

    
    def draw(self, size=None, limit_to_entries=None, data=None,
                 allowed_legacyseed=True, **kwargs):
        """ draw a random sampling of the parameters following
        the model DAG

        Parameters
        ----------
        size: int
            number of elements you want to draw.

        limit_to_entries: list
            if given, entries not in this list will be ignored.
            see self.entries

        data: pandas.DataFrame
            starting point for the draw.

        **kwargs goes to get_model()

        Returns
        -------
        pandas.DataFrame
            N=size lines and at least 1 column per model entry
        """
        if data is not None:
            prior_inputs = data.columns
        else:
            prior_inputs = None
            
        model = self.get_model(prior_inputs=prior_inputs, **kwargs)

        return self._draw(model, size=size, limit_to_entries=limit_to_entries,
                              allowed_legacyseed=allowed_legacyseed,
                              data=data)
    
    def draw_param(self, name=None, func=None, size=None, xx=None, allowed_legacyseed=True, **kwargs):
        """ draw a single entry of the model

        Parameters
        ----------
        name: str
            name of the variable
            
        func: str, function
            what func should be used to draw the parameter

        size: int
            number of line to be draw

        xx: str, array
           provide this *if* the func returns the pdf and not sampling.
           xx defines where the pdf will be evaluated.
           if xx is a string, it will be assumed to be a np.r_ entry (``np.r_[xx]``)

        Returns
        -------
        list 
            
        """        
        # Flexible origin of the sampling method
        func = self._parse_input_func(name=name, func=func)
        if "MT19937" in str(func) and allowed_legacyseed:
            #state = np.random.get_state()
            np.random.seed()
            #np.random.set_state(state)
            #_ = np.random.seed() # empty seed, just need to set it

        
        # Check the function parameters
        try:
            func_arguments = list(inspect.getfullargspec(func).args)
        except: # fail for Cython functions
            #warnings.warn(f"inspect failed for {name}{func} -> {func}")
            func_arguments = ["size"] # let's assume this as for numpy.random or scipy.

        # -> save function that are purely
        actual_args = [l for l in func_arguments if l not in ["self"]]
        if len(actual_args) == 0 and size is not None: # pure *arg, **kwargs func. like scipy's rvs() | assume size
            func_arguments += ["size"] # add size.
            

                
        # And set the correct parameters that may be missing
        prop = {}
        if "size" in func_arguments:
            prop["size"] = size
            
        if "func" in func_arguments and func is not None: # means you left the default
            prop["func"] = func

        if "xx" in func_arguments and xx is not None: # if xx is None
            if type(xx) == str: # assumed r_ input
                xx = eval(f"np.r_[{xx}]")
                
            prop["xx"] = xx

        # Draw it.
        actual_prop = prop | kwargs
        draw_ = func( **actual_prop )
        if "xx" in func_arguments: # draw_ was actually a pdf
            xx_, pdf_ = draw_
            draw_ = self.draw_from_pdf(pdf_, xx_, size)
            
        return draw_
            
    @staticmethod
    def draw_from_pdf(pdf, xx, size):
        """ randomly draw from xx N=size elements following the pdf. """
        if type(xx) == str: # assumed r_ input
            xx = eval(f"np.r_[{xx}]")

        pdf = np.squeeze(pdf) # shape -> (1, size) -> (size,)
        
        if len( pdf.shape ) == 2:
            choices = np.hstack([np.random.choice(xx, size=1, p=pdf_/pdf_.sum())
                           for pdf_ in pdf])
        elif len( pdf.shape ) == 3: # assumed list of multivariate
            choices = draw_ndpdf(xx, pdf) # not size.
        else:
            choices = np.random.choice(xx, size=size, p=pdf/pdf.sum())

        return choices
    

    def _draw(self, model, size=None, limit_to_entries=None, data=None,
                  allowed_legacyseed=True):
        """ core method converting model into a DataFrame (interp) """
        if size == 0:
            columns = list(np.hstack([v.get("as", name) for name, v in model.items()]))
            return pandas.DataFrame(columns=columns)

        if data is None:
            data = pandas.DataFrame()
        else:
            data = data.copy() # make sure you are changing a copy

        #
        # The draw loop
        #
        for param_name, param_model in model.items():
            if limit_to_entries is not None and param_name not in limit_to_entries:
                continue

            params = dict(size=size) # starting point. This gets updated if @ arrives.
            # Default kwargs given
            if (inprop := param_model.get("kwargs", {})) is None:
                inprop = {}

            # parse the @**
            for k, v in inprop.items():
                if type(v) is str and "@" in v:
                    key = v.split("@")[-1].split(" ")[0]
                    inprop[k] = data[key].values
                    params["size"] = None
                    
            # set the model ; this overwrite prop['model'] but that make sense.
            inprop["func"] = param_model.get("func", None)
            
            # update the general properties for that of this parameters
            prop = {**params, **inprop}
            # 
            # Draw it
            samples = np.asarray(self.draw_param(param_name,
                                                 allowed_legacyseed=allowed_legacyseed,
                                                 **prop))

            # and feed
            output_name = param_model.get("as", param_name)
            if output_name is None: # solves 'as' = None case.
                output_name = param_name
            data[output_name] = samples.T

        return data
    
    def _parse_input_func(self, name=None, func=None):
        """ returns the function associated to the input func.

        """
        if callable(func): # func is a function. Good to go.
            return func
        
        # func is a method of this instance ?
        if func is not None and hasattr(self, func):
            func = getattr(self, func)
            
        # func is a method a given object ?
        elif func is not None and hasattr(self.obj, func):
            func = getattr(self.obj, func)
            
        # name is a draw_ method of this instance object ?            
        elif hasattr(self, f"draw_{name}"):
            func = getattr(self,f"draw_{name}")

        # name is a draw_ method of this instance object ?            
        elif hasattr(self.obj, f"draw_{name}"):
            func = getattr(self.obj, f"draw_{name}")
        
        else:
            try:
                func = eval(func) # if you input a string of a function known by python somehow ?
            except:
                raise ValueError(f"Cannot parse the input function {name}:{func}")
        
        return func

    # =================== #
    #   Properties        #
    # =================== #
    @property
    def model(self):
        """ the core element of the instance from which all are derived."""
        return self._model
    
    @property
    def entries(self):
        """ array of model entry names """
        modeldf = self.get_modeldf()
        return np.asarray(modeldf.index.unique(), dtype=str)

    @property
    def entry_dependencies(self):
        """ pandas series of entry input dependencies (exploded) | nan is not entry """
        modeldf = self.get_modeldf()
        return modeldf["input"]

    @property
    def entry_inputof(self):
        """ """
        # maybe a bit overcomplicated...
        modeldf = self.get_modeldf()
        return modeldf[~modeldf["input"].isna()].reset_index().set_index("input")["entry"]
        
