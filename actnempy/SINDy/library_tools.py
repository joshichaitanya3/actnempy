'''
Functions and Classes for creating library of terms
'''
import itertools
import numpy as np
import copy

class Function:
    def __init__(self, name, root=None, maxf=100, maxd=100):
        self._name = name
        self.diff_order = 0
        self.func_order = 1
        self.max_func_order = maxf
        self.max_diff_order = maxd
        if root is None:
            self.root = self._name
        else:
            self.root = root
    @staticmethod
    def unity():
        f = Function('1')
        f.func_order = 0
        f.diff_order = 0
        f.max_func_order = 0
        f.max_diff_order = 0
        f.root = '1'
        return f

    def __repr__(self):
        return self._name
    
    def __mul__(self, rhs):
        if self.func_order == 0:
            return rhs
        if rhs.func_order == 0:
            return self
        else:
            return MultiplyOp(self, rhs)


class MultiplyOp:
    def __init__(self, lhs, rhs):
        self._lhs = lhs
        self._rhs = rhs
        self.diff_order = lhs.diff_order + rhs.diff_order
        self.func_order = lhs.func_order + rhs.func_order

    def __repr__(self):
        return f"{self._lhs} \u00D7 {self._rhs}"

    def __mul__(self, rhs):
        if self.func_order == 0:
            return rhs
        if rhs.func_order == 0:
            return self
        else:
            return MultiplyOp(self, rhs)


class Derivative:
    def __init__(self, f, x, n=1):

        self._x = {x: n}
        self._n = n
        if isinstance(f, Derivative):
            self._f = f._f
            self._n += f._n
            for k, v in f._x.items():
                if k in self._x:
                    self._x[k] += v
                else:
                    self._x[k] = v
        else:
            self._f = f
        self.diff_order = self._n
        self.func_order = self._f.func_order
        self.root = f.root
        self.max_func_order = f.max_func_order
        self.max_diff_order = f.max_diff_order
    def __repr__(self):
        superscripts = {
            0: '',
            1: '',
            2: '\u00B2',
            3: '\u00B3',
        }
        superscripts.update((n, chr(ord('\u2070') + n)) for n in range(4, 10))

        # return f"Derivative({self._f}, {self._x}, {self._n})"
        numer = f"\u2202{superscripts[self._n]}{self._f}"
        denom = ""
        for k in sorted(self._x.keys()):
            v = self._x[k]
            if v > 1:
                denom += f"\u2202{k}{superscripts[v]}"
            else:
                denom += f"\u2202{k}"
        return f"({numer}/{denom})"

    def __mul__(self, rhs):
        if self.func_order == 0:
            return rhs
        if rhs.func_order == 0:
            return self
        else:
            return MultiplyOp(rhs, self)

def check_individual_constraints(term,funcs):
    out = True
    for func in funcs:
        termf = term[func.root]
        out = out and ((termf.func_order <= func.max_func_order) and (termf.diff_order <= func.max_diff_order))
    return out

def build_library_expr_with_base(funcs, ivars, constraints, base):
    """
    Given function ['f', 'g', ...] and independent variables (usually spatial coordinates)
    ['x', 'y', ... ], and a set of constraints (usually on total function order and differential
    order, we want all possible terms involving the functions and their derivatives subject to
    said constraints.

    Really, what this boils down to is constructing all possible derivatives, and a way of counting
    the differential order.
    """
    
    # np.empty([],dtype=data_base.dtype)
    term_order = {'1': Function.unity()}
    for func in funcs:
        f = copy.deepcopy(func)
        f.func_order = 0
        f.diff_order = 0
        if f.root not in term_order:
            term_order[f.root] = f
    
    for terms in itertools.combinations_with_replacement(base, constraints['func_order']):
        term = terms[0]
        term_order[terms[0].root].func_order += terms[0].func_order
        term_order[terms[0].root].diff_order += terms[0].diff_order
        for multiplier in terms[1:]:
            term_order[multiplier.root].func_order += multiplier.func_order
            term_order[multiplier.root].diff_order += multiplier.diff_order
            term = term * multiplier

        if ( (term.func_order <= constraints['func_order'])
                 and (term.diff_order <= constraints['diff_order'])
                 and (check_individual_constraints(term_order,funcs) )):
            yield term

        for func in funcs:
            term_order[func.root].func_order = 0
            term_order[func.root].diff_order = 0

def build_library_expr(funcs, ivars, constraints):
    """
    Given function ['f', 'g', ...] and independent variables (usually spatial coordinates)
    ['x', 'y', ... ], and a set of constraints (usually on total function order and differential
    order, we want all possible terms involving the functions and their derivatives subject to
    said constraints.

    Really, what this boils down to is constructing all possible derivatives, and a way of counting
    the differential order.
    """
    
    base = []
    # np.empty([],dtype=data_base.dtype)
    term_order = {'1': Function.unity()}
    for func in funcs:
        f = copy.deepcopy(func)
        f.func_order = 0
        f.diff_order = 0
        if f.root not in term_order:
            term_order[f.root] = f
    for vs in itertools.combinations_with_replacement(['1'] + ivars, constraints['diff_order']):
        for f in funcs:
            for v in vs:
                if v != '1':
                    f = Derivative(f, v)
            if f.diff_order <= constraints['diff_order'] and f.func_order > 0:
                base.append(f)

    base.append(Function.unity())
    for terms in itertools.combinations_with_replacement(base, constraints['func_order']):
        term = terms[0]
        term_order[terms[0].root].func_order += terms[0].func_order
        term_order[terms[0].root].diff_order += terms[0].diff_order
        for multiplier in terms[1:]:
            term_order[multiplier.root].func_order += multiplier.func_order
            term_order[multiplier.root].diff_order += multiplier.diff_order
            term = term * multiplier

        if ( (term.func_order <= constraints['func_order'])
                 and (term.diff_order <= constraints['diff_order'])
                 and (check_individual_constraints(term_order,funcs) )):
            yield term

        for func in funcs:
            term_order[func.root].func_order = 0
            term_order[func.root].diff_order = 0
            
def build_base_expr(funcs, ivars, constraints):
    """
    Given function ['f', 'g', ...] and independent variables (usually spatial coordinates)
    ['x', 'y', ... ], and a set of constraints (usually on total function order and differential
    order, we want all possible terms involving the functions and their derivatives subject to
    said constraints.

    Really, what this boils down to is con  structing all possible derivatives, and a way of counting
    the differential order.
    """
    
    base = []
    for vs in itertools.combinations_with_replacement(['1'] + ivars, constraints['diff_order']):
        for f in funcs:
            for v in vs:
                if v != '1':
                    f = Derivative(f, v)
            if f.diff_order <= constraints['diff_order'] and f.func_order > 0 and f.diff_order <= f.max_diff_order:
                base.append(f)

    base.append(Function.unity())
    return base

def get_term_val(lib,term):
    """
    get_term_val(lib,term):

    Function that returns a term from the library whose name is term.

    Parameters
    ----------
    lib  : dtype=[('name','U50'),('val','float64',U.shape)] NumPy
           structured array for the library (where U is a
           flattened array containing the terms). 
        Original library
    term : str
            Name of the term to be returned
    Returns
    -------
    arr : ndarray
        Array containing the values of the term
    """
    return np.squeeze(lib[lib['name']==str(term)]['val'],axis=0)

def get_rhs(lib,w):
    rhs_array = np.zeros_like(lib[0]['val'])
    for i in range(len(w)):
        if w[i] != 0:
            rhs_array += w[i] * lib[i]['val']
    return rhs_array

def delete_term(lib,term):
    """
    delete_term(lib,term):

    Function that deletes a term from the library whose name is term.

    Parameters
    ----------
    lib  : dtype=[('name','U50'),('val','float64',U.shape)] NumPy
            structured array for the library (where U is a
            flattened array containing the terms). 
    term : str
            Name of the term to be deleted
    Returns
    -------
    lib : dtype=[('name','U50'),('val','float64',U.shape)] NumPy
            structured array for the library (where U is a
            lattened array containing the terms).
            Library with the term deleted from it.
    """

    return np.delete( lib, np.argwhere( lib['name']==str(term) ) )

def get_desc_and_X(library):
    
    """
    get_desc_and_X(lib):

    Function that outputs the names of the terms as a list of strings
    and the values of the function as a corresponding 2D array
    prescribed coefficients.

    Parameters
    ----------
    library  : dtype=[('name','U50'),('val','float64',U.shape)] NumPy
                structured array for the library (where U is a
                flattened array containing the terms). 
    
    Returns
    -------
    desc : list of str
            List of strings containing the names of the terms
    X    : Numpy Array
            Numpy array containing the corresponding values of the terms
            such that the value of desc[0] is X[:,0]
    """

    desc = list(library['name'])
    X = library['val'].T
    return ( desc, np.real(X) )

def add_term(lib,term,term_val):

    new_term_struct = np.array([(str(term),term_val)],
                                dtype=lib.dtype)

    return np.append(lib,new_term_struct)

def convert_to_lib_as_type(lib,term,term_val):

    return np.array([(str(term),term_val)],
                                dtype=lib.dtype)
                                
def build_constrained_library_array(funcs, base, 
                                    data_base, ivars, 
                                    constraints,
                                    print_terms=False):

    term_order = {'1': Function.unity()}
    for func in funcs:
        f = copy.deepcopy(func)
        f.func_order = 0
        f.diff_order = 0
        if f.root not in term_order:
            term_order[f.root] = f
    
    # Count the number of terms first, to initialize the library array.
    library_length = len( list( build_library_expr_with_base(funcs, ivars, constraints, base) ) )
    # print(f"Expecting {library_length} terms...")
    library = np.empty(library_length, dtype=data_base.dtype)
    term_id = 0
    for terms in itertools.combinations_with_replacement(base, constraints['func_order']):
        term = terms[0]
        term_order[terms[0].root].func_order += terms[0].func_order
        term_order[terms[0].root].diff_order += terms[0].diff_order
        term_arr = np.squeeze(data_base[data_base['name']==str(term)]['val'],axis=0)
        for multiplier in terms[1:]:
            term = term * multiplier
            term_order[multiplier.root].func_order += multiplier.func_order
            term_order[multiplier.root].diff_order += multiplier.diff_order
            mul_arr = np.squeeze(data_base[data_base['name']==str(multiplier)]['val'],axis=0)
            term_arr = term_arr * mul_arr
        if ( (term.func_order <= constraints['func_order'])
                 and (term.diff_order <= constraints['diff_order'])
                 and (check_individual_constraints(term_order,funcs) )):
            if print_terms:
                print(f"({term_id}): {str(term)}")
            # new_term = np.array([(str(term),term_arr)],dtype=data_base.dtype)
            # library = np.append(library,new_term)
            library[term_id]['name'] = str(term)
            library[term_id]['val'] = term_arr
            
            term_id +=1
            # yield term
        for func in funcs:
            term_order[func.root].func_order = 0
            term_order[func.root].diff_order = 0

    return library

def combine_terms(library,
                  terms,
                  coeffs,
                  new_term,
                  show_combinations=False):

    """
    combine_terms(library, terms_to_combine, coeffs):

    Function to combine some terms in the library array with
    prescribed coefficients.

    Parameters
    ----------
    library  : dtype=[('name','U50'),('val','float64',U.shape)] NumPy
                structured array for the library (where U is a
                flattened array containing the terms). 
    terms    : list
                List of strings containing the terms to combine.
    coeffs   : list 
                List of coefficients to use to combine
                the terms.
    new_term : str
                Name of the resulting new term
    Returns
    -------
    new_library : dtype=[('name','U50'),('val','float64',U.shape)]
                    NumPy structured array containing the library with
                    the combined terms.
                    Returns the original library if the new term 
                    already exists in it.

    Raises
    ------
    ValueError 
        if some terms in the 'terms' list are not present in the 
        original library.
    ValueError 
        if the number of terms to combine is not equal to the
        number of coefficients provided.
    """

    library_terms = list(library['name'])
    
    if new_term in set(library_terms):
        msg = f"New term {new_term} already in the library!"
        print(msg)
        return library

    difference = set(terms).difference(library_terms)
    if ( len(difference) != 0 ):
        msg = f"The terms {difference} are not present in the \
                original library!"
        raise ValueError(msg)

    if ( len(terms) != len(coeffs) ):
        msg = "Number of terms to combine is not equal to the \
               number of coefficients provided!"
        raise ValueError(msg)
    
    library_dtype = library.dtype
    new_term_array = np.zeros_like(library[0]['val'])
    new_term_str = 'Combining {\n'
    for (term, coeff) in zip(terms,coeffs):
        new_term_array += coeff * get_term_val( library, term )
        new_term_str += f'\t {coeff} × {term}\n'
        library = delete_term( library, term )

    new_term_str += '}' + f' → {new_term}'
    if show_combinations:
        print(new_term_str)    
    new_term_struct = np.array([(str(new_term),new_term_array)],
                                dtype=library_dtype)
    
    library = np.append(library,new_term_struct)

    return library
