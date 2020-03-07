'''
Non-linear least squares fitting for chemical kinetics fitting using
simulations of ordinary differential equations for an arbitary set chemical
reactions. 
'''
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as spo 
import scipy.integrate as spi 


class ODEnlls():
    def __init__(self, add_zero=True):
        """Initialize the instance...

        Parameters
        ----------
        add_zero : bool (default True)
            This controls whether or not to add t=0 data to the ode
            simulations. This should probably always be the case...
        """
        # I need to set a boolean flag for adding t=0 to the ode simulations
        # This should probably always be True... But I put in a kwarg to
        # ensure that it can be changed if necessary.
        self._add_zero = add_zero


    ##### FILE INPUT METHODS #####

    def read_data(self, filename, **kwargs):
        '''
        Read in a comma-separated data file. 
        
        This is a very simple wrapper function around pandas.read_csv
        function; however, it sets the data as the correct object attribute
        named `data`.

        Parameters
        ----------
        filename : string or file handle
            The name of the data file. Can be a relative or full path as well.
            See Pandas.read_csv for more info. 

        kwargs : dict-like
            Arbitrary keyword arguments that can be passed to pandas.read_csv
            function. See the docs for that function to see full options.
        '''
        self.data = pd.read_csv(filename, **kwargs)

        # increment the index by 1 in order to accomodate the added zero line
        # in the fitting, this shouldn't really break anything...
        self.data.index += 1

    def read_rxns(self, filename):
        '''
        Takes a file name of fundamental reaction steps and breaks them up
        into lists of ODEs, compounds, reactions, equilibrium constants, and
        initial concentrations.

        The reaction file must have one reaction per line. Reaction components
        must be separated by '+'; stoichiometry can be included using '*'
        (e.g. 2*A for two equivalents of A); irreversible reactions are
        denoted by '->'; reversible reactions are denoted by '='.
        '''
        # Check if string filename or file object
        if isinstance(filename, str):
            fileobj = open(filename)
        elif hasattr(filename, 'read'):
            fileobj = filename
        else:
            raise ValueError('The file input should be a string or '
                    'file-like object.')

        cpds = []
        self.odes = []
        self.rxns = []
        count = 1

        for rxn in fileobj:
            if rxn.isspace(): continue
            elif rxn[0] == '#': continue
            
            rxn = rxn.strip()
            self.rxns.append(rxn)
            kcount, cpdtemp, ratetemp = self._rxnRate(rxn, count=count)

            count += kcount 
            for c, r in zip(cpdtemp, ratetemp):
                if c not in cpds:
                    cpds.append(c)
                    self.odes.append(r)
                else:
                    index = cpds.index(c)
                    self.odes[index] += ' + ' + r

        fileobj.close()
        
        ks = ['k{:d}'.format(i) for i in range(1, count)]
        self.params = pd.DataFrame(np.nan, columns=['guess', 'fix'], 
                                index=cpds + ks)
        self._ks = ks
        self._cpds = cpds

        self._ode_function_gen()

    def _rxnRate(self, rxn, count=1):
        '''
        Convert a fundamental reaction into lists of equilibrium constants
        (all zero), reaction components, and their corresponding rate
        expressions.

        Requires the function halfrxn() for each part of the reaction.
        '''
        # Determine the reaction type -- reversible or not -- and split the
        # reaction accordingly.
        if '->' in rxn:
            halves = rxn.split('->')
            rev = False
            kcount = 1
        elif '=' in rxn:
            halves = rxn.split('=')
            rev = True
            kcount = 2
        else:
            er = "The following rxn is incorrect:\n"
            er += rxn + "\n"
            er += "The reaction must contain a '=' or '->'"
            raise ValueError(er)

        # Process the left half reaction. This function returns a lits of
        # compounds in the half reaction and the rate expression for that
        # half.
        sms, lrate, lstoic = self._halfRxn(halves[0])
        prs, rrate, rstoic = self._halfRxn(halves[1])

        # Multiply the concentrations by the equilibrium constant
        rate = 'k{:d}*{}'.format(count, lrate)
        
        cmpds = []
        rxns = []
        lstoic_prod = np.prod(lstoic)

        # Process the reactants, which are disappearing
        self._buildOde(rate, cmpds, rxns, sms, lstoic, lstoic_prod, '-')
        # Process the products, which are growing in
        self._buildOde(rate, cmpds, rxns, prs, rstoic, lstoic_prod, '')

        if rev == True:
            # If this is reversible, add the reversible terms as well, which
            # are opposite to the forward rxns 
            rate_rev = 'k{:d}*{}'.format(count+1, rrate)
            rstoic_prod = np.prod(rstoic)

            self._buildOde(rate_rev, cmpds, rxns, sms, lstoic, rstoic_prod,
                    '')
            self._buildOde(rate_rev, cmpds, rxns, prs, rstoic, rstoic_prod,
                    '-')

        return kcount, cmpds, rxns


    def _buildOde(self, rate, allcpds, allrxns, cpds, stoics, prod, sign=''):
        '''Create the appropriate ODE for each compound. 

        This can be done for either the products or reactants, and they are
        combined into a larger full compound (allcpds) and full rxn (allrxn)
        lists.
        '''
        for (cpd, stoic) in zip(cpds, stoics):
            # Calculate a coefficient that adjusts the rate to take into
            # account relative stoichiometry
            coef = stoic/prod

            # If the coefficient is pretty much 1, then we don't need to put
            # in any adjustment. ### TODO... add a threshold
            if coef >= 0.99999 and coef <= 1.00001:
                temp = sign + rate
            else:
                # Add the factor to take into account the stoichiometry. I
                # didn't use the coef directly, so I wouldn't introduce
                # rounding errors
                temp = sign + '({:.2f}/{:.2f})*{}'.format(stoic, prod, rate)

            if cpd not in allcpds:
                # Add the compound/rxn to the big lists if cpd isn't there
                allcpds.append(cpd)
                allrxns.append(temp)
            else:
                # Append the ODE to the existing one if cpd is already there.
                idx = allcpds.index(cpd)
                allrxns[idx] += ' + {}'.format(temp)
        

    def _halfRxn(self, half):
        '''
        Break apart a half reaction. Returns a lists of reactants and their
        corresponding rate expressions.
        '''
        reactants = []
        stoic = []
        rate = ''

        # Create a list of all the compounds in this half reaction
        if '+' in half:
            sp = half.split('+')
            sp = [x.strip() for x in sp]
        else:
            sp = [half.strip()]

        # Some of the compounds might be multiplied by a coefficient. Remove
        # the coefficient to set the stoichiometry. 
        for comp in sp:
            if '*' in comp:
                coef, comp = comp.split('*')
                coef = float(coef)
            else:
                coef = 1.0

            if comp not in reactants:
                reactants.append(comp)
                stoic.append( float(coef) )
            else:
                idx = reactants.index(comp)
                stoic[idx] += float(coef)

            if coef == 1.0:
                rate += '*([' + comp + '])'
            else:
                # Compounds multiplied by a stoiciometric coeffcient need to
                # be raised to that power
                rate += '*([{}]**{:.2f})'.format(comp, coef)

        return reactants, rate[1:], stoic

    
    def _ode_function_gen(self, ):
        '''
        Dynamic ODE function creation.

        This function creates a dynamic function that represents the ODEs for
        this system. It does this by first creating a properly formatted
        string that can be exectued by the interpreter to dynamically create a
        function represented by the input reactions.

        See scipy.integrate.odeint for function structure.
        '''
        # Start the function definition
        funct = 'def f(y, t, '

        # Add the k values to the function definition.
        funct += ', '.join(self._ks)

        # Close the function def line, and start the return list of ODEs
        funct += '):\n'
        funct += '    return [\n' 

        # Add the ODE rate lines to the function string
        for rate in self.odes:
            funct += ' '*8 + rate + ',\n'
        funct += ' '*8 + ']'

        # Replace the compound names with the appropriate values from the
        # concentration list (i.e. `y`)
        # WARNING: This has the potential to cause some problems if the
        # compound names has some [] in it. However, this is a very unusual
        # case I would image
        for n, cpd in enumerate(self._cpds):
            funct = funct.replace('[' + cpd + ']', 'y[' + str(n) + ']')

        # I needed to create a temporary dictionary for the executed function
        # definition. This is new w/ Py3, and I'll need to explore this a
        # little to make it more streamlined.
        temp = {}
        exec(funct, temp)
        self._ode_function = temp['f']
        self._function_string = funct


    def print_odes(self, ):
        '''Pretty printing function for the ODEs'''
        if not hasattr(self, 'odes'):
            raise ValueError('No rxns processed yet, so no ODEs.')

        for c, o in zip(self._cpds, self.odes):
            print('D['+c+']/dt =', o)

    ##### PLOTTING METHODS #####
            
    def plot(self, plottype='guess', legend=True, colorlines=False, times=None):
        '''
        Display a plot of the ODE silulation and/or data.

        The possible plot 'type' options are as follows: 'guess' = Plot data
        and ode simulation using initial concentration and equilibrium
        constant guesses.  'fit' = Plot data and ode simulation using the
        fitted concentration and eq. constants.  'sim' = Plot only the ode
        simulation using the guess concentrations and eq constants.  'data' =
        Plot data only.  'res' = Plot the residuals from the current fit.

        The legend boolean controls the display of a figure legend.

        The colorLines boolean controls whether the ode lines are colored or
        black.
        '''
        if plottype in ['sim', 'guess', 'fit']:
            # Generate the simulated data
            self.simulate(times=times, simtype=plottype)

            if colorlines == False:
                plt.plot(self.sims['Times'], self.sims.iloc[:,1:], 'k-')
            else:
                for n, c in enumerate(self._cpds):
                    plt.plot(self.sims['Times'], self.sims.iloc[:,n+1], 
                            label=c)
            
        if plottype in ['guess', 'fit', 'data']:
            for col in self.data.columns[1:]:
                plt.plot(self.data.iloc[:,0], self.data[col], 'o', label=col)

        if plottype == 'res':
            for col in self.residuals.columns[1:]:
                plt.plot(self.residuals.iloc[:,0], self.residuals[col],
                        'o-', label=col)

        if legend == True and \
                not (plottype == 'sim' and colorlines == False): 
            plt.legend(numpoints=1)
    
    def simulate(self, times=None, npts=1000, simtype='sim'):
        # Create a times array
        if simtype in ['guess', 'fit']:
            startt = self.data.iloc[0, 0]
            finalt = self.data.iloc[-1, 0]
            times = np.linspace(startt, 1.05*finalt, npts)
        elif isinstance(times, type(None)) and simtype == 'sim':
            times = np.linspace(0, 100, npts)
        elif isinstance(times, (int, float)) and simtype == 'sim':
            times = np.linspace(0, times, npts)

        # Chose the parameter set for the simulation
        if simtype == 'fit':
            pars = 'fit'
        else:
            pars = None
        
        # Generate the ODE solution
        sims = self._sim_odes(pars=pars, times=times)

        self.sims = pd.DataFrame(np.c_[times, sims], 
                                columns=['Times']+self._cpds)

    ##### FITTING METHODS #####

    def set_param(self, *args, ptype='guess'):
        '''docstring: TODO
        '''
        if ptype not in ['guess', 'fix']:
            raise ValueError('ptype parameter can only be "guess" or "fix".')

        if isinstance(args[0], str) and len(args) == 2:
            pars = {args[0]:args[1],}

        elif isinstance(args[0], dict):
            pars = args[0]

        else:
            raise ValueError('There is something wrong with your input.')

        for row in pars:
            if row not in self.params.index:
                raise ValueError('The row {} is not valid.'.format(row))

            value = pars[row]
            if not isinstance(value, float):
                try:
                    value = float(value)
                except:
                    er = 'The parameter "{}" for row "{}" must be '\
                        'convertable to a float.'
                    print(er.format(value, row))
                    raise

            self.params.loc[row, ptype] = value

    def run_fit(self, add_zero=True):
        '''
        The data fitting function.

        Parameters
        ----------
        add_zero: bool
            This flag forces the addition of a t=0 line to the data file. This
            is most likely always going to be necessary, but the flag allows
            you to remove this if necessary.
        '''
        self._add_zero = add_zero

        # Get parameters for the fitting process. Skip the fixed variables.
        mask = self.params['guess'].notna()
        fitpars = list(self.params.loc[mask, 'guess']) 
        # I'll need the indices to create the covariance matrix later.
        self._fitpars_index = self.params.loc[mask, 'guess'].index

        # Get a list of data compounds. I'll need this to compare with the
        # parameter data, which might be different order/number of cpds
        self._data_cpds = self.data.columns[1:]
        self._data_idx = [self._cpds.index(i) for i in self._data_cpds]

        # Fit the data and convert the output to various lists.
        fitdata = spo.leastsq(self._residTotal, fitpars, full_output=1)
        p, cov, info, mesg, success = fitdata 
        if success not in [1,2,3,4] or type(cov) == type(None):
            er = 'Perhaps change your guess parameters or input reactions.'
            raise FitError(er)

        # Set the fit parameters in the `params` DataFrame
        self.params['fit'] = 0.0
        self.params.loc[mask, 'fit'] = p
        self.params.loc[~mask, 'fit'] = self.params.loc[~mask, 'fix']

        # Create a residuals DataFrame. I need to be careful to match the data
        # up to the correct DataFrame columns.
        raw = self.data.iloc[:,1:].values
        vals_shape = raw.shape
        res_temp = pd.DataFrame(info["fvec"].reshape(vals_shape), 
                            columns=self._data_cpds)
        times = self.data.iloc[:,[0]].copy()
        self.residuals = pd.concat([times, res_temp], axis=1)

        # Create some statistical parameters for the fit
        # Chi squared and sum of squares (ssq)
        squared = info["fvec"]**2
        ssq = squared.sum()
        chisq = (squared/raw.flatten()).sum()

        # R-squared
        y_ave = raw.mean()
        yave_diff = ((raw - y_ave)**2).sum()
        rsq = 1.0 - (ssq/yave_diff)

        # Akaike's Information Criterion. The use of this parameter is based
        # on the following reference, which is specific to chemical kinetics
        # fitting: Chem. Mater. 2009, 21, 4468-4479 The above reference gives
        # a number of additional references that are not directly related to
        # chemical kinetics.
        lenp = len(p)
        lenfvec = len(info["fvec"])
        aic = lenfvec*np.log(ssq) + 2*lenp + \
                ( 2*lenp*(lenp + 1) )/(lenfvec - lenp - 1)

        # Calculate parameter errors and covariance matrix. Make sure that
        # parameter errors don't get set for fixed values
        dof = lenfvec - lenp
        pcov = cov*(ssq/dof)
        self.pcov = pd.DataFrame(pcov, columns=self._fitpars_index,
                index=self._fitpars_index)

        sigma = np.sqrt(np.diag(pcov))
        self.params['error'] = np.nan
        self.params.loc[mask, 'error'] = sigma

        # Create a stats dictionary
        self.stats = {
                'chi**2': chisq,
                'R**2' : rsq,
                'AIC' : aic,
                'DOF' : dof,
                }

    def _residTotal(self, pars):
        '''
        Generate and return one list of all the residuals.

        This function generates residuals by comparing the differences between
        the ODE and the actual data only when the ODE compound name
        (self.cpds) is one of the data set names (keys of self.useData).
        '''
        solution = self._sim_odes(pars=pars)
        
        # Make the return np.array of residuals
        res = self.data.iloc[:,1:].values - solution[:, self._data_idx]

        # It must be 1D to work properly
        return res.flatten()

    def _sim_odes(self, pars=None, times=None):
        '''Run an ODE simulation with the given parameters.
        '''
        # Make a temporary parameter set so the original is unaffected.
        # I must check if it is a string b/c numpy/pandas can cause problems.
        if isinstance(pars, str) and pars == 'fit':
            guesses = self.params['fit'].copy()
        else:
            guesses = self.params['guess'].copy()
            mask = guesses.notna()
            if not isinstance(pars, type(None)):
                guesses.loc[mask] = pars
            guesses.loc[~mask] = self.params.loc[~mask, 'fix']

        conc = guesses[self._cpds]
        kvals = guesses[self._ks]

        if isinstance(times, type(None)):
            times = list(self.data.iloc[:,0])

        # Add a zero time to the simulation, if it is not present.
        # If it is present, then set a flag to note that is the case.
        if np.min(times) == 0.0:
            time_zero = True
        else:
            time_zero = False

        if self._add_zero and not time_zero:
            temp = [0.0,]
            temp.extend(times)
            times = temp
        
        solution = spi.odeint(self._ode_function, y0=list(conc), 
                t=times, args=tuple(kvals))

        # If there wasn't a time zero and one was added, remove the first row
        # from the simulation.
        if self._add_zero and not time_zero:
            solution = solution[1:,:]

        return solution


class FitError(Exception):
    '''A custom error for fitting.

    This doesn't do anything special. It is just a way to handle specific
    errors that are related to fitting.
    '''
    pass

# This function will determine the Akaike weights from a series of Akaike
# Information Criterion fit statistic values (ODEnlls.aic).

def weightAic(values):
    minimum = min(values)
    sum = 0
    diffs = []
    for v in values:
        temp = np.exp(-1*(v - minimum)/2)
        diffs.append(temp)
        sum += temp
    diffs = [d/sum for d in diffs]
    return diffs

