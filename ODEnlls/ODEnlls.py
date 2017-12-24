'''
Non-linear least squares fitting for chemical kinetics fitting using
simulations of ordinary differential equations for an arbitary set chemical
reactions. 
'''
import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as spo 
import scipy.integrate as spi 
import pandas as pd


class ODEnlls():
    ##### FILE INPUT METHODS #####

    def read_data(self, filename, comment='#'):
        '''
        Read in a comma-separated data file. 
        
        This function simply reads the data file into a Pandas DataFrame
        attribute. 

        Parameters
        ----------
        filename : string
            The name of the data file. Can be a relative or full path as well.
            See Pandas.read_csv for more info. 

        comment : string
            The character that will denote the start of comments in the file. 
        '''
        self.data = pd.read_csv(filename, comment=comment)

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
        cpds = []
        self.odes = []
        self.rxns = []
        count = 1

        fileobj = open(filename)
        for rxn in fileobj:
            if rxn.isspace(): continue
            elif rxn[0] == '#': continue
            
            rxn = rxn.strip()
            self.rxns.append(rxn)
            try:
                kcount, vtemp, rtemp = self._rxnRate(rxn, count=count)
            except:
                print("There was an error with your reaction file!")
                print("See line: {}".format(rxn))
                return False

            count += kcount 
            for num, v in enumerate(vtemp):
                if v not in cpds:
                    cpds.append(v)
                    self.odes.append(rtemp[num])
                else:
                    index = cpds.index(v)
                    self.odes[index] += ' + ' + rtemp[num]
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
        cmpds = []
        rxns = []
   
        # Determine the reaction type -- reversible or not -- and split the
        # reaction accordingly.
        if '->' in rxn:
            halves = rxn.split('->')
            rev = False
        elif '=' in rxn:
            halves = rxn.split('=')
            rev = True
        else:
            return False

        # Process the half reactions. This function returns a lits of
        # compounds in the half reaction and the rate expression for that
        # half.
        try:
            sm, lrate, lstoic = self._halfRxn(halves[0])
            pr, rrate, rstoic = self._halfRxn(halves[1])
        except:
            return False

        # Generate the full rate expression for the reaction without the
        # stoichiometry corrections necessary for each component.
        if rev == False:
            sm_rate = '-1*k{:d}*{}'.format(count, lrate)
            pr_rate = 'k{:d}*{}'.format(count, lrate)
            kcount = 1
        elif rev == True:
            sm_rate = '-1*k{:d}*{} + k{:d}*{}'.format(count, lrate, 
                                                count+1, rrate)
            pr_rate = 'k{:d}*{} + -1*k{:d}*{}'.format(count, lrate, 
                                                count+1, rrate)
            kcount = 2
        
        # For each compound on the left side of the reaction, add the name to
        # our compounds list and the corresponding rate expression to our rate
        # list with a correction for the stoichiometry.
        for (x, coef) in zip(sm, lstoic):
            cmpds.append(x)
            temp = '{:.2f}*({})'.format(coef, sm_rate)
            rxns.append(temp)

        # Do the same thing for the products; however, check to see if the
        # product is also in the sm list. If that's the case, then modify the
        # stoichiometry correction accordingly.
        for (y, coef) in zip(pr, rstoic):
            if y not in cmpds:
                cmpds.append(y)
                temp = '{:.2f}*({})'.format(coef, pr_rate)
                rxns.append(temp)
            else:
                index = cmpds.index(y)
                newcoef = coef - lstoic[index]
                rxns[index] = '{:.2f}*({})'.format(newcoef, pr_rate)
    
        return kcount, cmpds, rxns

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
            for i in sp:
                if sp.count(i) > 1:
                    print("You've used an improper format.", )
                    print("For example, instead of 'A + A' use '2*A'.")
                    return False
        else:
            sp = [half.strip()]

        # Some of the compounds might be multiplied by a coefficient. Remove
        # the coefficient to set the stoichiometry. Compounds that have a
        # stoic of >1 will need to be divided by the stoich.
        for comp in sp:
            if '*' in comp:
                sp2 = comp.split('*')
                reactants.append(sp2[1])
                stoic.append( float( sp2[0] ) )
                temp = '({}**{:.2f})/{:.2f}'.format(sp2[1], stoic[-1], 
                                                stoic[-1])
            else:
                reactants.append(comp)
                temp = comp 
                stoic.append( 1.0 )

            # Create the rate string for this half reaction. Multiple
            # compounds will need to be multiplied together.
            if rate == '':
                rate = '({})'.format(temp)
            else:
                rate += '*({})'.format(temp)

        return reactants, rate, stoic
    
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
        # WARNING: This has the potential to cause some problems. Replace may
        # goof up if one compound name is a subset of another.
        for n, cpd in enumerate(self._cpds):
            funct = funct.replace(cpd, 'y[' + str(n) + ']')

        # I needed to create a temporary dictionary for the executed function
        # definition. This is new w/ Py3, and I'll need to explore this a
        # little to make it more streamlined.
        temp = {}
        exec(funct, temp)
        self._ode_function = temp['f']
        self._function_string = funct

    ##### PLOTTING METHODS #####
            
    def plot(self, plottype='guess', legend=True, colorlines=False):
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
            startt = self.data.iloc[0, 0]
            finalt = self.data.iloc[-1, 0]
            times = np.linspace(startt, 1.05*finalt, 1000)

            if plottype == 'fit':
                solution = self._sim_odes(pars='fit', times=times)
            else: 
                solution = self._sim_odes(times=times)

            if plottype != 'sim' or colorlines == False:
                plt.plot(times, solution, 'k-')
                legend = False
            else:
                conc = self.params.filter(regex=r'^(?!k\d+$)', axis=0).index
                for n, c in enumerate(conc):
                    plt.plot(times, solution[:,n], label=c)
            
        if plottype in ['guess', 'fit', 'data']:
            for col in self.data.columns[1:]:
                plt.plot(self.data.iloc[:,0], self.data[col], 'o', label=col)

        if plottype == 'res':
            for col in self.residuals.columns[1:]:
                plt.plot(self.residuals.iloc[:,0], self.residuals[col],
                        'o-', label=col)

        if legend == True: plt.legend(numpoints=1)
    
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

    def run_fit(self):
        '''
        The data fitting function.
        '''
        # Get parameters for the fitting process. Skip the fixed variables.
        mask = self.params['guess'].notna()
        fitpars = list(self.params.loc[mask, 'guess']) 
        
        # Fit the data and convert the output to various lists.
        fitdata = spo.leastsq(self._residTotal, fitpars, full_output=1)
        p, cov, info, mesg, success = fitdata 
        if success not in [1,2,3,4] or type(cov) == type(None):
            print('There was a fitting error.')
            return False

        # Set the fit parameters in the `params` DataFrame
        self.params['fit'] = 0.0
        self.params.loc[mask, 'fit'] = p
        self.params.loc[~mask, 'fit'] = self.params.loc[~mask, 'fix']

        # Create a residuals DataFrame. I need to be careful to match the data
        # up to the correct DataFrame columns.
        raw = self.data.iloc[:,1:].values
        vals_shape = raw.shape
        res_temp = info["fvec"].reshape(vals_shape)
        self.residuals = self.data.copy()
        self.residuals[self._cpds] = res_temp

        # Create some statistical parameters for the fit
        # Chi squared
        chisq = (info["fvec"]**2).sum()

        # R-squared
        y_ave = raw.mean()
        yave_diff = ((raw - y_ave)**2).sum()
        rsq = 1.0 - (chisq/yave_diff)

        # Akaike's Information Criterion. The use of this parameter is based
        # on the following reference, which is specific to chemical kinetics
        # fitting: Chem. Mater. 2009, 21, 4468-4479 The above reference gives
        # a number of additional references that are not directly related to
        # chemical kinetics.
        lenp = len(p)
        lenfvec = len(info["fvec"])
        aic = lenfvec*np.log(chisq) + 2*lenp + \
                ( 2*lenp*(lenp + 1) )/(lenfvec - lenp - 1)

        # Calculate parameter errors. Make sure that parameter errors don't
        # get set for fixed values
        dof = lenfvec - lenp
        sqcsdof = np.sqrt(chisq/dof)
        sigma = np.sqrt(np.diag(cov))*sqcsdof
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
        res = self.data[self._cpds].values - solution

        # It must be 1D to work properly
        return res.flatten()

    def _sim_odes(self, pars=None, times=None):
        '''Run an ODE simulation with the given parameters.
        '''
        # Make a temporary parameter set so the original is unaffected.
        if isinstance(pars, str) and pars == 'fit':
            guesses = self.params['fit'].copy()
            mask = guesses.notna()
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
        
        solution = spi.odeint(self._ode_function, y0=list(conc), 
                t=times, args=tuple(kvals))

        return solution


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

