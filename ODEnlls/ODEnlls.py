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

    def _halfRxn(self, half):
        '''
        Break apart a half reaction. Returns a lists of reactants and their
        corresponding rate expressions.
        '''
        reactants = []
        stoic = []
        rate = ''

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

        for comp in sp:
            if '*' in comp:
                sp2 = comp.split('*')
                reactants.append(sp2[1])
                stoic.append( float( sp2[0] ) )
                temp = '(%s**%.2f)/%.2f' % (sp2[1], stoic[-1], stoic[-1])
            else:
                reactants.append(comp)
                temp = comp 
                stoic.append( 1.0 )

            if rate == '':
                rate = '(%s)' % temp
            else:
                rate += '*(%s)' % temp

        return reactants, rate, stoic
    
    def _rxnRate(self, rxn, count=1):
        '''
        Convert a fundamental reaction into lists of equilibrium constants
        (all zero), reaction components, and their corresponding rate
        expressions.

        Requires the function halfrxn() for each part of the reaction.
        '''
        cmpds = []
        rxns = []
        ks = []
   
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
            sm_rate = '-1*k%d*%s' % (count, lrate)
            pr_rate = 'k%d*%s' % (count, lrate)
            ks = [0.0]
            #count += 1
        elif rev == True:
            sm_rate = '-1*k%d*%s + k%d*%s' % (count, lrate, count+1, rrate)
            pr_rate = 'k%d*%s + -1*k%d*%s' % (count, lrate, count+1, rrate)
            ks = [0.0, 0.0]
            #count += 2
        
        # For each compound on the left side of the reaction, add the name to
        # our compounds list and the corresponding rate expression to our rate
        # list with a correction for the stoichiometry.
        for (x, coef) in zip(sm, lstoic):
            cmpds.append(x)
            temp = '%.2f*(%s)' % (coef, sm_rate)
            rxns.append(temp)

        # Do the same thing for the products; however, check to see if the
        # product is also in the sm list. If that's the case, then modify the
        # stoichiometry correction accordingly.
        for (y, coef) in zip(pr, rstoic):
            if y not in cmpds:
                cmpds.append(y)
                temp = '%.2f*(%s)' % (coef, pr_rate)
                rxns.append(temp)
            else:
                index = cmpds.index(y)
                newcoef = coef - lstoic[index]
                rxns[index] = '%.2f*(%s)' % (newcoef, pr_rate)
    
        return ks, cmpds, rxns

    def _functGen(self, rates, ks, vars):
        '''
        Take lists of rates, equilibrium constants, and reaction components
        and creates a Python function string. This string will work with
        scipy.odeint function, but needs to be modified if using the Sage ODE
        simulations.
        '''
        k_labels = ['k%d' % (num+1,) for num in range(len(ks))]
        funct = 'def f(y, t'
        for k in k_labels:
            funct += ', %s' % k
        funct += '):\n%sreturn [\n' % (' '*4,)
        for rate in rates:
            for num, var in enumerate(vars):
                if var in rate:
                    rate = rate.replace(var,'y[%d]' % num)
            funct += '%s%s,\n' % (' '*8, rate)
        funct += '%s]' % (' '*8,)
        return funct

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
        ks = []
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
                ktemp, vtemp, rtemp = self._rxnRate(rxn, count=count)
            except:
                print("There was an error with your reaction file!")
                print("See line: {}".format(rxn))
                return False

            count += len(ktemp)
            ks.extend(ktemp)
            for num, v in enumerate(vtemp):
                if v not in cpds:
                    cpds.append(v)
                    self.odes.append(rtemp[num])
                else:
                    index = cpds.index(v)
                    self.odes[index] += ' + %s' % rtemp[num]
        fileobj.close()
        
        klabel = ['k{:d}'.format(i+1) for i in range(len(ks))]
        self.params = pd.DataFrame(np.nan, columns=['guess', 'fix'], 
                                index=cpds + klabel)

        self.functString = self._functGen(self.odes, ks, cpds)
        temp = {}
        exec(self.functString, temp)
        self.function = temp['f']

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


    ##### PLOTTING METHODS #####
            
    def displayPlot(self, type='guess', legend=True, colorLines=False):
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
        # Make sure the correct figure number is being used, and if already
        # present, clear it and start over.
        plt.figure(self.figNum)
        if plt.fignum_exists(self.figNum):
            plt.clf()

        # Get to the actual plotting.
        # Plot types that contain the data. Be sure to normalize the data if
        # requested.
        if (type == 'data' or type == 'guess' or type == 'fit'):
            for key in self.useData:
                if self.useNorm == True:
                    self.useData[key] = self._normalize(self.inData[key])
                    plt.plot(self.times, self.useData[key], 'o', label=key)
                else:
                    self.useData[key] = self.inData[key]
                    plt.plot(self.times, self.useData[key], 'o', label=key)
            if type == 'guess' or type == 'fit':
                if type == 'guess': ctemp, ktemp = self._paramConvert()
                elif type == 'fit': ctemp, ktemp = \
                        self._paramConvert(params=self.fitPar)
                self._ODEplot(ctemp, ktemp, color=colorLines)
        
        # This simply plots the ODE simulation lines.
        elif type == 'sim':
            if legend == True and colorLines == False: legend = False
            ctemp, ktemp = self._paramConvert()
            self._ODEplot(ctemp, ktemp, color=colorLines) 

        # Plot the residuals of the data minus the ode solution.
        elif type == 'res':
            for key in self.resid:
                plt.plot(self.times, self.resid[key], '-o', label=key)

        if legend == True: plt.legend(numpoints=1)        
        plt.draw()

    def _ODEplot(self, c, k, color=False):
        '''
        Generic plotting function for the systems ODEs. Needs a list of
        concentrations and equilibrium constants (as floats), respectively.
        '''
        if self.times != []:
            self.odeTimes = np.linspace(0, 1.05*self.times[-1], 1000)
        else:
            self.odeTimes = np.linspace(0, 100, 1000)
        self._solution = spi.odeint(self.function, y0=c, args=tuple(k), 
                t=self.odeTimes)
        self._odeSolConvert()
        for n, cpd in enumerate(self.cpds):
            if color == False:
                plt.plot(self.odeTimes, self.odeData[n], 'k-')
            elif color == True:
                plt.plot(self.odeTimes, self.odeData[n], '-', label=cpd)
    
    ##### FITTING METHODS #####

    def _residTotal(self, pars):
        '''
        Generate and return one list of all the residuals.

        This function generates residuals by comparing the differences between
        the ODE and the actual data only when the ODE compound name
        (self.cpds) is one of the data set names (keys of self.useData).
        '''
        # Make a temporary parameter set so the original is unaffected.
        guesses = self.params['guess'].copy()
        mask = guesses.notna()
        guesses.loc[mask] = pars
        guesses.loc[~mask] = self.params.loc[~mask, 'fix']
        conc = guesses.filter(regex=r'^(?!k\d+$)')
        ks = guesses.filter(regex=r'^k\d+$')
        times = list(self.data.iloc[:,0])
        
        # Run the ODE simulation.
        self._solution = spi.odeint(self.function, y0=list(conc), t=times,
                args=tuple(ks))

        # Make the return np.array of residuals
        res = self.data[conc.index].values - self._solution
        return res.flatten()

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
        cpds = self.params.filter(regex=r'^(?!k\d+$)', axis=0)
        self.residuals = self.data.copy()
        self.residuals[cpds.index] = res_temp

        # Chi squared
        self.chisq = (info["fvec"]**2).sum()

        # R-squared
        y_ave = raw.mean()
        yave_diff = ((raw - y_ave)**2).sum()
        self.rsq = 1.0 - (self.chisq/yave_diff)

        # Akaike's Information Criterion. The use of this parameter is based
        # on the following reference, which is specific to chemical kinetics
        # fitting: Chem. Mater. 2009, 21, 4468-4479 The above reference gives
        # a number of additional references that are not directly related to
        # chemical kinetics.
        lenp = len(p)
        lenfvec = len(info["fvec"])
        self.aic = lenfvec*np.log(self.chisq) + 2*lenp + \
                ( 2*lenp*(lenp + 1) )/(lenfvec - lenp - 1)

        # Calculate parameter errors. Make sure that parameter errors don't
        # get set for fixed values
        self.dof = lenfvec - lenp
        sqcsdof = np.sqrt(self.chisq/self.dof)
        sigma = np.sqrt(np.diag(cov))*sqcsdof
        self.params['error'] = np.nan
        self.params.loc[mask, 'error'] = sigma


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

