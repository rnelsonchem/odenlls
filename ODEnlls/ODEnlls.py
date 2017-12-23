#!/usr/bin/env python
'''
Kinetics fitting object for using ordinary differential equations for
non-linear least squares fitting of chemical kinetics data.  Version 9 -
October 2010
'''
import pickle

import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as spo 
import scipy.integrate as spi 
import pandas as pd


class ODEnlls():
    def __init__(self):
        '''
        Initalize the object. Set data/ODE normalization, make a data
        dictionary, make a time list, set the figure number, and turn on
        interactive plotting for ipython.
        '''
        self.__version__ = 9.0
        self.useNorm = False
        self.times = []
        self.figNum = 1
        plt.ion()

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

    def rxnFile(self, filename):
        '''
        Takes a file name of fundamental reaction steps and breaks them up
        into lists of ODEs, compounds, reactions, equilibrium constants, and
        initial concentrations.

        The reaction file must have one reaction per line. Reaction components
        must be separated by '+'; stoichiometry can be included using '*'
        (e.g. 2*A for two equivalents of A); irreversible reactions are
        denoted by '->'; reversible reactions are denoted by '='.
        '''
        self.rxnFileName = filename
        self.rxnFileComments = []
        ks = []
        cpds = []
        self.odes = []
        self.rxns = []
        count = 1

        fileobj = open(self.rxnFileName)
        for rxn in fileobj:
            if rxn.isspace(): continue
            elif rxn[0] == '#': 
                self.rxnFileComments.append( rxn.strip() )
                continue
            rxn = rxn.strip()
            self.rxns.append(rxn)
            try:
                ktemp, vtemp, rtemp = self._rxnRate(rxn, count=count)
            except:
                print("There was an error with your reaction file!")
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


    ##### DATA MODIFICATION METHODS #####
                
    def _normalize(self, alist):
        '''
        Normalization function... This needs updating. rcn 10.01.10
        '''
        minval = min(alist)
        maxval = max(alist)
        return ((alist - minval)/(maxval - minval)) 

    def _odeSolConvert(self):
        '''
        Convert the ODE simulation solution into usable np.arrays.

        The ODE simulation gives the solution as an np.array for each time
        point, but plotting is easier with a each solution as an individual
        np.array.
        '''
        self.odeData = []
        for n in range(len(self._solution[0])):
            temp = self._solution[:,n]
            if self.useNorm == True:
                temp = self._normalize(temp)
            self.odeData.append(temp)

    def _paramConvert(self, params=None):
        '''
        Check the concentration and equilibrium constant guesses for fixed
        values (strings) and make new output lists of floats instead. If a
        params list is given, use those values for the non-fixed values.
        '''
        k_temp = []
        c_temp = []
        if type(params) != type(None): params = list(params)

        for c in self.cGuess:
            if isinstance(c, str):
                c_temp.append(float(c))
            else:
                if params == None: c_temp.append(float(c))
                else: c_temp.append(params.pop(0))

        for k in self.kGuess:
            if isinstance(k, str):
                k_temp.append(float(k))
            else:
                if params == None: k_temp.append(float(k))
                else: k_temp.append(params.pop(0))

        return c_temp, k_temp
    
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
        vals_shape = self.data.iloc[:,1:].values.shape
        res_temp = info["fvec"].reshape(vals_shape)
        cpds = self.params.filter(regex=r'^(?!k\d+$)', axis=0)
        self.residuals = self.data.copy()
        self.residuals[cpds.index] = res_temp

        # Chi squared
        self.chisq = (info["fvec"]**2).sum()

#        self.sigma = sigma; self.chiSq = chiSq
#        self.rSqrd = rSqrd; self.dof = dof
        
#        # Get the fit residuals and the average y value for the fitted data
#        # set (for statistical purposes).
#        self.resid = {}
#        num = 0
#        y_ave = 0
#        for k in self.useData:
#            if k in self.cpds:
#                self.resid[k] = info["fvec"][num:len(self.useData[k])+num]
#                num += len(self.useData[k])
#                y_ave += sum(self.useData[k])
#        y_ave /= len(info["fvec"])
#        
#        # Calculate the sum of squared differences of the data y values from
#        # the average y value, again for statistical purposes.
#        yave_diff = 0
#        for k in self.useData:
#            if k in self.cpds:
#                yave_diff += sum((self.useData[k]-y_ave)**2)
#
#        # Calculate the statistical parameters for the fit.
#        chiSq = sum(info["fvec"]*info["fvec"])
#        rSqrd = 1.0 - (chiSq/yave_diff)
#
#        # This is an additional statistical parameter for the fit called
#        # Akaike's Information Criterion. The use of this parameter is based
#        # on the following reference, which is specific to chemical kinetics
#        # fitting: Chem. Mater. 2009, 21, 4468-4479 The above reference gives
#        # a number of additional references that are not directly related to
#        # chemical kinetics.
#        self.aic = len(info["fvec"])*np.log(chiSq) + 2*len(p) + \
#                ( 2*len(p)*(len(p) + 1) )/(len(info["fvec"]) - len(p) - 1)
#
#        dof = len(info["fvec"]) - len(p)
#        sigma = np.array([np.sqrt(cov[i,i])*np.sqrt(chiSq/dof) for i in
#            range(len(p))])
#        

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

    ##### FILE OUTPUT METHODS #####

    def saveFigure(self, fname, type='pdf'):
        '''
        A simple convenience function for the matplotlib.pyplot.savefig
        function. Only takes a filename (fname) and filetype (type, default
        pdf) at present.
        '''
        plt.savefig(fname, format=type)

    def writeOut(self, fname):
        '''
        Simple command to write out the fit parameters to a text file (fname).
        '''
        out = open(fname, 'w')
        out.write('Fitted Parameters:\n')
        for n, z in enumerate(self.fitPar):
            out.write('%.4e +- %.4e\n' % (z, self.sigma[n]))
        out.write('Chisq: %.4e\n' % self.chiSq)
        out.write('Rsqrd: %.4e\n' % self.rSqrd)
        out.close()

    def dataWrite(self, fname):
        '''
        Write out the current data to a given file (fname).

        The data will be writen as comma separated values, and the first row
        will be column labels. Note: This fuction uses the last plotted/fitted
        data set. This is importatant for normalization. You will have to
        replot the data after changing the useNorm flag, then run this
        function to get the updated data.
        '''
        out = open(fname, 'w')
        out.write('Time,')
        for key in self.useData:
            out.write('%s,' % key)
        out.write('\n')
        for n, time in enumerate(self.times):
            out.write('%.3f,' % time)
            for key in self.useData:
                out.write('%.3f,' % self.useData[key][n])
            out.write('\n')
        out.close()

    def odeWrite(self, fname):
        '''
        Write out the current ode data to a given file (fname).

        The data will be writen as comma separated values, and the first row
        will be column labels. Note: This fuction uses the last plotted/fitted
        data set. This is importatant for normalization and fitting. You will
        have to replot the data after changing the useNorm flag, then run
        this function to get the updated data. Also, if this is run directly
        after fitData(), the ODE data you will get will only be for the input
        times, not a smooth line.
        '''
        out = open(fname, 'w')
        out.write('Time,')
        for key in self.cpds:
            out.write('%s,' % key)
        out.write('\n')
        for n, time in enumerate(self.odeTimes):
            out.write('%.3f,' % time)
            for m in range(len(self.odeData)):
                out.write('%.3f,' % self.odeData[m][n])
            out.write('\n')
        out.close()

    def saveState(self, fname):
        '''
        Save the current state of the ODEnlls object to a pickle file.

        Arguments:
        fname = File name for the pickled data.
        '''

        f = open(fname, 'wb')
        p = pickle.Pickler(f)
        # General parameters
        p.dump(self.__version__)
        p.dump(self.figNum)
        p.dump(self.useNorm)
        
        # The rxn file parameters
        try:
            p.dump(self.rxnFileName)
            p.dump(self.rxnFileComments)
            p.dump(self.kGuess)
            p.dump(self.cGuess)
            p.dump(self.cpds)
            p.dump(self.odes)
            p.dump(self.rxns)
            p.dump(self.functString)
        except:
            p.dump("No reaction file.")

        # Data file values.

        try:
            p.dump(self.dataFileName)
            p.dump(self.dataFileComments)
            p.dump(self.inData)
            p.dump(self.useData)
            p.dump(self.times)
        except:
            p.dump("No data file.")

        # ODE values

        try:
            p.dump(self.odeTimes)
            p.dump(self.odeData)
        except:
            p.dump("No ODE values")

        # Fit values

        try:
            p.dump(self.aic)
            p.dump(self.fitPar)
            p.dump(self.sigma)
            p.dump(self.chiSq)
            p.dump(self.rSqrd)
            p.dump(self.dof)
        except:
            p.dump("No Fit values")

        f.close()

    def loadState(self, fname):
        '''
        Recover the values from a previous instance of the class.

        Arguments: fname = This is the filename of a pickle file generated
        using the saveState function.
        '''

        fin = open(fname, 'rb')
        p = pickle.Unpickler(fin)

        test_version = p.load()
        if test_version != self.__version__:
            warn = 'Warning, this saved state was generated with ' 
            warn += 'ODEnlls version %.1f!'
            print(warn % (test_version,))
            print('Current verison of ODEnlls is %.1f!' % (self.__version__,))
            return False

        self.figNum = p.load()
        self.useNorm = p.load()

        next = p.load()
        if next != "No reaction file.":
            self.rxnFileName = next
            self.rxnFileComments = p.load()
            self.kGuess = p.load()
            self.cGuess = p.load()
            self.cpds = p.load()
            self.odes = p.load()
            self.rxns = p.load()
            self.functString = p.load()
            exec(self.functString)
            self.function = f

        next = p.load()
        if next != "No data file.":
            self.dataFileName = next
            self.dataFileComments = p.load()
            self.inData = p.load()
            self.useData = p.load()
            self.times = p.load()

        next = p.load()
        if next != "No ODE values":
            self.odeTimes = next
            self.odeData = p.load()
        
        next = p.load()
        if next != "No Fit values":
            self.aic = next
            self.fitPar = p.load()
            self.sigma = p.load()
            self.chiSq = p.load()
            self.rSqrd = p.load()
            self.dof = p.load()

        fin.close()

# This function will calculate the Akaike Information Criteria parameter given
# the sum of squared residuals (SS), the number of data points (N), and the
# number of parameters (K).

def aicCalc(SS, N, K):
    return N*np.log(SS) + 2*K + (2*K*(K + 1)) / (N - K - 1)

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

if __name__ == '__main__':
#    x = ODEnlls()
#    x.useNorm = True
#    x.dataFile('audata.txt')
#    x.rxnFile('aurxn.txt')
#    x.cGuess = ['1', '0', '0']
#    x.kGuess = [0.005, 0.005, 0.0005, 0.01, 0.01]
#    x.displayPlot('guess')
#    y = ODEnlls()
#    y.figNum = 2
#    y.dataFile('prodata.txt')
#    y.rxnFile('prorxn.txt')
#    y.fitData()
#    y.displayPlot('fit')
#    y.saveState('junk.pickle')
#    z = ODEnlls()
#    z.loadState('junk.pickle')
    print('Ready to run.')

# Check: http://wiki.sagemath.org/sage_matlab for getting Tk to work.
#SAGE_ROOT/local/lib/python/site-packages/sage/gsl/ode.pyx

#T = ode_solver()

#
#T.function=f
#
#T.y_0=[1,0,0.000]
#
#T.ode_solve(t_span=[0,10], params=(1.0,1.0,5.,0.001), num_points=100)
#
#T.plot_solution(i=0, filename='A.png')
#T.plot_solution(i=1, filename='B.png')
#T.plot_solution(i=2, filename='C.png')
#
