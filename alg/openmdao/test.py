import openmdao.api as om
import numpy as np
# from tqdm import tqdm
import random
from paretoset import paretoset
from eq import *
from mosa_driver import MOSADriver

class EqusComp(om.ExplicitComponent):
    def setup(self):
        if circuit=='neg':
            self.add_input('alpha', val=0.0)
            self.add_input('n', val=0.0)
            self.add_output('S_alpha', val=0.0)
            self.add_output('S_n', val=0.0)

        elif circuit=='posneg':
            self.add_input('betax', val=0.0)
            self.add_input('betay', val=0.0)
            self.add_input('n', val=0.0)
            self.add_output(labels[choice1], val=0.0)
            self.add_output(labels[choice2], val=0.0)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if circuit=='neg':
            alpha = float(inputs['alpha'])
            n = float(inputs['n'])
            xss = ssfinder(alpha, n)

            if np.isnan(xss) or xss <= 0:
                outputs['S_alpha'] = 1e6
                outputs['S_n'] = 1e6

            else:
                S_alpha = S_alpha_xss_analytic(xss, alpha, n)
                S_n = S_n_xss_analytic(xss, alpha, n)
                outputs['S_alpha'] = S_alpha
                outputs['S_n'] = S_n

        elif circuit=='posneg':
            betax = float(inputs['betax'])
            betay = float(inputs['betay'])
            n = float(inputs['n'])
            xss, yss = ssfinder(betax, betay, n)

            if np.isnan(xss) or xss <= 0 or np.isnan(yss) or yss <= 0:
                outputs[labels[choice1]] = 1e6
                outputs[labels[choice2]] = 1e6

            else:
                S1, S2 = senpair(xss, yss, betax, betay, n, choice1, choice2)
                outputs[labels[choice1]] = S1
                outputs[labels[choice2]] = S2

class SingleRunComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('run_index', types=int)
        self.options.declare('circuit', types=str)
        self.options.declare('choice1', default=None)
        self.options.declare('choice2', default=None)
        self.options.declare('labels', default=None)

    def setup(self):
        self.add_output('f1_vals', shape_by_conn=True)
        self.add_output('f2_vals', shape_by_conn=True)

    def compute(self, inputs, outputs):
        run_index = self.options['run_index']
        circuit = self.options['circuit']
        choice1 = self.options['choice1']
        choice2 = self.options['choice2']
        labels = self.options['labels']

        random.seed(run_index)

        prob = om.Problem(reports=False)
        model = prob.model

        model.add_subsystem('equs', EqusComp(), promotes=['*'])

        if circuit == 'neg':
            driver = MOSADriver(params=['alpha', 'n'])
        elif circuit == 'posneg':
            driver = MOSADriver(params=['betax', 'betay', 'n'])

        prob.driver = driver
        prob.driver.declare_coloring()

        if circuit == 'neg':
            model.add_design_var('alpha', lower=0.01, upper=50.0)
            model.add_design_var('n', lower=0.01, upper=10.0)
            model.add_objective('S_alpha')
            model.add_objective('S_n')

        elif circuit == 'posneg':
            model.add_design_var('betax', lower=0.01, upper=50.0)
            model.add_design_var('betay', lower=0.01, upper=50.0)
            model.add_design_var('n', lower=0.01, upper=10.0)
            model.add_objective(labels[choice1])
            model.add_objective(labels[choice2])

        prob.setup()
        prob.run_driver()

        pareto = prob.driver.pareto_front

        if circuit == 'neg':
            pareto = sorted(pareto, key=lambda p: p['f']['S_alpha'])
            f1_vals = [p['f']['S_alpha'] for p in pareto]
            f2_vals = [p['f']['S_n'] for p in pareto]

        elif circuit == 'posneg':
            pareto = sorted(pareto, key=lambda p: p['f'][labels[choice1]])
            f1_vals = [p['f'][labels[choice1]] for p in pareto]
            f2_vals = [p['f'][labels[choice2]] for p in pareto]

        outputs['f1_vals'] = np.array(f1_vals)
        outputs['f2_vals'] = np.array(f2_vals)

class ParallelRunsGroup(om.Group):
    def initialize(self):
        self.options.declare('runs', types=int)
        self.options.declare('circuit', types=str)
        self.options.declare('choice1', default=None)
        self.options.declare('choice2', default=None)
        self.options.declare('labels', default=None)

    def setup(self):
        runs = self.options['runs']
        circuit = self.options['circuit']
        choice1 = self.options['choice1']
        choice2 = self.options['choice2']
        labels = self.options['labels']

        par = self.add_subsystem('parallel', om.ParallelGroup(), promotes=['*'])

        for i in range(runs):
            par.add_subsystem(f'run_{i}',
                              SingleRunComp(run_index=i, circuit=circuit,
                                            choice1=choice1, choice2=choice2, labels=labels))

def custom_run(runs=10):
    prob = om.Problem()
    prob.model = ParallelRunsGroup(runs=runs, circuit='neg') 

    prob.setup()
    prob.run_model()

    f1_all = []
    f2_all = []
    for i in range(4):
        f1_all.extend(prob.get_val(f'run_{i}.f1_vals'))
        f2_all.extend(prob.get_val(f'run_{i}.f2_vals'))

    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)

    mask = paretoset(np.column_stack((f1_all, f2_all)), sense=['min', 'min'])
    pareto_front = np.column_stack((f1_all[mask], f2_all[mask]))

    return pareto_front