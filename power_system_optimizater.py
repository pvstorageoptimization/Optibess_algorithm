import os
import time
from abc import ABC, abstractmethod
import numpy as np
from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.constraints import and_, or_
from mystic.solvers import diffev2
import gradient_free_optimizers as gfo
import nevergrad as ng
import pygad as pg

from Optibess_algorithm.financial_calculator import FinancialCalculator
from Optibess_algorithm.output_calculator import OutputCalculator
from Optibess_algorithm.power_storage import LithiumPowerStorage
from Optibess_algorithm.producers import PvProducer

root_folder = os.path.dirname(os.path.abspath(__file__))


class PowerSystemOptimizer(ABC):

    def __init__(self, financial_calculator: FinancialCalculator = None, use_memory: bool = True, max_aug_num: int = 6,
                 initial_aug_num: int = None, budget: int = 2000):
        """
        initialize the simulation objects for the optimizer
        :param financial_calculator: calculator to use for objective function
        :param use_memory: whether to use memory to get score for already calculated values
        """
        # check inputs are in the expected range
        if max_aug_num < 1:
            raise ValueError("Number of maximum augmentations should be at least 1")
        if initial_aug_num is not None and initial_aug_num < 1:
            raise ValueError("Number of maximum augmentations should be at least 1")
        if budget < 1:
            raise ValueError("Optimization budget should be at least 1")

        # create financial calculator if not passed
        if financial_calculator is None:
            self._storage = LithiumPowerStorage(25, 5000, battery_hours=6, use_default_aug=True)

            prod = PvProducer(os.path.join(root_folder, "test.csv"), pv_peak_power=13000)

            self._output = OutputCalculator(25, 5000, prod, self._storage, save_all_results=False)
            self._financial_calculator = FinancialCalculator(self._output, 97)
        else:
            self._output = financial_calculator.output_calculator
            self._storage = self._output.power_storage
            self._financial_calculator = financial_calculator

        # set the default values for bound and constraints
        self._month_bound = (self._financial_calculator.num_of_years - 1) * 12
        self._first_entry_bound = self._output.aug_table[0, 1]
        self._month_diff = 36

        # memory initialization
        self._use_memory = use_memory
        if use_memory:
            self._memory = {}
            self._results = []

        # variables for augmentation choice and tracking
        self._max_aug_num = min((self._financial_calculator.num_of_years - 1) // 3 + 1, max_aug_num)
        if initial_aug_num is None:
            self._initial_aug_num = self._max_aug_num // 2
        else:
            self._initial_aug_num = min(initial_aug_num, self._max_aug_num)
        self._def_aug_diff = self._financial_calculator.num_of_years // (self._max_aug_num - 1)
        self._first_aug_of_size = {i: False for i in range(1, self._max_aug_num + 1)}

        self._budget = budget

    @abstractmethod
    def get_aug_table(self, arr) -> tuple[tuple[int, int], ...]:
        """
        create a preliminary augmentation table (As tuple of tuples) for the given collection
        :param arr: a collection (array-like, dict, tensor, etc.) containing the months and then the nuber of blocks for
            the augmentation table
        """
        pass

    def get_candid(self, arr):
        """
        create a candid for the optimization (a tuple containing: a tuple of tuples for the aug table and another number
        for the percent of producer used)
        :param arr - a collection (array-like, dict, tensor, etc.) containing info for a candid
        """
        aug_table = self.get_aug_table(arr)
        candid = (aug_table, arr[-1])
        return candid

    def maximize_objective(self, arr):
        """
        an objective function that returns the irr given an augmentation table (gives negative value for invalid
        augmentation table)
        :param arr: a collection (array-like, dict, tensor, etc.) containing info for a candid
        """
        # create candid solution from parameters
        candid = self.get_candid(arr)
        print(candid)
        aug_table = candid[0]
        # print and save if this is the first solution with x augmentations
        if not self._first_aug_of_size[len(aug_table)]:
            print('\x1b[6;30;42m' + f'first aug of size {len(aug_table)}' + '\x1b[0m')
            self._first_aug_of_size[len(aug_table)] = True
        # if optimizer uses memory and has the value for this solution in memory use the value in memory
        if self._use_memory:
            if candid in self._memory:
                self._results.append(self._memory[candid])
                return self._memory[candid]
        # if solution is not valid return big loss
        try:
            self._output.aug_table = aug_table
            self._output.producer_factor = candid[-1] / 100
        except ValueError:
            result = -100
        else:
            # perform simulation and irr calculation
            self._output.run()
            self._financial_calculator.output_calculator = self._output
            result = self._financial_calculator.get_irr()
        # save result to memory if the optimizer uses memory
        if self._use_memory:
            self._memory[candid] = result
            self._results.append(result)
        return result

    def minimize_objective(self, arr):
        """
        an objective function that returns minus the irr given an augmentation table (gives negative value for invalid
        augmentation table)
        :param arr: a collection (array-like, dict, tensor, etc.) containing the months and then the nuber of blocks for
            the augmentation table
        """
        return -self.maximize_objective(arr)

    @abstractmethod
    def _set_variables(self):
        """
        create the variables for the optimization, or bounds for them
        """
        pass

    @abstractmethod
    def _set_constraints(self):
        """
        create the constraints for the optimization
        """
        pass

    @abstractmethod
    def _optimize(self, progress_recorder=None):
        """
        runs the optimization itself (after parameters and constraints are created)
        :param progress_recorder: an object that record the progress of the task (should have method set_progress)
        """
        pass

    def run(self, progress_recorder=None):
        """
        runs optimization
        :param progress_recorder: an object that record the progress of the task (should have method set_progress)
        """
        self._set_variables()
        self._set_constraints()
        return self._optimize(progress_recorder)


class GFOOptimizer(PowerSystemOptimizer):
    def get_aug_table(self, arr: dict) -> tuple[tuple[int, int], ...]:
        aug_table = []
        num_of_aug = len(arr) // 2
        # if the augmentation date is smaller than the previous don't use it or the augmentations after it
        for i in range(1, num_of_aug + 1):
            if i == 1 or arr[f"x{i - 1}"] < arr[f"x{i}"]:
                # ignore augmentations with 0 blocks
                if arr[f"x{i + num_of_aug}"] > 0:
                    aug_table.append((arr[f"x{i}"], arr[f"x{i + num_of_aug}"]))
            else:
                break
        print(aug_table)
        return tuple(aug_table)

    def get_candid(self, arr):
        aug_table = self.get_aug_table(arr)
        candid = (aug_table, arr["x13"])
        return candid

    def _set_variables(self):
        # set months to be at most at the month bound. set the first augmentation to be at most 2 times the initial
        # value, and the other augmentations half the initial first value
        self._search_space = {
            "x1": np.arange(0, self._month_bound, 12),
            "x2": np.arange(0, self._month_bound, 12),
            "x3": np.arange(0, self._month_bound, 12),
            "x4": np.arange(0, self._month_bound, 12),
            "x5": np.arange(0, self._month_bound, 12),
            "x6": np.arange(0, self._month_bound, 12),
            "x7": np.arange(1, self._first_entry_bound * 2, 1),
            "x8": np.arange(0, self._first_entry_bound // 2, 1),
            "x9": np.arange(0, self._first_entry_bound // 2, 1),
            "x10": np.arange(0, self._first_entry_bound // 2, 1),
            "x11": np.arange(0, self._first_entry_bound // 2, 1),
            "x12": np.arange(0, self._first_entry_bound // 2, 1),
            "x13": np.arange(1, 100, 1),
        }

    def _set_constraints(self):
        def constraints_gen(para):
            """
            check the difference between months is valid
            """
            return all([para[f"x{i + 1}"] - para[f"x{i}"] >= self._month_diff for i in
                        range(1, 6)])

        self._constraints_list = [constraints_gen]

    def _optimize(self, progress_recorder=None):
        opt = gfo.DownhillSimplexOptimizer(self._search_space, constraints=self._constraints_list)
        opt.search(self.maximize_objective, n_iter=self._budget)
        return opt.search_data


class MysticOptimizer(PowerSystemOptimizer):

    def get_aug_table(self, arr) -> tuple[tuple[int, int], ...]:
        aug_table = []
        num_of_aug = len(arr) // 2
        # if the augmentation date is smaller than the previous don't use it or the augmentations after it
        for i in range(num_of_aug):
            # ignore augmentations with 0 blocks
            if i == 0 or arr[i - 1] < arr[i]:
                aug_table.append((arr[i], arr[i + num_of_aug]))
            else:
                break
        print(aug_table)
        return tuple(aug_table)

    def _set_variables(self):
        # set months to be at most at the month bound. set the first augmentation to be at most 2 times the initial
        # value, and the other augmentations half the initial first value
        self._bounds = [(0, self._month_bound) for _ in range(6)] + [(1, self._first_entry_bound * 5)] + \
                       [(1, self._first_entry_bound) for _ in range(5)]
        # use the default augmentation table (with 5 battery hours) as initial guess
        temp_aug_table = self._output.aug_table.copy()
        self._x0 = [temp_aug_table[0, 0], temp_aug_table[1, 0], temp_aug_table[2, 0], 0, 0, 0,
                    temp_aug_table[1, 0], temp_aug_table[1, 1], temp_aug_table[2, 1], 1, 1, 1]

    def _set_constraints(self):
        # check the difference between months is valid
        temp = [or_(generate_constraint(generate_solvers(simplify(f"x{i + 1}-x{i}<={self._month_bound}"))),
                    generate_constraint(generate_solvers(simplify(f"x{i + 1}==0"))))
                for i in range(1, 6)]
        cf = and_(*temp)
        self._constraints = and_(np.round, cf)

    def _optimize(self, progress_recorder=None):
        result = diffev2(self.minimize_objective, x0=self._x0, bounds=self._bounds, constraints=self._constraints,
                         npop=10, gtol=50, full_output=True)
        print(result)


class PyGadOptimizer(PowerSystemOptimizer):

    def __init__(self, financial_calculator: FinancialCalculator = None, use_memory: bool = True):
        super().__init__(financial_calculator, use_memory)
        self._best_solution = None
        self._best_score = None

    def get_aug_table(self, arr) -> tuple[tuple[int, int], ...]:
        aug_table = []
        num_of_aug = len(arr) // 2
        # if the augmentation date is smaller than the previous don't use it or the augmentations after it
        for i in range(num_of_aug):
            if i == 0 or arr[i - 1] < arr[i]:
                # ignore augmentations with 0 blocks
                if arr[i + num_of_aug] > 0:
                    aug_table.append((arr[i], arr[i + num_of_aug]))
            else:
                break
        print(aug_table)
        return tuple(aug_table)

    def fitness_func(self, ga_instance, sol, sol_idx):
        return self.maximize_objective(sol)

    def _set_variables(self):
        # set months to be at most at the month bound. set the first augmentation to be at most 2 times the initial
        # value, and the other augmentations half the initial first value
        self._gene_space = [range(0, int(self._month_bound), 12) for _ in range(6)] + \
                           [range(0, int(self._first_entry_bound * 2))] + \
                           [range(0, int(self._first_entry_bound // 2)) for _ in range(5)]
        # create a starting population by tweaking the value of the default augmentation table (with 5 battery hours)
        temp = round(self._first_entry_bound * 0.2)
        self._init_population = [
            [0, 96, 192, 0, 0, 0, self._first_entry_bound, temp, temp, 1, 1, 1],
            [0, 60, 120, 180, 240, 0, self._first_entry_bound, temp, temp, temp, temp, 1],
            [0, 96, 192, 0, 0, 0, self._first_entry_bound, temp // 2, temp // 2, 1, 1, 1],
            [0, 84, 168, 252, 0, 0, self._first_entry_bound, temp, temp, temp, 1, 1],
            [0, 108, 216, 0, 0, 0, self._first_entry_bound, temp, temp, 1, 1, 1],
            [0, 60, 120, 180, 240, 0, self._first_entry_bound, temp // 2, temp // 2, temp // 2, temp // 2, 1],
            [0, 48, 84, 120, 192, 252, self._first_entry_bound, temp, temp, temp, temp, temp],
            [0, 48, 84, 120, 192, 252, self._first_entry_bound, temp, temp, temp // 2, temp // 2, temp // 2],
            [0, 108, 216, 0, 0, 0, self._first_entry_bound, temp, temp // 2, 1, 1, 1],
            [0, 84, 168, 252, 0, 0, self._first_entry_bound, temp, temp, temp // 2, 1, 1],
            [0, 108, 216, 0, 0, 0, self._first_entry_bound, temp, temp, 1, 1, 1],
            [0, 72, 144, 192, 240, 0, self._first_entry_bound, temp, temp, temp // 2, temp // 2, 1],
            [0, 48, 108, 0, 0, 0, self._first_entry_bound, temp // 2, temp // 2, 1, 1, 1],
            [0, 84, 144, 216, 0, 0, self._first_entry_bound, temp, temp // 2, temp, 1, 1],
            [0, 48, 96, 156, 216, 252, self._first_entry_bound, temp, temp, temp, temp // 2, temp],
            [0, 36, 84, 180, 240, 0, self._first_entry_bound, temp // 2, temp // 2, temp, temp // 2, 1],
            [0, 60, 84, 120, 192, 240, self._first_entry_bound, temp, temp, temp, temp, temp // 2],
            [0, 48, 96, 144, 192, 252, self._first_entry_bound, temp, temp, temp // 2, temp // 2, temp // 2],
            [0, 108, 252, 0, 0, 0, self._first_entry_bound, temp, temp // 2, 1, 1, 1],
            [0, 84, 168, 216, 0, 0, self._first_entry_bound, temp, temp, temp // 2, 1, 1]
        ]

    def _set_constraints(self):
        pass

    def _optimize(self, progress_recorder=None):
        num_mating = 5

        def on_fitness(ga_inst, pop_fitness):
            # change number of mating parent after 1/3 of the optimization to the chosen number (instead of all the
            # population)
            if ga_inst.generations_completed < ga_inst.num_generations // 3:
                ga_inst.num_parents_mating = ga_inst.pop_size[0]
            else:
                ga_inst.num_parents_mating = num_mating

        def parent_selection_func(fitness, num_parents, ga_inst):
            # change strategy after 1/3 of the optimization
            if ga_inst.generations_completed < ga_inst.num_generations // 3:
                # half chosen randomly, half using steady state
                parents1, indices1 = ga_inst.random_selection(fitness, num_parents // 2)
                parents2, indices2 = ga_inst.steady_state_selection(fitness, num_parents - num_parents // 2)
                parents = np.concatenate((parents1, parents2))
                indices = np.concatenate((indices1, indices2))
                return parents, indices
            else:
                return ga_inst.steady_state_selection(fitness, num_parents)

        def crossover_func(parents, offspring_size, ga_inst):
            # change strategy after 1/3 of the optimization
            if ga_inst.generations_completed < ga_inst.num_generations // 3:
                if ga_inst.gene_type_single:
                    offspring = np.empty(offspring_size, dtype=ga_inst.gene_type[0])
                else:
                    offspring = np.empty(offspring_size, dtype=object)
                for i in range(offspring_size[0]):
                    # pick random parents for each offspring
                    last_gen_parents, _ = ga_inst.random_selection(ga_inst.last_generation_fitness, 2)
                    offspring[i, :] = ga_inst.single_point_crossover(last_gen_parents, (1, offspring_size[1]))[0, :]
                return offspring
            else:
                return ga_inst.single_point_crossover(parents, offspring_size)

        def on_generation(ga_inst):
            # save the best solution
            best_match_idx = np.where(ga_inst.last_generation_fitness == np.max(ga_inst.last_generation_fitness))[0][0]
            if self._best_score is None or ga_inst.last_generation_fitness[best_match_idx] > self._best_score:
                self._best_score = ga_inst.last_generation_fitness[best_match_idx]
                self._best_solution = ga_inst.population[best_match_idx, :].copy()
            print(f"generation: {ga_inst.generations_completed}")

        ga_instance = pg.GA(
            num_generations=300,
            num_parents_mating=num_mating,
            fitness_func=self.fitness_func,
            initial_population=self._init_population,
            # parent_selection_type=parent_selection_func,
            crossover_type=crossover_func,
            keep_elitism=5,
            gene_space=self._gene_space,
            gene_type=int,
            allow_duplicate_genes=False,
            on_generation=on_generation,
            # on_fitness=on_fitness,
            # stop_criteria="saturate_10"
        )
        ga_instance.run()
        # print(ga_instance.best_solution(), f"number of generations: {ga_instance.generations_completed}")
        return self._best_solution, self._best_score


class NevergradOptimizer(PowerSystemOptimizer):

    def __init__(self, financial_calculator: FinancialCalculator = None, use_memory: bool = True, max_aug_num: int = 6,
                 initial_aug_num: int = None, budget: int = 2000, max_no_change_steps: int = None,
                 min_change_size: float = 0.0001):
        """
        initialize the simulation objects for the optimizer
        :param financial_calculator: calculator to use for objective function
        :param use_memory: whether to use memory to get score for already calculated values
        """
        super().__init__(financial_calculator, use_memory, max_aug_num, initial_aug_num, budget)
        self._max_no_change_steps = max_no_change_steps
        self._min_change_size = min_change_size
        self._no_change_steps = 0

    def get_aug_table(self, arr) -> tuple[tuple[int, int], ...]:
        aug_table = []
        num_of_aug = self._max_aug_num
        # if the augmentation date is smaller than the previous don't use it or the augmentations after it
        for i in range(num_of_aug):
            if i == 0 or arr[i - 1] < arr[i]:
                # ignore augmentations with 0 blocks
                if arr[i + num_of_aug] > 0:
                    aug_table.append((arr[i] * 12, arr[i + num_of_aug]))
            else:
                break
        return tuple(aug_table)

    def _set_variables(self):
        # set months to be at most at the month bound. set the first augmentation to be at most 2 times the initial
        # value, and the other augmentations half the initial first value
        # use the default augmentation table (with 5 battery hours) as initial guess
        temp_aug_table = self._output.aug_table.copy()
        params = [ng.p.Scalar(init=min(i * self._def_aug_diff, self._output.num_of_years - 1), lower=0,
                              upper=self._month_bound // 12).set_integer_casting()
                  for i in range(self._max_aug_num)] + \
                 [ng.p.Scalar(init=temp_aug_table[0, 1], lower=0, upper=self._first_entry_bound * 2).
                  set_integer_casting()] + \
                 [ng.p.Scalar(init=temp_aug_table[1, 1], lower=0, upper=self._first_entry_bound // 2).
                  set_integer_casting() for _ in range(self._initial_aug_num - 1)] + \
                 [ng.p.Scalar(init=0, lower=0, upper=self._first_entry_bound // 2).set_integer_casting()
                  for _ in range(self._max_aug_num - self._initial_aug_num)] + \
                 [ng.p.Scalar(init=100, lower=1, upper=100).set_integer_casting()]
        self._instru = ng.p.Tuple(*params)

    def _set_constraints(self):
        def constraints_gen(para):
            """
            check the difference between months is valid
            """
            return all([para[i + 1] - para[i] >= self._month_diff // 12 for i in range(0, self._max_aug_num - 1)])

        self._constraints = constraints_gen
        self._instru.register_cheap_constraint(self._constraints)

    def _create_optimizer(self):
        """
        create the optimizer
        """
        # use CMAPara (more exploration) at the start (half), then use CMATuning (smaller steps) (quarter), then use
        # Powell (local method) (quarter)
        opt = ng.optimizers.Chaining([ng.optimizers.ParametrizedCMA(scale=0.8905, popsize_factor=8, elitist=True,
                                                                    diagonal=True,
                                                                    inopts={"integer_variables": range(13)}),
                                      ng.optimizers.ParametrizedCMA(scale=0.4847, popsize_factor=1, elitist=True,
                                                                    inopts={"integer_variables": range(13)}),
                                      ng.optimizers.Powell], [self._budget // 2, self._budget // 4]) \
            (parametrization=self._instru, budget=self._budget)
        return opt

    def _register_callbacks(self, opt, progress_recorder=None):
        """
        create callbacks function for the optimizer and register them
        :param opt: the optimizer
        :param progress_recorder: an object that record the progress of the task (should have method set_progress)
        """
        # updates the progress of the optimizer if a recorder is provided
        def update_progress(optim, candide, value):
            progress_recorder.set_progress(optim.num_ask)

        if progress_recorder:
            opt.register_callback("tell", update_progress)

        # add stopping criteria for best value not changing
        def no_change_stopping(optim):
            if len(self._results) >= 3 * self._max_no_change_steps:
                # get max of 3 last group of 10 result
                max1 = max(self._results[-(3 * self._max_no_change_steps):-(2 * self._max_no_change_steps)])
                max2 = max(self._results[-(2 * self._max_no_change_steps):-self._max_no_change_steps])
                max3 = max(self._results[-self._max_no_change_steps:])
                if (abs(abs(max1) - abs(max2)) < self._min_change_size and
                        abs(abs(max2) - abs(max3)) < self._min_change_size) and \
                        max(max1, max2, max3) <= max(self._results):
                    raise ng.errors.NevergradEarlyStopping(f"no change in max result in last 3 groups of"
                                                           f"{self._max_no_change_steps} steps")

        if self._max_no_change_steps is not None and self._use_memory:
            opt.register_callback("ask", no_change_stopping)

    def _optimize(self, progress_recorder=None):
        opt = self._create_optimizer()
        self._register_callbacks(opt, progress_recorder)

        try:
            recommendation = opt.minimize(self.minimize_objective, verbosity=2)
        except ng.errors.NevergradEarlyStopping:
            recommendation = opt.provide_recommendation()
        return recommendation.value, recommendation.loss


class NevergradDerivativeOptimizer(NevergradOptimizer):

    def _set_variables(self):
        temp_aug_table = self._output.aug_table.copy()
        params = [ng.p.Scalar(init=min(i * self._def_aug_diff, self._output.num_of_years - 1), lower=0,
                              upper=self._month_bound // 12).set_integer_casting()
                  for i in range(self._max_aug_num)] + \
                 [ng.p.Scalar(init=temp_aug_table[0, 1] // 2, lower=0, upper=self._first_entry_bound).
                  set_integer_casting()] + \
                 [ng.p.Scalar(init=temp_aug_table[1, 1], lower=0, upper=self._first_entry_bound // 2).
                  set_integer_casting() for _ in range(self._initial_aug_num - 1)] + \
                 [ng.p.Scalar(init=0, lower=0, upper=self._first_entry_bound // 2).set_integer_casting()
                  for _ in range(self._max_aug_num - self._initial_aug_num)] + \
                 [ng.p.Scalar(init=100, lower=1, upper=100).set_integer_casting()]
        self._instru = ng.p.Tuple(*params)

    def _create_optimizer(self):
        opt = ng.optimizers.CMAsmall(parametrization=self._instru, budget=self._budget)
        self._step = 0
        self._derivatives_calcs = 0
        self._derivatives = np.zeros(len(self._instru))
        self._last_parameters = None
        self._last_result = None
        return opt

    def _check_param_diff(self, arr, tol):
        """
        check if the difference between the given solution and the saved parameters is in the acceptable tolerance
        :param arr: the given solution
        :param tol: the tolerance
        """
        for i in (list(range(self._max_aug_num)) + ([len(arr) - 1])):
            # for augmentation years and producer factor check difference of each one
            if not (self._last_parameters[i] - tol * 10) < arr[i] < (self._last_parameters[i] + tol * 10):
                return False
        # check the total blocks in the solution is in the tolerance range compared to the total block in the saved
        # parameters
        temp = sum(self._last_parameters[self._max_aug_num: -1])
        if not temp * (1 - tol) < sum(arr[self._max_aug_num: -1]) < temp * (1 + tol):
            return False
        return True

    def maximize_objective(self, arr):
        """
        an objective function that returns the irr given an augmentation table (gives negative value for invalid
        augmentation table)
        :param arr: a collection (array-like, dict, tensor, etc.) containing info for a candid
        """
        candid = self.get_candid(arr)
        print(candid)
        aug_table = candid[0]
        if not self._first_aug_of_size[len(aug_table)]:
            print('\x1b[6;30;42m' + f'first aug of size {len(aug_table)}' + '\x1b[0m')
            self._first_aug_of_size[len(aug_table)] = True
        if self._use_memory and self._step % 100 != 0:
            if candid in self._memory:
                self._step += 1
                self._results.append(self._memory[candid])
                return self._memory[candid]
        # if the optimization step is not a multiple of 100, try using derivatives instead of full simulation
        # if the parameters are not within 10% of the parameters use for derivatives calculation, use the full
        # simulation to calculate the value
        if self._step % 100 != 0 and self._check_param_diff(arr, 0.15):
            self._step += 1
            self._derivatives_calcs += 1
            result = self._last_result + np.dot(np.array(arr) - self._last_parameters, self._derivatives)
            # print(np.array(arr) - self._last_parameters, self._derivatives,
            #       np.dot(np.array(arr) - self._last_parameters, self._derivatives), result)
            if self._use_memory:
                self._results.append(result)
            return result
        try:
            self._output.aug_table = aug_table
            self._output.producer_factor = candid[-1] / 100
        except ValueError:
            self._step += 1
            result = -100
        else:
            self._output.run()
            self._financial_calculator.output_calculator = self._output
            result = self._financial_calculator.get_irr()
            if self._step % 100 == 0:
                self._last_result = result
                self._last_parameters = np.array(arr)
                # calculate derivatives for each parameter
                for i in range(len(arr)):
                    temp = np.array(arr)
                    # change the ith value slightly
                    if i < self._max_aug_num:
                        if i < self._output.num_of_years - 1:
                            temp[i] += 1
                        else:
                            temp[i] -= 1
                    elif i == len(arr) - 1:
                        if temp[i] == 0:
                            temp[i] = 5
                        else:
                            temp[i] *= 0.95
                    else:
                        temp[i] = max(temp[i] * 1.1, temp[i] + 10)
                    temp_candid = self.get_candid(temp)
                    # use memory if result is available
                    if self._use_memory and temp_candid in self._memory:
                        temp_result = self._memory[temp_candid]
                    else:
                        # calculate the irr with the changed parameters
                        self._output.set_aug_table(temp_candid[0], False)
                        self._output.producer_factor = temp_candid[-1] / 100
                        self._output.run()
                        self._financial_calculator.output_calculator = self._output
                        temp_result = self._financial_calculator.get_irr()
                    if self._use_memory:
                        self._memory[temp_candid] = temp_result
                    # calculate the directional derivatives of the function
                    self._derivatives[i] = (temp_result - result) / (temp[i] - arr[i])
                    if np.isnan(self._derivatives[i]):
                        self._derivatives[i] = 0

        if self._use_memory:
            self._results.append(result)
            self._memory[candid] = result
        self._step += 1
        return result


def flatten_tuples(t):
    for x in t:
        if isinstance(x, tuple):
            yield from flatten_tuples(x)
        else:
            yield x


if __name__ == '__main__':
    storage = LithiumPowerStorage(25, 180000, use_default_aug=True)
    producer = PvProducer("../../test docs/Ramat hovav.csv", pv_peak_power=300000)
    output = OutputCalculator(25, 180000, producer, storage, save_all_results=False)
    test = FinancialCalculator(output, 2272, capex_per_land_unit=215000, capex_per_kwp=370, opex_per_kwp=5,
                               battery_capex_per_kwh=150, battery_opex_per_kwh=5, battery_connection_capex_per_kw=50,
                               battery_connection_opex_per_kw=0.5, fixed_capex=150000000, fixed_opex=10000000,
                               interest_rate=0.04, cpi=0.02, battery_cost_deg=0.05)

    start_time = time.time()
    optimizer = NevergradOptimizer(test, budget=2000)
    opt_output, res = optimizer.run()
    print(optimizer.get_candid(opt_output), res)
    print(optimizer._first_aug_of_size)
    print(f"Optimization took {time.time() - start_time} seconds")
    # print(f"Used {optimizer._derivatives_calcs} calculations with derivatives")
    # # write results to file
    # with open('c:/Users/user/Documents/solar optimization project/poc docs/optimization_results1.csv', 'w', newline='') \
    #         as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["parameters", "result"])
    #     for k, v in optimizer._memory.items():
    #         writer.writerow([tuple(flatten_tuples(k)), v])
