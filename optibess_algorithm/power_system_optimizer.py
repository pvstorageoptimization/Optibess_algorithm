import logging
import os
from abc import ABC, abstractmethod
import nevergrad as ng

from .constants import MAX_BATTERY_HOURS
from .financial_calculator import FinancialCalculator
from .output_calculator import OutputCalculator
from .power_storage import LithiumPowerStorage
from .producers import PvProducer

root_folder = os.path.dirname(os.path.abspath(__file__))


class PowerSystemOptimizer(ABC):

    def __init__(self, financial_calculator: FinancialCalculator | None = None, use_memory: bool = True,
                 max_aug_num: int = 6, initial_aug_num: int | None = None, budget: int = 2000):
        """
        initialize the simulation objects for the optimizer

        :param financial_calculator: calculator to use for objective function
        :param use_memory: whether to use memory to get score for already calculated values
        :param max_aug_num: the maximum number of augmentations the optimizer will try in a solution
        :param initial_aug_num: the number of augmentation in the initial guess
        :param budget: the number of simulation to use for optimization
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
        logging.info(candid)
        aug_table = candid[0]
        # print and save if this is the first solution with x augmentations
        if not self._first_aug_of_size[len(aug_table)]:
            logging.info('\x1b[6;30;42m' + f'first aug of size {len(aug_table)}' + '\x1b[0m')
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


class NevergradOptimizer(PowerSystemOptimizer):

    def __init__(self, financial_calculator: FinancialCalculator | None = None, use_memory: bool = True,
                 max_aug_num: int = 6, initial_aug_num: int | None = None, budget: int = 2000,
                 max_no_change_steps: int | None = None, min_change_size: float = 0.0001, verbosity: int = 2):
        """
        initialize the simulation objects for the optimizer

        :param financial_calculator: calculator to use for objective function
        :param use_memory: whether to use memory to get score for already calculated values
        :param max_aug_num: the maximum number of augmentations the optimizer will try in a solution
        :param initial_aug_num: the number of augmentation in the initial guess
        :param budget: the number of simulation to use for optimization
        :param max_no_change_steps: the maximum number of optimization step with no change before stopping (if none,
            does not use early stopping)
        :param min_change_size: the minimum change between steps to consider as a change for early stopping
        :param verbosity: print information from the optimization algorithm (0: None, 1: fitness values, 2: fitness
            values and recommendation)
        """
        super().__init__(financial_calculator, use_memory, max_aug_num, initial_aug_num, budget)
        self._max_no_change_steps = max_no_change_steps
        self._min_change_size = min_change_size
        self._no_change_steps = 0
        self._verbosity = verbosity

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
                 [ng.p.Scalar(init=100, lower=1, upper=200).set_integer_casting()]
        self._instru = ng.p.Tuple(*params)

    def _set_constraints(self):
        def constraints_gen(para):
            """
            check the difference between months is valid
            """
            return all([para[i + 1] - para[i] >= self._month_diff // 12 for i in range(0, self._max_aug_num - 1)])

        def check_battery_size(para):
            if not self._storage.check_battery_size(self.get_aug_table(para)):
                logging.warning(f"The battery is bigger than {MAX_BATTERY_HOURS} battery hour with augmentations "
                                f"{para}")
                return False
            return True

        self._instru.register_cheap_constraint(constraints_gen)
        self._instru.register_cheap_constraint(check_battery_size)

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
                                      ng.optimizers.Powell], [self._budget // 2, self._budget // 4])(
            parametrization=self._instru, budget=self._budget)
        # reduce number of tries for finding candidates
        opt._constraints_manager.max_trials = 10
        return opt

    def _register_callbacks(self, opt, progress_recorder=None):
        """
        create callbacks function for the optimizer and register them

        :param opt: the optimizer
        :param progress_recorder: an object that record the progress of the task (should have method set_progress)
        """
        # updates the progress of the optimizer if a recorder is provided
        def update_progress(optim, candide, value):
            progress_recorder.set_progress(optim.num_ask, optim.budget)

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
            recommendation = opt.minimize(self.minimize_objective, verbosity=self._verbosity)
        except ng.errors.NevergradEarlyStopping:
            recommendation = opt.provide_recommendation()
        return recommendation.value, recommendation.loss
