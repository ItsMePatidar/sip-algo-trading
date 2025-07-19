# optimization/genetic_optimizer.py
"""
Genetic Algorithm Optimizer for SIP Strategy Parameters

This module implements a genetic algorithm to optimize strategy parameters
by evolving populations of parameter sets through selection, crossover, and mutation.

The genetic algorithm is particularly useful for:
- Multi-dimensional parameter optimization
- Non-linear parameter relationships
- Avoiding local optima
- Exploring complex parameter spaces
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import BaseSIPStrategy, StrategyConfig
from strategies import create_strategy

logger = logging.getLogger(__name__)

@dataclass
class ParameterBounds:
    """Define bounds and type for a parameter"""
    name: str
    min_value: float
    max_value: float
    param_type: str = 'float'  # 'float', 'int', 'choice'
    choices: List[Any] = field(default_factory=list)
    step: Optional[float] = None  # For discrete parameters
    
    def generate_random_value(self):
        """Generate a random value within bounds"""
        if self.param_type == 'choice':
            return random.choice(self.choices)
        elif self.param_type == 'int':
            return random.randint(int(self.min_value), int(self.max_value))
        elif self.param_type == 'float':
            if self.step:
                # Discrete float values
                n_steps = int((self.max_value - self.min_value) / self.step)
                step_index = random.randint(0, n_steps)
                return self.min_value + step_index * self.step
            else:
                # Continuous float values
                return random.uniform(self.min_value, self.max_value)
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")
    
    def clip_value(self, value):
        """Clip value to valid bounds"""
        if self.param_type == 'choice':
            return value if value in self.choices else random.choice(self.choices)
        elif self.param_type == 'int':
            return max(int(self.min_value), min(int(self.max_value), int(value)))
        elif self.param_type == 'float':
            clipped = max(self.min_value, min(self.max_value, value))
            if self.step:
                # Round to nearest step
                n_steps = round((clipped - self.min_value) / self.step)
                return self.min_value + n_steps * self.step
            return clipped
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")

@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population"""
    parameters: Dict[str, Any]
    fitness: float = 0.0
    evaluated: bool = False
    generation: int = 0
    
    def __post_init__(self):
        self.id = random.randint(100000, 999999)  # Unique ID for tracking

@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm"""
    population_size: int = 50
    num_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elitism_ratio: float = 0.1  # Keep top 10% of population
    early_stopping_patience: int = 20  # Stop if no improvement for N generations
    parallel_evaluation: bool = True
    max_workers: int = 4
    random_seed: Optional[int] = None

class FitnessFunction:
    """Base class for fitness evaluation"""
    
    def __init__(self, strategy_name: str, base_config: StrategyConfig, market_data: Dict[str, pd.DataFrame]):
        self.strategy_name = strategy_name
        self.base_config = base_config
        self.market_data = market_data
    
    def evaluate(self, parameters: Dict[str, Any]) -> float:
        """
        Evaluate fitness of parameter set
        
        Args:
            parameters: Dictionary of strategy parameters
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Create strategy with parameters
            config = copy.deepcopy(self.base_config)
            config.parameters.update(parameters)
            
            strategy = create_strategy(self.strategy_name, config)
            
            # Simulate strategy performance
            portfolio_value = 100000.0  # Starting value
            total_invested = 0.0
            portfolio_units = {symbol: 0 for symbol in strategy.symbols}
            
            # Get date range for simulation
            all_dates = set()
            for data in self.market_data.values():
                all_dates.update(data.index)
            
            simulation_dates = sorted(list(all_dates))
            
            # Simulate monthly executions
            last_execution = None
            monthly_returns = []
            
            for date in simulation_dates[::21]:  # Approximate monthly (every 21 trading days)
                try:
                    # Check if strategy should execute
                    if strategy.should_execute(date, last_execution):
                        # Get market data up to current date
                        current_market_data = {}
                        for symbol, data in self.market_data.items():
                            current_data = data[data.index <= date]
                            if not current_data.empty:
                                current_market_data[symbol] = current_data
                        
                        if not current_market_data:
                            continue
                        
                        # Create portfolio state
                        portfolio_state = {
                            'holdings': portfolio_units.copy(),
                            'cash': portfolio_value - sum(portfolio_units[s] * current_market_data[s].iloc[-1]['Close'] 
                                                        for s in portfolio_units if s in current_market_data),
                            'total_invested': total_invested
                        }
                        
                        # Make investment decisions
                        decisions = strategy.make_investment_decisions(
                            current_market_data, portfolio_state, date
                        )
                        
                        # Apply decisions
                        for decision in decisions:
                            if decision.amount > 0:
                                portfolio_units[decision.symbol] += decision.quantity
                                total_invested += decision.amount
                        
                        last_execution = date
                    
                    # Calculate current portfolio value
                    current_value = 0.0
                    for symbol, units in portfolio_units.items():
                        if symbol in self.market_data and not self.market_data[symbol][self.market_data[symbol].index <= date].empty:
                            current_price = self.market_data[symbol][self.market_data[symbol].index <= date].iloc[-1]['Close']
                            current_value += units * current_price
                    
                    # Calculate period return
                    if total_invested > 0:
                        period_return = (current_value - total_invested) / total_invested
                        monthly_returns.append(period_return)
                    
                except Exception as e:
                    logger.warning(f"Error in simulation at date {date}: {str(e)}")
                    continue
            
            # Calculate fitness metrics
            if not monthly_returns:
                return -1000.0  # Penalty for failed simulation
            
            returns_series = pd.Series(monthly_returns)
            
            # Calculate metrics
            total_return = returns_series.sum()
            volatility = returns_series.std() * np.sqrt(12)  # Annualized
            
            # Sharpe ratio (assuming 6% risk-free rate)
            risk_free_rate = 0.06
            sharpe_ratio = (total_return - risk_free_rate) / max(volatility, 0.01)
            
            # Maximum drawdown
            cumulative_returns = (1 + returns_series).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # Calmar ratio
            calmar_ratio = total_return / max(max_drawdown, 0.01)
            
            # Combined fitness score
            fitness = (
                sharpe_ratio * 0.4 +          # 40% weight on risk-adjusted return
                total_return * 0.3 +          # 30% weight on total return
                calmar_ratio * 0.2 +          # 20% weight on drawdown-adjusted return
                (1 / max(volatility, 0.01)) * 0.1  # 10% weight on low volatility
            )
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating fitness: {str(e)}")
            return -1000.0  # Penalty for failed evaluation

class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for strategy parameters
    """
    
    def __init__(self, 
                 parameter_bounds: List[ParameterBounds],
                 fitness_function: FitnessFunction,
                 config: GeneticAlgorithmConfig = None):
        """
        Initialize genetic optimizer
        
        Args:
            parameter_bounds: List of parameter bounds to optimize
            fitness_function: Function to evaluate parameter fitness
            config: Genetic algorithm configuration
        """
        self.parameter_bounds = {pb.name: pb for pb in parameter_bounds}
        self.fitness_function = fitness_function
        self.config = config or GeneticAlgorithmConfig()
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Optimization tracking
        self.population = []
        self.best_individual = None
        self.generation_stats = []
        self.evaluation_count = 0
        
        logger.info(f"Initialized genetic optimizer with {len(parameter_bounds)} parameters")
    
    def generate_random_individual(self, generation: int = 0) -> Individual:
        """Generate a random individual"""
        parameters = {}
        for name, bounds in self.parameter_bounds.items():
            parameters[name] = bounds.generate_random_value()
        
        return Individual(parameters=parameters, generation=generation)
    
    def create_initial_population(self) -> List[Individual]:
        """Create initial population"""
        logger.info(f"Creating initial population of {self.config.population_size} individuals")
        
        population = []
        for i in range(self.config.population_size):
            individual = self.generate_random_individual(generation=0)
            population.append(individual)
        
        return population
    
    def evaluate_individual(self, individual: Individual) -> Individual:
        """Evaluate fitness of an individual"""
        if not individual.evaluated:
            individual.fitness = self.fitness_function.evaluate(individual.parameters)
            individual.evaluated = True
            self.evaluation_count += 1
        
        return individual
    
    def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """Evaluate fitness of entire population"""
        unevaluated = [ind for ind in population if not ind.evaluated]
        
        if not unevaluated:
            return population
        
        logger.info(f"Evaluating {len(unevaluated)} individuals...")
        
        if self.config.parallel_evaluation and len(unevaluated) > 1:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_individual = {
                    executor.submit(self.fitness_function.evaluate, ind.parameters): ind 
                    for ind in unevaluated
                }
                
                for future in as_completed(future_to_individual):
                    individual = future_to_individual[future]
                    try:
                        individual.fitness = future.result()
                        individual.evaluated = True
                        self.evaluation_count += 1
                    except Exception as e:
                        logger.error(f"Error evaluating individual {individual.id}: {str(e)}")
                        individual.fitness = -1000.0
                        individual.evaluated = True
        else:
            # Sequential evaluation
            for individual in unevaluated:
                self.evaluate_individual(individual)
        
        return population
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual, generation: int) -> Tuple[Individual, Individual]:
        """Single-point crossover"""
        if random.random() > self.config.crossover_rate:
            # No crossover, return copies of parents
            child1 = Individual(parameters=parent1.parameters.copy(), generation=generation)
            child2 = Individual(parameters=parent2.parameters.copy(), generation=generation)
            return child1, child2
        
        # Perform crossover
        param_names = list(self.parameter_bounds.keys())
        crossover_point = random.randint(1, len(param_names) - 1)
        
        child1_params = {}
        child2_params = {}
        
        for i, param_name in enumerate(param_names):
            if i < crossover_point:
                child1_params[param_name] = parent1.parameters[param_name]
                child2_params[param_name] = parent2.parameters[param_name]
            else:
                child1_params[param_name] = parent2.parameters[param_name]
                child2_params[param_name] = parent1.parameters[param_name]
        
        child1 = Individual(parameters=child1_params, generation=generation)
        child2 = Individual(parameters=child2_params, generation=generation)
        
        return child1, child2
    
    def mutate(self, individual: Individual):
        """Mutate an individual"""
        for param_name, param_bounds in self.parameter_bounds.items():
            if random.random() < self.config.mutation_rate:
                if param_bounds.param_type == 'choice':
                    # Random choice mutation
                    individual.parameters[param_name] = param_bounds.generate_random_value()
                elif param_bounds.param_type in ['int', 'float']:
                    # Gaussian mutation
                    current_value = individual.parameters[param_name]
                    range_size = param_bounds.max_value - param_bounds.min_value
                    
                    # Standard deviation is 10% of range
                    std_dev = range_size * 0.1
                    
                    if param_bounds.param_type == 'int':
                        mutation = int(random.gauss(0, std_dev))
                        new_value = current_value + mutation
                    else:
                        mutation = random.gauss(0, std_dev)
                        new_value = current_value + mutation
                    
                    # Clip to bounds
                    individual.parameters[param_name] = param_bounds.clip_value(new_value)
        
        # Mark as unevaluated since parameters changed
        individual.evaluated = False
    
    def create_next_generation(self, population: List[Individual], generation: int) -> List[Individual]:
        """Create next generation through selection, crossover, and mutation"""
        
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Elitism: keep top individuals
        elite_count = int(self.config.population_size * self.config.elitism_ratio)
        next_generation = population[:elite_count].copy()
        
        # Generate offspring
        while len(next_generation) < self.config.population_size:
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2, generation)
            
            # Mutation
            self.mutate(child1)
            self.mutate(child2)
            
            # Add to next generation
            next_generation.append(child1)
            if len(next_generation) < self.config.population_size:
                next_generation.append(child2)
        
        # Ensure exact population size
        next_generation = next_generation[:self.config.population_size]
        
        return next_generation
    
    def calculate_generation_stats(self, population: List[Individual], generation: int) -> Dict[str, Any]:
        """Calculate statistics for current generation"""
        fitnesses = [ind.fitness for ind in population if ind.evaluated]
        
        if not fitnesses:
            return {}
        
        stats = {
            'generation': generation,
            'best_fitness': max(fitnesses),
            'average_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'std_fitness': np.std(fitnesses),
            'evaluations': self.evaluation_count
        }
        
        # Update best individual
        best_in_generation = max(population, key=lambda x: x.fitness if x.evaluated else -float('inf'))
        if self.best_individual is None or best_in_generation.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(best_in_generation)
        
        return stats
    
    def should_stop_early(self) -> bool:
        """Check if optimization should stop early"""
        if len(self.generation_stats) < self.config.early_stopping_patience:
            return False
        
        # Check if best fitness hasn't improved in patience generations
        recent_best = [stats['best_fitness'] for stats in self.generation_stats[-self.config.early_stopping_patience:]]
        
        # If all recent best fitnesses are the same (no improvement)
        return len(set(recent_best)) == 1
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization
        
        Returns:
            Optimization results including best parameters and statistics
        """
        logger.info("Starting genetic algorithm optimization...")
        
        # Create initial population
        self.population = self.create_initial_population()
        
        # Evaluate initial population
        self.population = self.evaluate_population(self.population)
        
        # Track statistics
        stats = self.calculate_generation_stats(self.population, 0)
        self.generation_stats.append(stats)
        
        logger.info(f"Generation 0: Best fitness = {stats['best_fitness']:.4f}, "
                   f"Average = {stats['average_fitness']:.4f}")
        
        # Evolution loop
        for generation in range(1, self.config.num_generations + 1):
            # Create next generation
            self.population = self.create_next_generation(self.population, generation)
            
            # Evaluate new individuals
            self.population = self.evaluate_population(self.population)
            
            # Calculate statistics
            stats = self.calculate_generation_stats(self.population, generation)
            self.generation_stats.append(stats)
            
            # Log progress
            if generation % 10 == 0 or generation == self.config.num_generations:
                logger.info(f"Generation {generation}: Best fitness = {stats['best_fitness']:.4f}, "
                           f"Average = {stats['average_fitness']:.4f}")
            
            # Check for early stopping
            if self.should_stop_early():
                logger.info(f"Early stopping at generation {generation}")
                break
        
        # Prepare results
        results = {
            'best_parameters': self.best_individual.parameters,
            'best_fitness': self.best_individual.fitness,
            'total_evaluations': self.evaluation_count,
            'generations_completed': len(self.generation_stats) - 1,
            'generation_stats': self.generation_stats,
            'convergence_generation': self._find_convergence_generation()
        }
        
        logger.info(f"Optimization completed. Best fitness: {results['best_fitness']:.4f}")
        logger.info(f"Best parameters: {results['best_parameters']}")
        
        return results
    
    def _find_convergence_generation(self) -> int:
        """Find the generation where the algorithm converged"""
        if not self.generation_stats:
            return 0
        
        best_fitness = max(stats['best_fitness'] for stats in self.generation_stats)
        
        for i, stats in enumerate(self.generation_stats):
            if abs(stats['best_fitness'] - best_fitness) < 1e-6:
                return i
        
        return len(self.generation_stats) - 1