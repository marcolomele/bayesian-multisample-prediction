import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import random
from scipy.special import gammaln
from priors import UniformPrior, GammaPrior


def _defaultdict_int_factory():
    """Factory function for creating nested defaultdict(int). Picklable."""
    return defaultdict(int)


class HierarchicalPitmanYorProcess:
    """
    Hierarchical Pitman-Yor Process (HPYP) implementation.
    
    The HPYP models a collection of distributions G_j, where each G_j is drawn from
    a Pitman-Yor process with base distribution G_0, which itself is drawn from
    a Pitman-Yor process with base measure H.
    
    Formulation:
        G_0 ~ PY(d_0, θ_0, H)
        G_j ~ PY(d_j, θ_j, G_0)  for j = 1, ..., J
    
    where PY(d, θ, H) denotes a Pitman-Yor process with:
        - d: discount parameter (0 ≤ d < 1)
        - θ: strength parameter (θ > -d)
        - H: base measure
    
    This implementation uses the Chinese Restaurant Franchise (CRF) representation
    for efficient inference and sampling.
    """
    
    def __init__(
        self,
        d_0: float = 0.5,
        theta_0: float = 1.0,
        d_j: float = 0.5,
        theta_j: float = 1.0,
        base_measure: Optional[Any] = None,
        num_groups: int = 1
    ):
        """
        Initialize the Hierarchical Pitman-Yor Process.
        
        Args:
            d_0: Discount parameter for the base distribution G_0 (0 ≤ d_0 < 1)
            theta_0: Strength parameter for the base distribution G_0 (θ_0 > -d_0)
            d_j: Discount parameter for group distributions G_j (0 ≤ d_j < 1)
            theta_j: Strength parameter for group distributions G_j (θ_j > -d_j)
            base_measure: Base measure H. If None, uses a uniform distribution over integers.
            num_groups: Number of groups J
        """
        # Validate parameters
        if not (0 <= d_0 < 1):
            raise ValueError("d_0 must be in [0, 1)")
        if not (0 <= d_j < 1):
            raise ValueError("d_j must be in [0, 1)")
        if theta_0 <= -d_0:
            raise ValueError("theta_0 must be > -d_0")
        if theta_j <= -d_j:
            raise ValueError("theta_j must be > -d_j")
        
        self.d_0 = d_0
        self.theta_0 = theta_0
        self.d_j = d_j
        self.theta_j = theta_j
        self.num_groups = num_groups
        
        # Base measure H (if None, we'll use a simple discrete distribution)
        self.base_measure = base_measure
        
        # Priors for Bayesian inference
        self.d_0_prior = UniformPrior(0, 1)
        self.d_j_prior = UniformPrior(0, 1)
        self.theta_0_prior = GammaPrior(shape=300, rate=0.2)
        self.theta_j_prior = GammaPrior(shape=300, rate=0.2)
        
        # Adaptive proposal variances for M-H sampling
        self._proposal_d = 0.01
        self._proposal_theta = 10.0
        self._accept_counts = {'d_0': 0, 'd_j': 0, 'theta_0': 0, 'theta_j': 0}
        self._proposal_counts = {'d_0': 0, 'd_j': 0, 'theta_0': 0, 'theta_j': 0}
        
        # Chinese Restaurant Franchise representation
        # Tables at the base level (restaurant 0)
        self.base_tables = {}  # table_id -> dish (atom)
        self.base_table_counts = defaultdict(int)  # table_id -> number of customers
        self.base_num_tables = 0
        
        # Tables at group level (restaurants j = 1, ..., J)
        self.group_tables = defaultdict(dict)  # group_id -> {table_id -> base_table_id}
        self.group_table_counts = defaultdict(_defaultdict_int_factory)  # group_id -> {table_id -> count}
        self.group_num_tables = defaultdict(int)  # group_id -> number of tables
        
        # Customer assignments
        # For each group j, track which customers sit at which tables
        self.group_customers = defaultdict(list)  # group_id -> [(customer_id, table_id), ...]
        
        # Track unique dishes (atoms) and their assignments
        self.dish_counts = defaultdict(int)  # dish -> total count across all groups
        self.dish_to_base_table = {}  # dish -> base_table_id
        
        # Counter for generating new table/dish IDs
        self._next_table_id = 0
        self._next_dish_id = 0
        
    def _sample_from_base_measure(self) -> Any:
        """
        Sample from the sure H.
        
        Returns:
            A sample from H
        """
        if self.base_measure is None:
            # Default: return a new integer ID
            dish = self._next_dish_id
            self._next_dish_id += 1
            return dish
        elif callable(self.base_measure):
            return self.base_measure()
        else:
            # Assume it's a distribution object with a sample method
            return self.base_measure.sample()
    
    def _get_base_table_for_dish(self, dish: Any) -> int:
        """
        Get or create a base-level table serving the given dish.
        
        Args:
            dish: The dish (atom) to serve
            
        Returns:
            Base table ID
        """
        if dish in self.dish_to_base_table:
            return self.dish_to_base_table[dish]
        
        # Create new base table for this dish
        table_id = self._next_table_id
        self._next_table_id += 1
        
        self.base_tables[table_id] = dish
        self.base_table_counts[table_id] = 0
        self.base_num_tables += 1
        self.dish_to_base_table[dish] = table_id
        
        return table_id
    
    def sample(self, group_id: int, num_samples: int = 1) -> List[Any]:
        """
        Sample observations from group j's distribution G_j.
        
        Args:
            group_id: The group index j (0-indexed, but represents group j+1)
            num_samples: Number of samples to draw
            
        Returns:
            List of samples from G_j
        """
        if group_id < 0 or group_id >= self.num_groups:
            raise ValueError(f"group_id must be in [0, {self.num_groups-1}]")
        
        samples = []
        for _ in range(num_samples):
            sample = self._sample_single(group_id, dish=None)
            samples.append(sample)
        
        return samples
    
    def _sample_from_base_restaurant(self) -> Any:
        """
        Sample a dish from the base restaurant (G_0).
        
        Returns:
            A dish (atom) from G_0
        """
        total_base_customers = sum(self.base_table_counts.values())
        
        if total_base_customers == 0:
            # First customer: sample from base measure
            return self._sample_from_base_measure()
        
        # Compute probabilities for existing base tables and new table
        probs = []
        dishes = []
        
        # Probability of choosing existing base table k
        for table_id, count in self.base_table_counts.items():
            prob = (count - self.d_0) / (self.theta_0 + total_base_customers)
            probs.append(prob)
            dishes.append(self.base_tables[table_id])
        
        # Probability of new table (new dish from H)
        prob_new_table = (self.theta_0 + self.d_0 * self.base_num_tables) / \
                         (self.theta_0 + total_base_customers)
        probs.append(prob_new_table)
        dishes.append(None)  # None indicates new dish from H
        
        # Sample dish
        chosen_idx = np.random.choice(len(probs), p=np.array(probs) / sum(probs))
        
        if dishes[chosen_idx] is None:
            # New dish from base measure
            return self._sample_from_base_measure()
        else:
            return dishes[chosen_idx]
    
    def get_group_distribution(self, group_id: int) -> Dict[Any, float]:
        """
        Get the empirical distribution for group j.
        
        Args:
            group_id: The group index j
            
        Returns:
            Dictionary mapping dishes to their probabilities in G_j
        """
        if group_id < 0 or group_id >= self.num_groups:
            raise ValueError(f"group_id must be in [0, {self.num_groups-1}]")
        
        # Count occurrences of each dish in this group
        dish_counts = defaultdict(int)
        for customer_id, table_id in self.group_customers[group_id]:
            base_table_id = self.group_tables[group_id][table_id]
            dish = self.base_tables[base_table_id]
            dish_counts[dish] += 1
        
        total = sum(dish_counts.values())
        if total == 0:
            return {}
        
        # Normalize to get probabilities
        distribution = {dish: count / total for dish, count in dish_counts.items()}
        return distribution
    
    def get_base_distribution(self) -> Dict[Any, float]:
        """
        Get the empirical distribution for the base G_0.
        
        Returns:
            Dictionary mapping dishes to their probabilities in G_0
        """
        total = sum(self.base_table_counts.values())
        if total == 0:
            return {}
        
        # Count occurrences of each dish at base level
        dish_counts = defaultdict(int)
        for table_id, dish in self.base_tables.items():
            dish_counts[dish] += self.base_table_counts[table_id]
        
        # Normalize to get probabilities
        distribution = {dish: count / total for dish, count in dish_counts.items()}
        return distribution
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the HPYP.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            'base_num_tables': self.base_num_tables,
            'base_num_unique_dishes': len(self.base_tables),
            'base_total_customers': sum(self.base_table_counts.values()),
            'group_stats': {}
        }
        
        for group_id in range(self.num_groups):
            num_customers = len(self.group_customers[group_id])
            num_tables = self.group_num_tables[group_id]
            unique_dishes = len(set(
                self.base_tables[self.group_tables[group_id][table_id]]
                for table_id in self.group_tables[group_id]
            ))
            
            stats['group_stats'][group_id] = {
                'num_customers': num_customers,
                'num_tables': num_tables,
                'num_unique_dishes': unique_dishes,
                'avg_customers_per_table': num_customers / num_tables if num_tables > 0 else 0
            }
        
        return stats
    
    def reset(self):
        """
        Reset the HPYP to its initial state.
        """
        self.base_tables = {}
        self.base_table_counts = defaultdict(int)
        self.base_num_tables = 0
        
        self.group_tables = defaultdict(dict)
        self.group_table_counts = defaultdict(_defaultdict_int_factory)
        self.group_num_tables = defaultdict(int)
        
        self.group_customers = defaultdict(list)
        self.dish_counts = defaultdict(int)
        self.dish_to_base_table = {}
        
        self._next_table_id = 0
        self._next_dish_id = 0
        
        # Reset M-H acceptance tracking
        self._accept_counts = {'d_0': 0, 'd_j': 0, 'theta_0': 0, 'theta_j': 0}
        self._proposal_counts = {'d_0': 0, 'd_j': 0, 'theta_0': 0, 'theta_j': 0}
    
    def fit_from_data(
        self, 
        data_dict: Dict[int, List[str]], 
        num_iterations: int = 1000,
        burn_in: int = 500,
        update_params: bool = True,
        verbose: bool = True,
        iteration_callback: Optional[callable] = None
    ) -> Dict[str, List]:
        """
        Fit the HPYP from observed data using Gibbs sampling.
        
        Args:
            data_dict: {group_id: [observation, observation, ...]}
            num_iterations: Total number of Gibbs iterations
            burn_in: Number of burn-in iterations to discard
            update_params: Whether to update hyperparameters via M-H
            verbose: Whether to print progress
            iteration_callback: Optional callback function called after each iteration
            
        Returns:
            Dictionary with posterior samples after burn-in
        """
        from tqdm import tqdm
        
        # Reset the model
        self.reset()
        
        # Initialize by adding all observations
        if verbose:
            print("Initializing model with observed data...")
        
        for group_id, observations in data_dict.items():
            for obs in observations:
                _ = self._sample_single(group_id, dish=obs)
        
        # Store posterior samples
        posterior_samples = {
            'num_base_tables': [],
            'num_group_tables': defaultdict(list),
            'num_unique_dishes': [],
            'theta_0': [],
            'theta_j': [],
            'd_0': [],
            'd_j': []
        }
        
        # Run Gibbs sampler
        iterator = tqdm(range(num_iterations), desc="Gibbs sampling") if verbose else range(num_iterations)
        
        for iteration in iterator:
            # Resample table assignments for all customers
            # We need to rebuild the model state for each iteration
            for group_id, observations in data_dict.items():
                # Store current state
                current_tables = dict(self.group_tables[group_id])
                current_table_counts = dict(self.group_table_counts[group_id])
                
                # Clear this group's state
                self.group_tables[group_id] = {}
                self.group_table_counts[group_id] = defaultdict(int)
                self.group_num_tables[group_id] = 0
                self.group_customers[group_id] = []
                
                # Update base counts (decrement for removed customers)
                for table_id, base_table_id in current_tables.items():
                    count = current_table_counts[table_id]
                    dish = self.base_tables[base_table_id]
                    self.base_table_counts[base_table_id] -= count
                    self.dish_counts[dish] -= count
                    
                    # Remove empty base tables
                    if self.base_table_counts[base_table_id] == 0:
                        del self.base_table_counts[base_table_id]
                        del self.base_tables[base_table_id]
                        self.base_num_tables -= 1
                        if dish in self.dish_to_base_table:
                            del self.dish_to_base_table[dish]
                
                # Resample all customers for this group
                for obs in observations:
                    _ = self._sample_single(group_id, dish=obs)
            
            # Update hyperparameters via Metropolis-Hastings
            if update_params and iteration % 10 == 0:
                self._update_parameters_mh(data_dict, in_burnin=(iteration < burn_in))
            
            # Store samples after burn-in
            if iteration >= burn_in:
                posterior_samples['num_base_tables'].append(self.base_num_tables)
                posterior_samples['num_unique_dishes'].append(len(self.base_tables))
                posterior_samples['theta_0'].append(self.theta_0)
                posterior_samples['theta_j'].append(self.theta_j)
                posterior_samples['d_0'].append(self.d_0)
                posterior_samples['d_j'].append(self.d_j)
                
                for group_id in data_dict.keys():
                    posterior_samples['num_group_tables'][group_id].append(
                        self.group_num_tables[group_id]
                    )
            
            # Call callback after each iteration if provided
            if iteration_callback is not None:
                iteration_callback()
        
        if verbose:
            print(f"\nFitting complete. Posterior samples: {len(posterior_samples['theta_0'])}")
            print(f"Posterior mean parameters:")
            print(f"  θ_0 = {np.mean(posterior_samples['theta_0']):.2f}")
            print(f"  θ_j = {np.mean(posterior_samples['theta_j']):.2f}")
            print(f"  δ_0 = {np.mean(posterior_samples['d_0']):.4f}")
            print(f"  δ_j = {np.mean(posterior_samples['d_j']):.4f}")
        
        return posterior_samples
    
    def _sample_single(self, group_id: int, dish: Any = None) -> Any:
        """
        Sample a single observation from group j's distribution.
        Modified to optionally accept a specific dish (for fitting).
        
        Args:
            group_id: The group index j
            dish: If provided, assign this dish (for fitting from data)
            
        Returns:
            The sampled dish
        """
        # Get current state for this group
        group_table_counts = self.group_table_counts[group_id]
        num_customers_in_group = sum(group_table_counts.values())
        
        if dish is None:
            # Prediction mode: sample dish from distribution
            # Compute probabilities for existing tables and new table
            probs = []
            table_ids = []
            
            # Probability of sitting at existing table k
            for table_id, count in group_table_counts.items():
                prob = (count - self.d_j) / (self.theta_j + num_customers_in_group)
                probs.append(prob)
                table_ids.append(table_id)
            
            # Probability of sitting at a new table
            prob_new_table = (self.theta_j + self.d_j * self.group_num_tables[group_id]) / \
                             (self.theta_j + num_customers_in_group)
            probs.append(prob_new_table)
            table_ids.append(None)  # None indicates new table
            
            # Sample table assignment
            chosen_idx = np.random.choice(len(probs), p=np.array(probs) / sum(probs)) 
            
            if table_ids[chosen_idx] is None:
                # New table: choose dish from base restaurant
                dish = self._sample_from_base_restaurant()
                return self._add_customer_new_table(group_id, dish)
            else:
                # Existing table
                table_id = table_ids[chosen_idx]
                base_table_id = self.group_tables[group_id][table_id]
                dish = self.base_tables[base_table_id]
                return self._add_customer_existing_table(group_id, table_id, dish)
        else:
            # Fitting mode: we know the dish, compute probabilities for table assignments
            probs = []
            table_ids = []
            
            # Find tables serving this dish
            for table_id, count in group_table_counts.items():
                base_table_id = self.group_tables[group_id][table_id]
                if self.base_tables[base_table_id] == dish:
                    # Existing table with this dish
                    prob = (count - self.d_j) / (self.theta_j + num_customers_in_group)
                    probs.append(prob)
                    table_ids.append(table_id)
            
            # Probability of new table with this dish
            # This depends on base restaurant probability for this dish
            base_prob_dish = self._get_base_prob_for_dish(dish)
            prob_new_table = ((self.theta_j + self.d_j * self.group_num_tables[group_id]) / 
                             (self.theta_j + num_customers_in_group)) * base_prob_dish
            probs.append(prob_new_table)
            table_ids.append(None)
            
            # If no existing tables and very small prob, force new table
            if len(probs) == 1 and probs[0] < 1e-10:
                return self._add_customer_new_table(group_id, dish)
            
            # Sample table assignment
            probs_array = np.array(probs)
            if probs_array.sum() < 1e-10:
                # Fallback: create new table
                return self._add_customer_new_table(group_id, dish)
            
            chosen_idx = np.random.choice(len(probs), p=probs_array / probs_array.sum())
            
            if table_ids[chosen_idx] is None:
                # New table
                return self._add_customer_new_table(group_id, dish)
            else:
                # Existing table
                table_id = table_ids[chosen_idx]
                return self._add_customer_existing_table(group_id, table_id, dish)
    
    def _add_customer_new_table(self, group_id: int, dish: Any) -> Any:
        """Add a customer to a new table."""
        new_table_id = self._next_table_id
        self._next_table_id += 1
        
        # Get or create base table for this dish
        base_table_id = self._get_base_table_for_dish(dish)
        
        self.group_tables[group_id][new_table_id] = base_table_id
        self.group_table_counts[group_id][new_table_id] = 1
        self.group_num_tables[group_id] += 1
        self.base_table_counts[base_table_id] += 1
        self.dish_counts[dish] += 1
        
        # Track customer assignment
        customer_id = len(self.group_customers[group_id])
        self.group_customers[group_id].append((customer_id, new_table_id))
        
        return dish
    
    def _add_customer_existing_table(self, group_id: int, table_id: int, dish: Any) -> Any:
        """Add a customer to an existing table."""
        base_table_id = self.group_tables[group_id][table_id]
        
        # Update counts
        self.group_table_counts[group_id][table_id] += 1
        self.base_table_counts[base_table_id] += 1
        self.dish_counts[dish] += 1
        
        # Track customer assignment
        customer_id = len(self.group_customers[group_id])
        self.group_customers[group_id].append((customer_id, table_id))
        
        return dish
    
    def _remove_customer(self, group_id: int, customer_idx: int):
        """Remove a customer and update counts accordingly."""
        if customer_idx >= len(self.group_customers[group_id]):
            return
        
        _, table_id = self.group_customers[group_id][customer_idx]
        base_table_id = self.group_tables[group_id][table_id]
        dish = self.base_tables[base_table_id]
        
        # Decrement counts
        self.group_table_counts[group_id][table_id] -= 1
        self.base_table_counts[base_table_id] -= 1
        self.dish_counts[dish] -= 1
        
        # Remove empty tables
        if self.group_table_counts[group_id][table_id] == 0:
            del self.group_table_counts[group_id][table_id]
            del self.group_tables[group_id][table_id]
            self.group_num_tables[group_id] -= 1
        
        if self.base_table_counts[base_table_id] == 0:
            del self.base_table_counts[base_table_id]
            del self.base_tables[base_table_id]
            self.base_num_tables -= 1
            if dish in self.dish_to_base_table:
                del self.dish_to_base_table[dish]
    
    def _get_base_prob_for_dish(self, dish: Any) -> float:
        """Get the probability of a dish from the base restaurant."""
        total_base_customers = sum(self.base_table_counts.values())
        
        if total_base_customers == 0:
            return 1.0  # First customer
        
        # Check if dish exists at base level
        if dish in self.dish_to_base_table:
            base_table_id = self.dish_to_base_table[dish]
            count = self.base_table_counts.get(base_table_id, 0)
            return (count - self.d_0) / (self.theta_0 + total_base_customers)
        else:
            # New dish from base measure
            return (self.theta_0 + self.d_0 * self.base_num_tables) / (self.theta_0 + total_base_customers)
    
    def _compute_log_likelihood(self) -> float:
        """
        Compute log-likelihood of current CRF configuration.
        
        Based on the Chinese Restaurant Franchise representation.
        Uses Pochhammer symbol identities for efficient computation.
        
        Returns:
            Log-likelihood of current table configuration
        """
        log_lik = 0.0
        
        # Base level (G_0) contribution
        total_base_customers = sum(self.base_table_counts.values())
        
        if total_base_customers > 0 and self.base_num_tables > 0:
            # Numerator: Product_{r=1}^{k-1} (theta_0 + r * d_0)
            for r in range(1, self.base_num_tables):
                log_lik += np.log(self.theta_0 + r * self.d_0)
            
            # Denominator: (theta_0)_{n} = Gamma(theta_0 + n) / Gamma(theta_0)
            # Computed as Product_{i=0}^{n-1} (theta_0 + i)
            for i in range(total_base_customers):
                log_lik -= np.log(self.theta_0 + i)
            
            # Table size contributions: Product over tables of (1 - d_0)_{n_t - 1}
            for count in self.base_table_counts.values():
                if count > 1:
                    # (1-d_0)_{count-1} = (1-d_0) * (2-d_0) * ... * (count-1-d_0)
                    # = Gamma(count) / Gamma(1-d_0) * Gamma(count + 1 - d_0) / Gamma(count)
                    # For numerical stability, use: sum log(i - d_0) for i=1..count-1
                    for i in range(1, count):
                        log_lik += np.log(i - self.d_0)
        
        # Group level (G_j) contributions
        for group_id in range(self.num_groups):
            group_table_counts = self.group_table_counts[group_id]
            num_customers_in_group = sum(group_table_counts.values())
            num_tables_in_group = self.group_num_tables[group_id]
            
            if num_customers_in_group > 0 and num_tables_in_group > 0:
                # Numerator: Product_{r=1}^{l-1} (theta_j + r * d_j)
                for r in range(1, num_tables_in_group):
                    log_lik += np.log(self.theta_j + r * self.d_j)
                
                # Denominator: (theta_j)_{n_j}
                for i in range(num_customers_in_group):
                    log_lik -= np.log(self.theta_j + i)
                
                # Table size contributions
                for count in group_table_counts.values():
                    if count > 1:
                        for i in range(1, count):
                            log_lik += np.log(i - self.d_j)
        
        return log_lik
    
    def _compute_log_prior(self) -> float:
        """
        Compute log-prior probability for current hyperparameters.
        
        Priors:
            d_0, d_j ~ Uniform(0, 1)
            theta_0, theta_j ~ Gamma(shape=300, rate=0.2)
        
        Returns:
            Log-prior probability
        """
        return (self.d_0_prior.log_pdf(self.d_0) +
                self.d_j_prior.log_pdf(self.d_j) +
                self.theta_0_prior.log_pdf(self.theta_0) +
                self.theta_j_prior.log_pdf(self.theta_j))
    
    def _update_parameters_mh(self, data_dict: Dict[int, List[str]], in_burnin: bool = False):
        """
        Update hyperparameters using Metropolis-Hastings with proper acceptance ratios.
        
        Args:
            data_dict: Dictionary of observed data
            in_burnin: Whether currently in burn-in phase (for adaptive proposals)
        """
        # Compute current log-posterior
        log_post_current = self._compute_log_likelihood() + self._compute_log_prior()
        
        # Update d_0
        d_0_prop = self.d_0 + np.random.normal(0, self._proposal_d)
        self._proposal_counts['d_0'] += 1
        
        if 0 < d_0_prop < 1:
            # Temporarily update
            d_0_old = self.d_0
            self.d_0 = d_0_prop
            
            # Compute proposed log-posterior
            try:
                log_post_prop = self._compute_log_likelihood() + self._compute_log_prior()
                
                # Metropolis-Hastings acceptance
                log_alpha = min(0, log_post_prop - log_post_current)
                if np.log(np.random.random()) < log_alpha:
                    # Accept
                    log_post_current = log_post_prop
                    self._accept_counts['d_0'] += 1
                else:
                    # Reject
                    self.d_0 = d_0_old
            except (ValueError, FloatingPointError):
                # Reject on numerical errors
                self.d_0 = d_0_old
        
        # Update d_j
        d_j_prop = self.d_j + np.random.normal(0, self._proposal_d)
        self._proposal_counts['d_j'] += 1
        
        if 0 < d_j_prop < 1:
            d_j_old = self.d_j
            self.d_j = d_j_prop
            
            try:
                log_post_prop = self._compute_log_likelihood() + self._compute_log_prior()
                log_alpha = min(0, log_post_prop - log_post_current)
                
                if np.log(np.random.random()) < log_alpha:
                    log_post_current = log_post_prop
                    self._accept_counts['d_j'] += 1
                else:
                    self.d_j = d_j_old
            except (ValueError, FloatingPointError):
                self.d_j = d_j_old
        
        # Update theta_0
        theta_0_prop = self.theta_0 + np.random.normal(0, self._proposal_theta)
        self._proposal_counts['theta_0'] += 1
        
        if theta_0_prop > -self.d_0:
            theta_0_old = self.theta_0
            self.theta_0 = theta_0_prop
            
            try:
                log_post_prop = self._compute_log_likelihood() + self._compute_log_prior()
                log_alpha = min(0, log_post_prop - log_post_current)
                
                if np.log(np.random.random()) < log_alpha:
                    log_post_current = log_post_prop
                    self._accept_counts['theta_0'] += 1
                else:
                    self.theta_0 = theta_0_old
            except (ValueError, FloatingPointError):
                self.theta_0 = theta_0_old
        
        # Update theta_j
        theta_j_prop = self.theta_j + np.random.normal(0, self._proposal_theta)
        self._proposal_counts['theta_j'] += 1
        
        if theta_j_prop > -self.d_j:
            theta_j_old = self.theta_j
            self.theta_j = theta_j_prop
            
            try:
                log_post_prop = self._compute_log_likelihood() + self._compute_log_prior()
                log_alpha = min(0, log_post_prop - log_post_current)
                
                if np.log(np.random.random()) < log_alpha:
                    self._accept_counts['theta_j'] += 1
                else:
                    self.theta_j = theta_j_old
            except (ValueError, FloatingPointError):
                self.theta_j = theta_j_old
        
        # Adaptive tuning during burn-in (every 50 iterations)
        if in_burnin and self._proposal_counts['d_0'] % 50 == 0:
            # Adjust proposal variances to target 25-40% acceptance
            for param in ['d_0', 'd_j']:
                if self._proposal_counts[param] > 0:
                    accept_rate = self._accept_counts[param] / self._proposal_counts[param]
                    if accept_rate < 0.25:
                        self._proposal_d *= 0.9
                    elif accept_rate > 0.40:
                        self._proposal_d *= 1.1
            
            for param in ['theta_0', 'theta_j']:
                if self._proposal_counts[param] > 0:
                    accept_rate = self._accept_counts[param] / self._proposal_counts[param]
                    if accept_rate < 0.25:
                        self._proposal_theta *= 0.9
                    elif accept_rate > 0.40:
                        self._proposal_theta *= 1.1
    
    def sample_predictive(
        self,
        group_id: int,
        num_samples: int,
        observed_dishes: set
    ) -> Tuple[List[Any], Dict[str, int]]:
        """
        Generate predictive samples conditional on current state.
        
        Args:
            group_id: Which group to generate samples for
            num_samples: Number of samples to generate
            observed_dishes: Set of dishes observed in training data
            
        Returns:
            Tuple of:
            - List of sampled dishes
            - Dictionary with counts of new vs existing dishes
        """
        samples = []
        new_count = 0
        existing_count = 0
        
        for _ in range(num_samples):
            dish = self._sample_single(group_id, dish=None)
            samples.append(dish)
            
            if dish not in observed_dishes:
                new_count += 1
            else:
                existing_count += 1
        
        return samples, {'new': new_count, 'existing': existing_count}
    
    def get_state_snapshot(self) -> Dict:
        """Get current state of the model."""
        return {
            'base_tables': dict(self.base_tables),
            'base_table_counts': dict(self.base_table_counts),
            'base_num_tables': self.base_num_tables,
            'group_tables': {k: dict(v) for k, v in self.group_tables.items()},
            'group_table_counts': {k: dict(v) for k, v in self.group_table_counts.items()},
            'group_num_tables': dict(self.group_num_tables),
            'parameters': {
                'theta_0': self.theta_0,
                'theta_j': self.theta_j,
                'd_0': self.d_0,
                'd_j': self.d_j
            }
        }
    
    def copy(self) -> 'HierarchicalPitmanYorProcess':
        """
        Create a deep copy of the model.
        More efficient than copy.deepcopy() for nested defaultdicts.
        Optimized for speed by avoiding unnecessary conversions.
        """
        new_model = HierarchicalPitmanYorProcess(
            d_0=self.d_0,
            theta_0=self.theta_0,
            d_j=self.d_j,
            theta_j=self.theta_j,
            base_measure=self.base_measure,
            num_groups=self.num_groups
        )
        
        # Copy base level - use dict() constructor which is faster than dict() for defaultdict
        new_model.base_tables = dict(self.base_tables)
        new_model.base_table_counts = defaultdict(int, self.base_table_counts)
        new_model.base_num_tables = self.base_num_tables
        
        # Copy group level - avoid double conversion where possible
        new_model.group_tables = {k: dict(v) for k, v in self.group_tables.items()}
        new_model.group_table_counts = defaultdict(_defaultdict_int_factory)
        for group_id, counts in self.group_table_counts.items():
            # Direct assignment is faster than dict(counts) for defaultdict
            new_model.group_table_counts[group_id] = defaultdict(int, counts)
        new_model.group_num_tables = defaultdict(int, self.group_num_tables)
        
        # Copy customers - use list() for shallow copy (sufficient for tuples)
        new_model.group_customers = defaultdict(list, {k: list(v) for k, v in self.group_customers.items()})
        
        # Copy dish tracking
        new_model.dish_counts = defaultdict(int, self.dish_counts)
        new_model.dish_to_base_table = dict(self.dish_to_base_table)
        
        # Copy counters
        new_model._next_table_id = self._next_table_id
        new_model._next_dish_id = self._next_dish_id
        
        # Copy M-H proposal variances and acceptance tracking
        new_model._proposal_d = self._proposal_d
        new_model._proposal_theta = self._proposal_theta
        new_model._accept_counts = dict(self._accept_counts)
        new_model._proposal_counts = dict(self._proposal_counts)
        
        return new_model
    
    def __repr__(self) -> str:
        """
        String representation of the HPYP.
        """
        return (f"HierarchicalPitmanYorProcess("
                f"d_0={self.d_0}, theta_0={self.theta_0}, "
                f"d_j={self.d_j}, theta_j={self.theta_j}, "
                f"num_groups={self.num_groups})")