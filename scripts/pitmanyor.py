import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import random


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
        
        # Chinese Restaurant Franchise representation
        # Tables at the base level (restaurant 0)
        self.base_tables = {}  # table_id -> dish (atom)
        self.base_table_counts = defaultdict(int)  # table_id -> number of customers
        self.base_num_tables = 0
        
        # Tables at group level (restaurants j = 1, ..., J)
        self.group_tables = defaultdict(dict)  # group_id -> {table_id -> base_table_id}
        self.group_table_counts = defaultdict(lambda: defaultdict(int))  # group_id -> {table_id -> count}
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
        Sample from the base measure H.
        
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
            sample = self._sample_single(group_id)
            samples.append(sample)
        
        return samples
    
    def _sample_single(self, group_id: int) -> Any:
        """
        Sample a single observation from group j's distribution.
        
        Uses the Chinese Restaurant Franchise seating process.
        
        Args:
            group_id: The group index j
            
        Returns:
            A single sample from G_j
        """
        # Get current state for this group
        group_table_counts = self.group_table_counts[group_id]
        num_customers_in_group = sum(group_table_counts.values())
        
        # Compute probabilities for existing tables and new table
        probs = []
        table_ids = []
        
        # Probability of sitting at existing table k
        for table_id, count in group_table_counts.items():
            prob = (count - self.d_j) / (self.theta_j + num_customers_in_group)
            probs.append(prob)
            table_ids.append(table_id)
        
        # Probability of sitting at a new table
        # This involves the base restaurant
        prob_new_table = (self.theta_j + self.d_j * self.group_num_tables[group_id]) / \
                         (self.theta_j + num_customers_in_group)
        probs.append(prob_new_table)
        table_ids.append(None)  # None indicates new table
        
        # Sample table assignment
        chosen_idx = np.random.choice(len(probs), p=np.array(probs) / sum(probs))
        
        if table_ids[chosen_idx] is None:
            # New table: need to choose dish from base restaurant
            dish = self._sample_from_base_restaurant()
            
            # Create new table in this group
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
        else:
            # Existing table
            table_id = table_ids[chosen_idx]
            base_table_id = self.group_tables[group_id][table_id]
            dish = self.base_tables[base_table_id]
            
            # Update counts
            self.group_table_counts[group_id][table_id] += 1
            self.base_table_counts[base_table_id] += 1
            self.dish_counts[dish] += 1
            
            # Track customer assignment
            customer_id = len(self.group_customers[group_id])
            self.group_customers[group_id].append((customer_id, table_id))
            
            return dish
    
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
        self.group_table_counts = defaultdict(lambda: defaultdict(int))
        self.group_num_tables = defaultdict(int)
        
        self.group_customers = defaultdict(list)
        self.dish_counts = defaultdict(int)
        self.dish_to_base_table = {}
        
        self._next_table_id = 0
        self._next_dish_id = 0
    
    def __repr__(self) -> str:
        """
        String representation of the HPYP.
        """
        return (f"HierarchicalPitmanYorProcess("
                f"d_0={self.d_0}, theta_0={self.theta_0}, "
                f"d_j={self.d_j}, theta_j={self.theta_j}, "
                f"num_groups={self.num_groups})")

