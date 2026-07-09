from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any
import random
from dataclasses import dataclass
from collections import defaultdict

class MOCOProblem(ABC):
    """Abstract base class for Multi-Objective Combinatorial Optimization problems"""
    
    @abstractmethod
    def evaluate(self, solution: Any) -> Tuple:
        """Evaluate a solution and return objective values"""
        pass
    
    @abstractmethod
    def random_solution(self) -> Any:
        """Generate a random valid solution"""
        pass
    
    @property
    @abstractmethod
    def num_objectives(self) -> int:
        """Return number of objectives"""
        pass

class MultiObjectiveTSP(MOCOProblem):
    def __init__(self, n_cities: int, m_objectives: int):
        """
        Multi-Objective TSP where each city has M sets of 2D coordinates.
        Each objective is calculated based on its corresponding set of coordinates.
        
        Args:
            n_cities: Number of cities
            m_objectives: Number of objectives (coordinate sets)
        """
        self.n_cities = n_cities
        self._m_objectives = m_objectives
        
        # Generate M sets of 2D coordinates for each city
        # Each coordinate in [0,1]^2 as specified
        self.coordinates = []
        for _ in range(m_objectives):
            # Create a set of coordinates for this objective
            coords = np.random.uniform(0, 1, size=(n_cities, 2))
            self.coordinates.append(coords)
        
        # Pre-compute distance matrices for algorithms that expect them
        self.distance_matrices = self._compute_all_distance_matrices()
        
        # Set individual distance matrices for compatibility with older code
        for i in range(m_objectives):
            setattr(self, f"distances{i+1}", self.distance_matrices[i])
    
    def _compute_all_distance_matrices(self):
        """Compute distance matrices for all objectives"""
        matrices = []
        for m in range(self._m_objectives):
            matrix = self._compute_distance_matrix(m)
            matrices.append(matrix)
        return matrices
    
    def _compute_distance_matrix(self, objective_idx):
        """Compute full distance matrix for the specified objective"""
        coord_set = self.coordinates[objective_idx]
        distances = np.zeros((self.n_cities, self.n_cities))
        
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    distances[i, j] = self._calculate_distance(coord_set, i, j)
        
        return distances
    
    def _calculate_distance(self, coord_set, city1, city2):
        """Calculate Euclidean distance between two cities using specified coordinate set"""
        x1, y1 = coord_set[city1]
        x2, y2 = coord_set[city2]
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def evaluate(self, solution: List[int]) -> Tuple:
        """
        Evaluate a TSP tour using M different coordinate sets
        Returns M objective values (one for each coordinate set)
        """
        if len(solution) != self.n_cities:
            raise ValueError("Solution must visit all cities exactly once")
            
        objective_values = []
        
        # Calculate distance for each objective (coordinate set)
        for m in range(self._m_objectives):
            coord_set = self.coordinates[m]
            total_dist = 0
            for i in range(self.n_cities):
                from_city = solution[i]
                to_city = solution[(i + 1) % self.n_cities]
                total_dist += self._calculate_distance(coord_set, from_city, to_city)
            objective_values.append(total_dist)
        
        return tuple(objective_values)

    def random_solution(self) -> List[int]:
        """Generate a random valid tour (permutation of cities)"""
        solution = list(range(self.n_cities))
        random.shuffle(solution)
        return solution
    
    @property
    def num_objectives(self) -> int:
        return self._m_objectives

class BiObjectiveTSP(MultiObjectiveTSP):
    """Bi-Objective TSP (special case of MultiObjectiveTSP with 2 objectives)"""
    def __init__(self, n_cities: int):
        super().__init__(n_cities, m_objectives=2)
        
        # For explicit backward compatibility
        self.distances1 = self.distance_matrices[0]
        self.distances2 = self.distance_matrices[1]
    
    def evaluate(self, solution: List[int]) -> Tuple[float, float]:
        return super().evaluate(solution)
    
    @property
    def num_objectives(self) -> int:
        return 2

class TriObjectiveTSP(MultiObjectiveTSP):
    """Tri-Objective TSP (special case of MultiObjectiveTSP with 3 objectives)"""
    def __init__(self, n_cities: int):
        super().__init__(n_cities, m_objectives=3)
        
        # For explicit backward compatibility
        self.distances1 = self.distance_matrices[0]
        self.distances2 = self.distance_matrices[1]
        self.distances3 = self.distance_matrices[2]
    
    def evaluate(self, solution: List[int]) -> Tuple[float, float, float]:
        return super().evaluate(solution)
    
    @property
    def num_objectives(self) -> int:
        return 3

class MultiObjectiveKnapsack(MOCOProblem):
    def __init__(self, n_items: int, n_objectives: int, capacity: float):
        """
        Multi-Objective Knapsack Problem
        
        Args:
            n_items: Number of items
            n_objectives: Number of objectives (value sets)
            capacity: Knapsack capacity
        """
        self.n_items = n_items
        self._n_objectives = n_objectives
        self.capacity = capacity
        
        # Generate random weights in [0,1]
        self.weights = np.random.uniform(0, 1, size=n_items)
        
        # Generate random values for each objective in [0,1]
        self.values = []
        for _ in range(n_objectives):
            self.values.append(np.random.uniform(0, 1, size=n_items))
        
        # For compatibility with algorithms expecting specific attribute names
        for i in range(n_objectives):
            setattr(self, f"values{i+1}", self.values[i])
    
    def evaluate(self, solution: List[int]) -> Tuple:
        """
        Evaluate a knapsack solution
        Returns a tuple of objective values (sums of values)
        """
        if len(solution) != self.n_items:
            raise ValueError("Solution length must match number of items")
        
        # Check capacity constraint
        total_weight = sum(w * x for w, x in zip(self.weights, solution))
        if total_weight > self.capacity:
            return tuple([-float('inf')] * self._n_objectives)
        
        # Calculate objective values (maximize sum of values)
        objective_values = []
        for obj_values in self.values:
            obj_value = sum(v * x for v, x in zip(obj_values, solution))
            objective_values.append(obj_value)
            
        return tuple(objective_values)
    
    def random_solution(self) -> List[int]:
        """Generate a random valid solution (binary vector)"""
        while True:
            solution = [random.randint(0, 1) for _ in range(self.n_items)]
            total_weight = sum(w * x for w, x in zip(self.weights, solution))
            if total_weight <= self.capacity:
                return solution
    
    @property
    def num_objectives(self) -> int:
        return self._n_objectives


@dataclass
class Customer:
    """Customer data for CVRP"""
    id: int
    x: float
    y: float
    demand: int  # Changed to int with values in {1,...,9}

class BiObjectiveCVRP(MOCOProblem):
    def __init__(self, n_customers: int, n_vehicles: int = None, vehicle_capacity: float = None):
        """
        Bi-Objective Capacitated Vehicle Routing Problem (SPECIFICATION-COMPLIANT)
        
        Problem Definition:
        - n customer nodes + 1 depot node
        - Each node: 2D coordinates
        - Each customer: integer demand
        - Two conflicting objectives:
          1. Total tour length (sum of all route distances)
          2. Makespan (length of longest route)
        
        Research Specification Compliance:
        - Standard sizes: n=20/50/100 for MOCVRP
        - Coordinates: uniformly sampled from [0,1]²
        - Demands: uniformly sampled from {1,...,9} (discrete uniform)
        - Vehicle capacity: 30/40/50 for MOCVRP20/50/100 (auto-set)
        
        Args:
            n_customers: Number of customers (standard: 20, 50, or 100)
            n_vehicles: Number of vehicles (optional, auto-computed if None)
            vehicle_capacity: Vehicle capacity (optional, auto-set based on n_customers if None)
                            Default: 30/40/50 for n=20/50/100 per specification
        
        Example:
            >>> # Standard benchmark instances (specification-compliant)
            >>> cvrp20 = BiObjectiveCVRP(n_customers=20)   # capacity=30
            >>> cvrp50 = BiObjectiveCVRP(n_customers=50)   # capacity=40
            >>> cvrp100 = BiObjectiveCVRP(n_customers=100) # capacity=50
        """
        self.n_customers = n_customers
        
        # Set capacity based on problem size per specification if not provided
        if vehicle_capacity is None:
            capacity_map = {20: 30, 50: 40, 100: 50}
            if n_customers in capacity_map:
                self.vehicle_capacity = capacity_map[n_customers]
            else:
                # For non-standard sizes, interpolate
                self.vehicle_capacity = 30 + (n_customers - 20) * 0.25
                print(f"Warning: Non-standard problem size n={n_customers}. "
                      f"Using interpolated capacity={self.vehicle_capacity:.1f}")
        else:
            self.vehicle_capacity = vehicle_capacity
        
        # Estimate number of vehicles if not provided
        if n_vehicles is None:
            # Estimate based on average demand (5) and capacity
            avg_demand = 5  # Average of {1,...,9}
            total_expected_demand = n_customers * avg_demand
            self.n_vehicles = max(2, int(np.ceil(total_expected_demand / self.vehicle_capacity)))
            # self.n_vehicles = n_customers  # Unlimited - one vehicle per customer is always feasible
        else:
            self.n_vehicles = n_vehicles
        
        # Generate random customer data
        self.customers = self._generate_customers()
        
        # Depot at (0.5, 0.5) in the middle of the [0,1]^2 space
        self.depot = Customer(id=0, x=0.5, y=0.5, demand=0)
        
        # Calculate distance matrix
        self.distances = self._calculate_distances()
        
        # For backward compatibility
        self.distances1 = self.distances  # For total distance objective
        self.distances2 = self.distances  # For makespan objective
    
    def _generate_customers(self) -> Dict[int, Customer]:
        """Generate customers with coordinates in [0,1]^2 and demands in {1,...,9}"""
        customers = {}
        for i in range(1, self.n_customers + 1):
            customers[i] = Customer(
                id=i,
                x=np.random.uniform(0, 1),  # Coordinates in [0,1]
                y=np.random.uniform(0, 1),  # Coordinates in [0,1]
                demand=np.random.randint(1, 10)  # Demands in {1,...,9} (10 is exclusive)
            )
        return customers
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate distance matrix between all nodes (depot + customers)"""
        n = self.n_customers + 1
        distances = np.zeros((n, n))
        all_customers = {0: self.depot, **self.customers}
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    customer1 = all_customers[i]
                    customer2 = all_customers[j]
                    distances[i,j] = np.sqrt(
                        (customer1.x - customer2.x)**2 + 
                        (customer1.y - customer2.y)**2
                    )
        return distances
        # Example of what this creates:
        # If we have 1 depot + 3 customers
        # 
        # The distance matrix would be:
        #           Depot  Cust1  Cust2  Cust3
        # Depot  [   0   ,  d01  ,  d02  ,  d03  ]
        # Cust1  [  d10  ,   0   ,  d12  ,  d13  ]  
        # Cust2  [  d20  ,  d21  ,   0   ,  d23  ]
        # Cust3  [  d30  ,  d31  ,  d32  ,   0   ]
        #
        # Where d01 = distance from depot to customer 1, etc.


    def evaluate(self, solution: List[List[int]]) -> Tuple[float, float]:
        """
        Evaluate a CVRP solution
        Returns (total_distance, makespan)
        
        Note: Each customer must be visited exactly once (no split delivery)
        """
        total_distance = 0
        route_lengths = []
        
        # Check that each customer is visited exactly once (no split delivery)
        visited = set()
        for route in solution:
            for customer in route:
                if customer in visited:
                    # Split delivery detected - invalid solution
                    return float('inf'), float('inf')
                if customer < 1 or customer > self.n_customers:
                    # Invalid customer ID
                    return float('inf'), float('inf')
                visited.add(customer)
        
        # Check all customers are visited
        if visited != set(range(1, self.n_customers + 1)):
            return float('inf'), float('inf')
        
        for route in solution:
            if not route:
                route_lengths.append(0)
                continue
                
            # Check capacity constraint
            route_demand = sum(self.customers[c].demand for c in route)
            if route_demand > self.vehicle_capacity:
                return float('inf'), float('inf')
            
            # Calculate route length
            route_length = 0
            current_pos = 0  # Start at depot
            
            for customer_id in route:
                # Add distance from current position to next customer
                route_length += self.distances[current_pos, customer_id]
                current_pos = customer_id
            
            # Return to depot
            route_length += self.distances[current_pos, 0]
            
            total_distance += route_length
            route_lengths.append(route_length)
        
        # Makespan is the length of the longest route
        makespan = max(route_lengths) if route_lengths else 0
        
        return total_distance, makespan
    
    def random_solution(self) -> List[List[int]]:
        """Generate a random valid CVRP solution"""
        unassigned = list(range(1, self.n_customers + 1))
        np.random.shuffle(unassigned)
        
        solution = []
        
        # Build routes greedily
        while unassigned:
            route = []
            route_demand = 0
            
            # Try to add customers to current route
            for customer in unassigned[:]:  # Iterate over copy
                customer_demand = self.customers[customer].demand
                if route_demand + customer_demand <= self.vehicle_capacity:
                    route.append(customer)
                    route_demand += customer_demand
                    unassigned.remove(customer)
            
            # If we couldn't add any customer to a new route, 
            # it means no single customer fits in capacity - impossible instance
            if not route and unassigned:
                # Force add the customer with smallest demand
                customer = min(unassigned, key=lambda c: self.customers[c].demand)
                route.append(customer)
                unassigned.remove(customer)
            
            if route:
                solution.append(route)
        
        return solution
    
    @property
    def num_objectives(self) -> int:
        return 2


def run_benchmark_example(problem: MOCOProblem, name: str):
    """Run example for any MOCO problem"""
    print(f"\nTesting {name}:")
    solution = problem.random_solution()
    print(f"Random Solution: {solution}")
    objectives = problem.evaluate(solution)
    print(f"Objectives: {objectives}")
    print(f"Number of objectives: {problem.num_objectives}")

def main():
    """Test all benchmark problems"""
    # Test Bi-Objective TSP
    bitsp = BiObjectiveTSP(n_cities=5)
    run_benchmark_example(bitsp, "Bi-Objective TSP")
    
    # Test Tri-Objective TSP
    tritsp = TriObjectiveTSP(n_cities=5)
    run_benchmark_example(tritsp, "Tri-Objective TSP")
    
    # Test Multi-Objective Knapsack
    mokp = MultiObjectiveKnapsack(n_items=10, n_objectives=2, capacity=5.0)
    run_benchmark_example(mokp, "Multi-Objective Knapsack")
    
    # Test Bi-Objective CVRP
    bicvrp = BiObjectiveCVRP(n_customers=10, n_vehicles=3, vehicle_capacity=30)
    run_benchmark_example(bicvrp, "Bi-Objective CVRP")

if __name__ == "__main__":
    main()